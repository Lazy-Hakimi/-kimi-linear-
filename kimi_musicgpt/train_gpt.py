"""Train KimiMusicGPT (design doc §9.1) on codec token streams.

Pipeline per batch: synthetic waveforms -> (frozen) Kimi-Music-Codec ->
(T, Q) token frames -> delay-pattern stream (+ structure/condition
encoders) -> shifted CE with per-slot weighting (§9.1.4).

Sequence length is bounded by ``--max-frames``: stream tokens =
frames * Q + 2 (e.g. 500 frames @ 8 codebooks = 4002 tokens).

Mixed precision (V100 / FP16)
-----------------------------
Two storage layouts are supported via ``--layout``:

* ``pure_fp16`` (**default**, the production path on a 16GB card): every
  weight is *stored* fp16 -- only normalisation weights (LayerNorm /
  ``Fp32LayerNorm`` / the backbone's flagged RMS norms / the prefix projector)
  stay fp32.  AdamW updates on fp16 weights round away at typical lrs, so the
  optimiser is :class:`~kimi_linear.fp16.KahanAdamW8bit` (8-bit quantised
  moments + a Kahan compensation tensor that carries the sub-ulp residual),
  and loss scaling is :class:`~kimi_linear.fp16.FP16GradScaler` (drives the
  fused non-finite-check-and-unscale kernel directly -- ``torch.amp``
  GradScaler refuses fp16 gradients).  The frozen codec is cast to fp16 for
  its inference-only encode pass.  This is the same machinery the 1.5B
  backbone was validated with (see the project acceptance report §2).
* ``amp`` (comparison arm): fp32 master weights + autocast + GradScaler --
  never ``model.half()``; needs roughly 2.2x the steady-state memory and is
  what the original 56M demo used.

Either way, ``scaler.unscale_`` happens **before** ``clip_grad_norm_`` --
otherwise the clip threshold is applied to scaled gradients and is effectively
random.

The reference KDA kernel up-casts to fp32 internally, so the delta-rule
mathematics is unaffected; the carried recurrent state ``S`` is additionally
kept in fp32 (see ``kimi_linear/kda.py``).

Example (V100, 1.5B -- measured peaks on a 16GB card):
    # <=128 frames: keep Kahan (protects fine updates), peak 14.91 GiB
    python -m kimi_musicgpt.train_gpt \
        --gpt-config kimi_musicgpt/configs/musicgpt_1p5b.json \
        --codec-config kimi_musicgpt/configs/codec_tiny_16k.json \
        --codec-ckpt runs/codec_tiny/codec_latest.pt \
        --songs 64 --steps 3000 --batch-size 1 --max-frames 128 \
        --device cuda --layout pure_fp16 --optimizer adamw8bit --out runs/gpt_1p5b
    # 200 frames: drop Kahan to fit (peak 13.21 GiB); fine at lr >= 1e-4
    python -m kimi_musicgpt.train_gpt ... --max-frames 200 --no-kahan
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from .codec import KimiMusicCodec
from .data import SynthMusicDataset
from .text_tokenizer import ByteLevelBPETokenizer
from .structure import create_structure_tokens
from .model import KimiMusicGPT, MusicGPTConfig
from .audio_io import write_wav
from kimi_linear.precision import (PRECISION_CHOICES, resolve_precision,
                                   precision_name, configure_cuda_backends,
                                   autocast, make_grad_scaler, describe_device)
from kimi_linear.fp16 import (KahanAdamW8bit, KahanAdafactor, FP16GradScaler,
                              cast_for_fp16_training, assert_fp16_layout,
                              memory_report, param_dtype_report)


# --------------------------------------------------------------------------- #
def collate_identity(batch):
    return batch


def encode_texts(bpe, texts, max_len, device):
    ids = [bpe.encode(t, add_bos=True, add_eos=True)[:max_len] for t in texts]
    L = max(len(x) for x in ids)
    out = torch.full((len(ids), L), bpe.pad_id, dtype=torch.long)
    for i, x in enumerate(ids):
        out[i, :len(x)] = torch.tensor(x)
    return out.to(device)


def build_batch(model, codec, bpe, items, max_frames, device, dtype=None):
    """items: list of dataset dicts -> model inputs (padded).

    The frozen codec is encoded on the same autocast path as training so the
    tokenisation step also gets the fp16 tensor cores.
    """
    sr = codec.sample_rate
    wavs, metas = [], []
    for it in items:
        w = it["waveform"]
        max_samples = max_frames * codec.hop_length
        if w.shape[-1] > max_samples:
            w = w[..., :max_samples]
        wavs.append(torch.nn.functional.pad(
            w, (0, max_samples - w.shape[-1])))
        metas.append(it)
    wav = torch.stack(wavs).to(device)
    with torch.no_grad(), autocast(device, dtype):
        tokens = codec.encode(wav)                      # [B, T, Q]

    B = len(items)
    streams, slots, frames = [], [], []
    structs = {"section": [], "bar": [], "beat": [], "chord": [], "tempo": []}
    for i, it in enumerate(metas):
        d = model.build_stream(tokens[i:i + 1])
        streams.append(d["stream"][0]); slots.append(d["slots"][0])
        frames.append(d["src_frames"][0])
        s = create_structure_tokens(tokens.shape[1], bpm=it["bpm"],
                                    chords=it["chords"],
                                    sections=it["sections"],
                                    frame_rate=round(codec.frame_rate))
        for k in structs:
            structs[k].append(s[k])

    L = max(x.shape[0] for x in streams)
    pad_id = model.pad_id
    stream = torch.full((B, L), pad_id, dtype=torch.long)
    slot = torch.zeros(B, L, dtype=torch.long)
    frame = torch.zeros(B, L, dtype=torch.long)
    for i in range(B):
        n = streams[i].shape[0]
        stream[i, :n] = streams[i]; slot[i, :n] = slots[i]
        frame[i, :n] = frames[i]
    struct = {k: torch.stack(v) for k, v in structs.items()}

    text_ids = encode_texts(bpe, [m["text"] for m in metas],
                            model.config.music["condition"]["text_max_len"],
                            "cpu")
    style_ids = torch.tensor([m["style_id"] for m in metas], dtype=torch.long)
    return {"stream": stream.to(device), "slots": slot.to(device),
            "src_frames": frame.to(device), "struct": struct,
            "text_ids": text_ids, "style_ids": style_ids,
            "tokens": tokens, "wav": wav}


def lr_at(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt-config", required=True)
    ap.add_argument("--codec-config", required=True)
    ap.add_argument("--codec-ckpt", default=None)
    ap.add_argument("--bpe", default=None, help="BPE vocab json; trained on the fly if absent")
    ap.add_argument("--songs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="hard cap on stream tokens; 0 = min(8192, config "
                         "max_position_embeddings)")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="peak lr; 4.2e-4 (doc §9.1.3) targets 1.5B/2M-token "
                         "batches -- 56M model with batch 2 needs ~1e-4")
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--optimizer", default=None,
                    choices=["adamw", "adafactor", "adamw8bit", "kahan-adafactor"],
                    help="default: adamw8bit for --layout pure_fp16, adamw for "
                         "--layout amp.  adamw8bit = KahanAdamW8bit (8-bit "
                         "quantised moments + Kahan compensation, ~1 B/param "
                         "state); kahan-adafactor = KahanAdafactor (rank-1 "
                         "second moment, lowest state, update-RMS-clipped)")
    ap.add_argument("--layout", default="pure_fp16",
                    choices=["pure_fp16", "amp"],
                    help="pure_fp16: weights stored fp16 (norms fp32) + "
                         "Kahan optimiser + FP16GradScaler -- the 16GB "
                         "production path.  amp: fp32 master weights + "
                         "autocast (comparison arm, ~2.2x memory)")
    ap.add_argument("--accum-steps", type=int, default=1,
                    help="gradient accumulation (effective batch = batch-size x accum)")
    ap.add_argument("--wav-dir", default=None,
                    help="production corpus: directory of .wav (+ optional "
                         "<name>.txt/.json sidecars); replaces the synthetic dataset")
    ap.add_argument("--resume", default=None,
                    help="checkpoint (.pt) to resume model+optimizer+step from")
    ap.add_argument("--no-kahan", dest="kahan", action="store_false",
                    default=True,
                    help="pure_fp16 only: drop the fp16 Kahan compensation "
                         "(saves 2 B/param = 3.2 GiB at 1.5B).  Fine at "
                         "lr >= 1e-4; sub-ulp updates round away at small lr")
    ap.add_argument("--amp-init-scale", type=float, default=2.0 ** 10,
                    help="initial GradScaler scale; the 65536 default spends its "
                         "first ~6 steps just halving on overflow (no learning)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--precision", default="auto", choices=list(PRECISION_CHOICES),
                    help="compute dtype for autocast; 'auto' -> fp16 on the V100 "
                         "(sm_70: no bf16/tf32 tensor cores), bf16 on Ampere+")
    ap.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="gradient checkpointing on backbone blocks; measured "
                         "on the V100 it is BOTH ~2.7x faster and ~7x smaller "
                         "than off (storing the KDA/MLA activation graphs "
                         "thrashes memory), so it stays on by default")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = args.device
    dtype = resolve_precision(args.precision, device)
    backend = configure_cuda_backends(device)
    print(describe_device(device, dtype) +
          f" | cudnn.benchmark={backend['cudnn_benchmark']} tf32={backend['tf32']}")

    with open(args.codec_config) as f:
        codec_cfg = json.load(f)
    codec = KimiMusicCodec(codec_cfg).to(device)
    if args.codec_ckpt:
        ckpt = torch.load(args.codec_ckpt, map_location=device)
        codec.load_state_dict(ckpt["codec"])
        print(f"loaded codec checkpoint (step {ckpt.get('step')})")
    codec.eval()
    for p in codec.parameters():
        p.requires_grad = False

    if args.wav_dir:
        from .data import WavMusicDataset
        clip_seconds = args.max_frames / round(codec.frame_rate)
        ds = WavMusicDataset(args.wav_dir, sample_rate=codec_cfg["sample_rate"],
                             frame_rate=round(codec.frame_rate),
                             clip_seconds=clip_seconds)
        print(f"wav corpus: {len(ds)} files from {args.wav_dir}")
    else:
        ds = SynthMusicDataset(args.songs, sample_rate=codec_cfg["sample_rate"],
                               frame_rate=round(codec.frame_rate), seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=collate_identity, drop_last=True, num_workers=0)

    # ---- text tokenizer: load or train on the corpus descriptions ------- #
    if args.bpe and os.path.exists(args.bpe):
        bpe = ByteLevelBPETokenizer.load(args.bpe)
    else:
        bpe = ByteLevelBPETokenizer()
        bpe.train((ds[i]["text"] for i in range(min(len(ds), 512))),
                  vocab_size=2000, min_pair_count=2)
        if args.bpe:
            bpe.save(args.bpe)
    print(f"bpe vocab: {bpe.vocab_size}")

    cfg = MusicGPTConfig.from_json(args.gpt_config)
    # stream budget: the delay pattern emits frames*Q + 2 tokens
    Q = cfg.music["n_codebooks"]
    token_cap = args.max_tokens or min(8192, cfg.backbone.max_position_embeddings)
    if args.max_frames * Q + 2 > token_cap:
        raise SystemExit(f"--max-frames {args.max_frames} -> "
                         f"{args.max_frames * Q + 2} tokens exceeds the "
                         f"{token_cap} token budget")
    model = KimiMusicGPT(cfg, text_vocab_size=bpe.vocab_size)
    if args.layout == "pure_fp16":
        # every weight fp16 except normalisation weights (fp32) -- the model
        # is built on CPU in fp32 first (~4.4 GB host RAM for 1.5B), cast,
        # then moved, so the GPU never sees an fp32 copy of the weights
        cast_for_fp16_training(model, torch.float16)
        bad = assert_fp16_layout(model, torch.float16)
        if bad:
            raise SystemExit("fp16 layout violated: " + "; ".join(bad[:5]))
        # the frozen codec only ever runs encode() (inference) -> fp16 weights
        codec.half()
    model = model.to(device)
    model.gradient_checkpointing = args.grad_checkpoint
    print(f"music gpt params: {model.num_parameters() / 1e6:.2f}M | "
          f"stream budget {args.max_frames * Q + 2}/{token_cap} tokens | "
          f"grad-checkpoint={args.grad_checkpoint} | layout={args.layout}")
    if args.layout == "pure_fp16":
        print("param dtypes (M): "
              + ", ".join(f"{k}={v:.1f}" for k, v in param_dtype_report(model).items()))

    amp = autocast(device, dtype)
    if args.layout == "pure_fp16":
        scaler = FP16GradScaler(init_scale=args.amp_init_scale, device=device)
    else:
        scaler = make_grad_scaler(device, dtype, init_scale=args.amp_init_scale)

    if args.optimizer is None:
        args.optimizer = "adamw8bit" if args.layout == "pure_fp16" else "adamw"
    pure_opt = args.optimizer in ("adamw8bit", "kahan-adafactor")
    if (args.layout == "pure_fp16") != pure_opt:
        raise SystemExit(f"--optimizer {args.optimizer} requires "
                         f"--layout {'amp' if args.layout == 'pure_fp16' else 'pure_fp16'}")
    if args.optimizer == "adamw8bit":
        opt = KahanAdamW8bit(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                             weight_decay=0.1, kahan=args.kahan)
    elif args.optimizer == "kahan-adafactor":
        opt = KahanAdafactor(model.parameters(), lr=args.lr, weight_decay=0.0,
                             kahan=args.kahan)
        # Adafactor RMS-clips updates internally (clip_threshold), so the
        # global grad-norm clip below is skipped for it (Adafactor's own
        # paper formulation replaces it).
    elif args.optimizer == "adafactor":
        from .optim import Adafactor
        opt = Adafactor(model.parameters(), lr=args.lr, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                weight_decay=0.1)
    if args.layout == "pure_fp16":
        ledger = memory_report(model, opt)
        print("memory ledger (GiB): "
              + ", ".join(f"{k}={v:.2f}" for k, v in ledger.items()))

    step, t0, nan_skips = 0, time.time(), 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        step = int(ck.get("step", 0))
        print(f"resumed from {args.resume} at step {step}")
    accum = max(1, args.accum_steps)
    micro, tokens_per_step = 0, 0
    model.train()
    while step < args.steps:
        for items in dl:
            if step >= args.steps:
                break
            batch = build_batch(model, codec, bpe, items, args.max_frames,
                                device, dtype)
            struct = {k: v.to(device) for k, v in batch["struct"].items()}
            model.train()
            with amp:
                prefix = model.encode_conditions(
                    batch["text_ids"].to(device),
                    style_ids=batch["style_ids"].to(device))
                out = model(batch["stream"], batch["slots"], struct,
                            batch["src_frames"], cond_prefix=prefix,
                            labels=batch["stream"])
            loss = out["loss"] / accum
            if not torch.isfinite(loss):
                # numerical guard: skip the batch instead of poisoning weights
                nan_skips += 1
                opt.zero_grad(set_to_none=True)
                print(f"step {step + 1:5d} | [WARN] non-finite loss, batch skipped "
                      f"(total {nan_skips})")
                continue
            (scaler.scale_loss(loss) if args.layout == "pure_fp16"
             else scaler.scale(loss)).backward()
            tokens_per_step += int((batch["stream"] != model.pad_id).sum())

            # optimizer update every `accum` micro-batches
            micro += 1
            if micro % accum != 0:
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
            if args.layout == "pure_fp16":
                if scaler.unscale_(model.parameters()):
                    # fp16 grad overflow -> drop the whole step (standard AMP
                    # protocol: unscale already divided, grads are discarded)
                    nan_skips += 1
                    opt.zero_grad(set_to_none=True)
                    scaler.update(found_inf=True)
                    print(f"step {step + 1:5d} | [WARN] non-finite grads after "
                          f"unscale, step skipped (total {nan_skips})")
                    continue
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) \
                    if args.optimizer == "adamw8bit" else torch.tensor(0.0)
                opt.step()
                scaler.update(found_inf=False)
            else:
                scaler.unscale_(opt)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) \
                    if args.optimizer == "adamw" else torch.tensor(0.0)
                scaler.step(opt)
                scaler.update()
            opt.zero_grad(set_to_none=True)

            step += 1
            if step % args.log_every == 0:
                peak = torch.cuda.max_memory_allocated() / 2**30 \
                    if device.startswith("cuda") else 0.0
                live = torch.cuda.memory_allocated() / 2**30 \
                    if device.startswith("cuda") else 0.0
                print(f"step {step:5d} | loss {loss.item() * accum:.4f} "
                      f"| gnorm {float(gnorm):.2f} "
                      f"| lr {opt.param_groups[0]['lr']:.2e} "
                      f"| scale {scaler.get_scale():.0f} "
                      f"| peak {peak:.2f} live {live:.2f} GiB "
                      f"| tok/upd {tokens_per_step} "
                      f"| {time.time() - t0:.0f}s")
                tokens_per_step = 0
            if step % args.save_every == 0 or step == args.steps:
                torch.save({"model": model.state_dict(),
                            "optimizer": opt.state_dict(),
                            "gpt_config_path": args.gpt_config,
                            "optimizer_name": args.optimizer,
                            "layout": args.layout,
                            "text_vocab_size": bpe.vocab_size, "step": step,
                            "precision": precision_name(dtype)},
                           os.path.join(args.out, "gpt_latest.pt"))
                print(f"  saved checkpoint at step {step}")
    print("done.")


if __name__ == "__main__":
    main()
