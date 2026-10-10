"""Train Kimi-Music-Codec (design doc §2.4) -- full GAN training loop.

Generator losses: waveform L1 + multi-scale STFT + log-mel + commitment +
codebook + adversarial (hinge) + feature matching.
Discriminator: multi-period + multi-scale STFT, hinge loss.

Mixed precision (V100 / FP16)
-----------------------------
``--precision auto`` resolves to **fp16** on the sm_70 V100 (the only
tensor-core path there) and to bf16 on Ampere+.  Weights stay fp32 (master
weights) and every matmul/conv runs under autocast; because fp16 gradients
underflow, both optimisers are wrapped in a ``GradScaler`` (one for G, one for
D -- they have very different gradient magnitudes, so sharing a scaler makes
one of them oscillate).

    scaler.scale(loss).backward()
    scaler.unscale_(opt)                 # back to fp32 grads
    clip_grad_norm_(...)                 # must happen AFTER unscale_
    scaler.step(opt); scaler.update()

Measurement losses (STFT / mel / L1 / feature matching) and the VQ bottleneck
are pinned to fp32 inside their own modules, so the training signal is
identical to the fp32 reference run.

Example (synthetic corpus, tiny codec, V100):
    python -m kimi_musicgpt.train_codec \
        --codec-config kimi_musicgpt/configs/codec_tiny_16k.json \
        --songs 64 --steps 2000 --batch-size 16 --device cuda \
        --precision auto --out runs/codec_tiny
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from .codec import (KimiMusicCodec, CodecDiscriminator, MultiScaleSTFTLoss,
                    MelSpectrogramLoss, discriminator_loss, generator_loss,
                    feature_matching_loss, waveform_l1)
from .data import SynthMusicDataset
from .audio_io import write_wav
from kimi_linear.precision import (PRECISION_CHOICES, resolve_precision,
                                   precision_name, configure_cuda_backends,
                                   autocast, make_grad_scaler, describe_device)


def collate_wave(batch, clip_samples):
    wavs = []
    for item in batch:
        w = item["waveform"]
        if w.shape[-1] >= clip_samples:
            wavs.append(w[..., :clip_samples])
        else:
            wavs.append(torch.nn.functional.pad(w, (0, clip_samples - w.shape[-1])))
    return torch.stack(wavs)


def _train_step(codec, disc, wav, lw, stft_loss, mel_loss, opt_g, opt_d,
                scaler_g, scaler_d, amp, clip_grad):
    """One GAN update (discriminator, then generator).

    Returns a metrics dict, or ``None`` when the batch was skipped because the
    generator loss was non-finite (fp16 overflow guard -- skipping keeps the
    weights clean instead of poisoning them with a NaN gradient).
    """
    # ------------------------- discriminator step -----------------------
    with torch.no_grad(), amp:
        recon_d = codec(wav)["recon"]
    with amp:
        real_logits, _ = disc(wav)
        fake_logits, _ = disc(recon_d)
    d_loss = discriminator_loss(real_logits, fake_logits)
    opt_d.zero_grad(set_to_none=True)
    scaler_d.scale(d_loss).backward()
    scaler_d.unscale_(opt_d)
    torch.nn.utils.clip_grad_norm_(disc.parameters(), clip_grad)
    scaler_d.step(opt_d)
    scaler_d.update()

    # ----------------------------- generator step -----------------------
    with amp:
        out = codec(wav)
    recon = out["recon"]
    with amp:
        fake_logits, fake_fmaps = disc(recon)
    with torch.no_grad(), amp:
        _, real_fmaps = disc(wav)

    l_wave = waveform_l1(recon, wav)
    l_stft = stft_loss(recon, wav)
    l_mel = mel_loss(recon, wav)
    l_commit = out["vq_losses"]["commitment"]
    l_cb = out["vq_losses"]["codebook"]
    l_adv = generator_loss(fake_logits)
    l_fm = feature_matching_loss(real_fmaps, fake_fmaps)
    g_loss = (lw["w_waveform"] * l_wave + lw["w_stft"] * l_stft
              + lw["w_mel"] * l_mel + lw["w_commitment"] * l_commit
              + l_cb + lw["w_adversarial"] * l_adv
              + lw["w_feature_matching"] * l_fm)

    if not torch.isfinite(g_loss):
        opt_g.zero_grad(set_to_none=True)
        return None
    opt_g.zero_grad(set_to_none=True)
    scaler_g.scale(g_loss).backward()
    scaler_g.unscale_(opt_g)
    gnorm = torch.nn.utils.clip_grad_norm_(codec.parameters(), clip_grad)
    scaler_g.step(opt_g)
    scaler_g.update()
    return {"g": float(g_loss), "d": float(d_loss), "wave": float(l_wave),
            "stft": float(l_stft), "mel": float(l_mel),
            "commit": float(l_commit), "adv": float(l_adv), "gnorm": float(gnorm)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec-config", required=True)
    ap.add_argument("--songs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--clip-seconds", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--precision", default="auto", choices=list(PRECISION_CHOICES),
                    help="compute dtype for autocast; 'auto' -> fp16 on the V100 "
                         "(sm_70 has no bf16/tf32 tensor cores), bf16 on Ampere+")
    ap.add_argument("--clip-grad", type=float, default=1.0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    with open(args.codec_config) as f:
        cfg = json.load(f)
    sr = cfg["sample_rate"]
    clip = int(args.clip_seconds * sr)
    device = args.device
    dtype = resolve_precision(args.precision, device)
    backend = configure_cuda_backends(device)
    print(describe_device(device, dtype) +
          f" | cudnn.benchmark={backend['cudnn_benchmark']} tf32={backend['tf32']}")

    codec = KimiMusicCodec(cfg).to(device)
    disc = CodecDiscriminator(cfg).to(device)
    lw = cfg["loss"]
    stft_loss = MultiScaleSTFTLoss(lw["stft_scales"]).to(device)
    mel_loss = MelSpectrogramLoss(sr, lw["mel_n_fft"], lw["mel_hop"],
                                  lw["mel_n_mels"]).to(device)

    ds = SynthMusicDataset(args.songs, sample_rate=sr,
                           frame_rate=round(codec.frame_rate), seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate_wave(b, clip),
                    drop_last=True, num_workers=0)

    opt_g = torch.optim.AdamW(codec.parameters(), lr=args.lr, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr, betas=(0.8, 0.99))
    # Separate scalers: G and D gradients live on different scales, a shared
    # scaler would be driven by the larger of the two.
    scaler_g = make_grad_scaler(device, dtype)
    scaler_d = make_grad_scaler(device, dtype)
    amp = autocast(device, dtype)

    step, t0, nan_skips, oom_skips = 0, time.time(), 0, 0
    step_loss = {"g": 0.0, "d": 0.0, "wave": 0.0, "stft": 0.0, "mel": 0.0,
                 "commit": 0.0, "adv": 0.0, "gnorm": 0.0}
    codec.train(); disc.train()
    while step < args.steps:
        for wav in dl:
            if step >= args.steps:
                break
            wav = wav.to(device)

            try:
                m = _train_step(codec, disc, wav, lw, stft_loss, mel_loss,
                                opt_g, opt_d, scaler_g, scaler_d, amp,
                                args.clip_grad)
            except torch.OutOfMemoryError:
                # GAN + fp16 can spike transiently (an exploding forward makes
                # the very next one allocate through the roof).  Drop the
                # batch, release the cached blocks and carry on rather than
                # losing a multi-hour run.
                oom_skips += 1
                opt_g.zero_grad(set_to_none=True)
                opt_d.zero_grad(set_to_none=True)
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                print(f"step {step + 1:5d} | [WARN] CUDA OOM, batch skipped "
                      f"(total {oom_skips})")
                continue
            if m is None:
                nan_skips += 1
                print(f"step {step + 1:5d} | [WARN] non-finite g_loss, batch "
                      f"skipped (total {nan_skips})")
                continue

            step_loss = m
            step += 1
            if step % args.log_every == 0:
                usage = codec.semantic_quantizer.usage()
                peak = torch.cuda.max_memory_allocated() / 2**30 \
                    if device.startswith("cuda") else 0.0
                live = torch.cuda.memory_allocated() / 2**30 \
                    if device.startswith("cuda") else 0.0
                print(f"step {step:5d} | g {step_loss['g']:.3f} "
                      f"d {step_loss['d']:.3f} "
                      f"| wave {step_loss['wave']:.3f} "
                      f"stft {step_loss['stft']:.3f} "
                      f"mel {step_loss['mel']:.3f} "
                      f"commit {step_loss['commit']:.3f} "
                      f"adv {step_loss['adv']:.3f} | cb ppl {usage['perplexity']:.0f} "
                      f"active {usage['n_active']} "
                      f"| gnorm {float(step_loss['gnorm']):.2f} "
                      f"scale {scaler_g.get_scale():.0f} "
                      f"| peak {peak:.2f} live {live:.2f} GiB "
                      f"| {time.time() - t0:.0f}s")
            if step % args.save_every == 0 or step == args.steps:
                ckpt = {"codec": codec.state_dict(), "disc": disc.state_dict(),
                        "config": cfg, "step": step,
                        "precision": precision_name(dtype)}
                torch.save(ckpt, os.path.join(args.out, "codec_latest.pt"))
                with torch.no_grad(), amp:
                    sample = codec.decode(codec.encode(wav[:1]),
                                          out_length=clip)
                    write_wav(os.path.join(args.out, f"recon_step{step}.wav"),
                              sample[0, 0], sr)
                    write_wav(os.path.join(args.out, "target.wav"), wav[0, 0], sr)
                print(f"  saved checkpoint + samples at step {step}")
    print("done.")


if __name__ == "__main__":
    main()
