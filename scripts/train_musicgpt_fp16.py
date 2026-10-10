"""完整 Kimi-MusicGPT（内含 1.5B Kimi Linear）纯 FP16 正式训练管线。

两阶段，对应设计文档 §9：
    stage 1  codec  —— Split-RVQ 编解码器 + 多周期/多尺度 STFT 判别器（GAN）
    stage 2  gpt    —— 冻结 codec 编码出 token 流 -> 延迟交织 -> 1.5B Kimi Linear
                        主干做槽位加权 shift-CE

与 `kimi_musicgpt/train_gpt.py` / `train_codec.py`（fp32 主权重 + autocast 的
AMP 方案）不同，本脚本执行**纯 FP16 布局**：

* 权重**直接以 fp16 实例化并存储**（meta 设备 -> ``materialize_fp16_``，无
  fp32 中间副本），只有归一化层（``nn.LayerNorm`` 与标记了
  ``_keep_fp32_params`` 的 RMSNorm / HeadwiseRMSNorm）保持 fp32；
* 梯度因此也是 fp16，标准 ``torch.amp.GradScaler`` 拒绝 unscale fp16 梯度，
  改用 ``FP16GradScaler``（直接驱动其融合 CUDA kernel）；
* 优化器用 ``KahanAdamW8bit``：8-bit 分块量化动量（1 B/param/矩）+
  Kahan 补偿（解决 fp16 权重被小更新舍入掉的问题）+ 分母地板（小 v 反量化
  为 0 会造成 1/sqrt(v) 尖峰）；
* 梯度检查点、周期性全参数有限值扫描、溢出即整步丢弃 —— 保证**零 NaN/Inf**。

显存（V100-16GB，1.5B 主干）：稳态 = 可训练权重 + fp16 梯度 + 8-bit 状态 +
Kahan 补偿，条件编码器（文本/旋律）默认冻结（它们只在显存预算内才训练，
``--train-conditions`` 打开），codec 始终冻结。

用法（完整流程）：
    conda activate D:\\Envs\\GLM-TTS
    cd D:\\musicGPT
    python scripts/train_musicgpt_fp16.py --stage both ^
        --gpt-config kimi_musicgpt/configs/musicgpt_1p5b.json ^
        --codec-config kimi_musicgpt/configs/codec_tiny_16k.json ^
        --songs 64 --steps 20 --batch-size 1 --max-frames 100 ^
        --out runs/musicgpt_1p5b
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kimi_musicgpt.codec import (KimiMusicCodec, CodecDiscriminator,
                                 MultiScaleSTFTLoss, MelSpectrogramLoss,
                                 discriminator_loss, generator_loss,
                                 feature_matching_loss, waveform_l1)
from kimi_musicgpt.data import SynthMusicDataset
from kimi_musicgpt.text_tokenizer import ByteLevelBPETokenizer
from kimi_musicgpt.structure import create_structure_tokens
from kimi_musicgpt.model import KimiMusicGPT, MusicGPTConfig

from kimi_linear.precision import configure_cuda_backends, describe_device
from kimi_linear.fp16 import (materialize_fp16_, cast_for_fp16_training,
                              assert_fp16_layout, param_dtype_report,
                              FP16GradScaler, KahanAdamW8bit, find_nonfinite)
from kimi_linear.kda import RMSNorm, HeadwiseRMSNorm, KimiDeltaAttention

GiB = 2 ** 30


# --------------------------------------------------------------------------- #
# init helpers (meta -> materialised leaves the storage uninitialised)
# --------------------------------------------------------------------------- #
def _default_init_(m: nn.Module):
    """Full re-initialisation after meta -> materialise.

    Covers everything ``KimiMusicGPT._init`` does **not**: ShortConv's Conv1d
    kernels and the backbone RMSNorms (whose constructor ``torch.ones`` also
    never ran on meta), plus the KDA gate parameters.  Missing any of these
    leaves uninitialised memory in the network -- the weights look finite but
    the gates are garbage and the first backward is NaN.
    """
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d)):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if getattr(m, "bias", None) is not None:
            fan_in = m.weight.shape[1] if m.weight.dim() > 1 else m.weight.shape[0]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(m.bias, -bound, bound)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=1.0)
        if getattr(m, "padding_idx", None) is not None:
            with torch.no_grad():
                m.weight[m.padding_idx].fill_(0)
    elif isinstance(m, (RMSNorm, HeadwiseRMSNorm)):
        nn.init.ones_(m.weight)
    elif isinstance(m, KimiDeltaAttention):
        with torch.no_grad():
            m.A_log.fill_(m._a_log_init)
            m.dt_bias.fill_(m._dt_bias_init)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def materialize_init(module: nn.Module, device: str, init_fn=None) -> nn.Module:
    """meta -> fp16 layout (norms fp32) -> re-run initialisation.

    Used for the 1.5B GPT: a meta build means the fp32 copy of the weights is
    never materialised (6 GB of transient storage avoided).  Not usable for
    modules carrying ``weight_norm`` / parameterisations -- those keep a meta
    tensor around under ``to_empty``, so the codec and the discriminators go
    through :func:`cast_fp16_init` instead.
    """
    materialize_fp16_(module, device, torch.float16)
    module.apply(init_fn or _default_init_)
    keep_accumulators_fp32(module)
    return module


def cast_fp16_init(module: nn.Module, device: str) -> nn.Module:
    """Small modules (codec, discriminators): build fp32 -> cast to the fp16
    layout (norms stay fp32).  Safe with ``weight_norm``."""
    module.to(device)
    cast_for_fp16_training(module, verbose=False)
    keep_accumulators_fp32(module)
    return module


_ACCUMULATOR_BUFFERS = ("ema_cluster_size", "ema_embed_sum", "usage_ema")


def keep_accumulators_fp32(module: nn.Module):
    """Keep VQ EMA accumulators in fp32.

    Requirement ① is about *weights*; these buffers are running accumulators
    (``x.mul_(0.99).add_(...)``).  In fp16 the decay update is ~20 ulp and the
    accumulator drifts, which shows up as codebook collapse.  They are not
    parameters, so keeping them fp32 costs nothing (a few hundred KB).
    """
    for m in module.modules():
        for name in _ACCUMULATOR_BUFFERS:
            b = getattr(m, name, None)
            if isinstance(b, torch.Tensor) and b.is_floating_point() \
                    and b.dtype != torch.float32:
                with torch.no_grad():
                    b.data = b.data.float()


def ledger(model: nn.Module, bits: int = 8, kahan: bool = True,
           frozen_modules=()) -> dict:
    """Steady-state GPU memory (GiB) split into weights / grads / optimiser.

    Only *trainable* parameters carry gradients, 8-bit moments and Kahan
    compensation; frozen sub-modules (codec, condition encoders) contribute
    their weights only.
    """
    train = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in train)
    b_tr = sum(p.numel() * p.element_size() for p in train)
    frozen_b = 0.0
    for m in frozen_modules:
        if m is None:
            continue
        frozen_b += sum(p.numel() * p.element_size()
                        for p in m.parameters() if not p.requires_grad)
    state_b = n_tr * 2 * (bits / 8.0)                  # m + v, `bits` each
    kahan_b = n_tr * 2 if kahan else 0
    return {
        "weights_GiB": (b_tr + frozen_b) / GiB,
        "grads_GiB": b_tr / GiB,
        "optimizer_state_GiB": state_b / GiB,
        "kahan_GiB": kahan_b / GiB,
        "total_GiB": (b_tr + frozen_b + b_tr + state_b + kahan_b) / GiB,
    }


def mem_line(tag=""):
    if not torch.cuda.is_available():
        return "cpu"
    return (f"{tag}peak {torch.cuda.max_memory_allocated()/GiB:5.2f} GiB "
            f"live {torch.cuda.memory_allocated()/GiB:5.2f} GiB")


# --------------------------------------------------------------------------- #
# stage 1: codec (GAN) in pure fp16
# --------------------------------------------------------------------------- #
def codec_step(codec, disc, wav, lw, stft_loss, mel_loss,
               opt_g, opt_d, scaler_g, scaler_d, clip_grad, params_g, params_d):
    """One GAN update in the pure-fp16 layout (no autocast: weights are fp16).

    ``wav`` is kept in fp32 as the *loss target* (measurement precision); the
    networks are fed its fp16 view because the pure-fp16 layout has no fp32
    weights for autocast to fall back on.
    """
    metrics = {}
    w16 = wav.to(next(codec.parameters()).dtype)

    # ---- discriminator ----
    with torch.no_grad():
        recon_d = codec(w16)["recon"]
    real_logits, _ = disc(w16)
    fake_logits, _ = disc(recon_d)
    d_loss = discriminator_loss(real_logits, fake_logits)
    opt_d.zero_grad(set_to_none=True)
    scaler_d.scale_loss(d_loss).backward()
    if scaler_d.unscale_(params_d):
        opt_d.zero_grad(set_to_none=True)
        scaler_d.update(found_inf=True)
    else:
        torch.nn.utils.clip_grad_norm_(params_d, clip_grad)
        opt_d.step()
        scaler_d.update(found_inf=False)
        opt_d.zero_grad(set_to_none=True)
    metrics["d"] = float(d_loss)

    # ---- generator ----
    out = codec(w16)
    recon = out["recon"]
    fake_logits, fake_fmaps = disc(recon)
    with torch.no_grad():
        _, real_fmaps = disc(w16)
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
    scaler_g.scale_loss(g_loss).backward()
    if scaler_g.unscale_(params_g):
        opt_g.zero_grad(set_to_none=True)
        scaler_g.update(found_inf=True)
        return None
    gnorm = torch.nn.utils.clip_grad_norm_(params_g, clip_grad)
    opt_g.step()
    scaler_g.update(found_inf=False)
    opt_g.zero_grad(set_to_none=True)
    metrics.update({"g": float(g_loss), "wave": float(l_wave),
                    "stft": float(l_stft), "mel": float(l_mel),
                    "commit": float(l_commit), "adv": float(l_adv),
                    "gnorm": float(gnorm)})
    return metrics


def run_codec_stage(args, device, log):
    from kimi_musicgpt.train_codec import collate_wave

    with open(args.codec_config) as f:
        cfg = json.load(f)
    sr = cfg["sample_rate"]
    clip = int(args.clip_seconds * sr)
    lw = cfg["loss"]

    torch.manual_seed(args.seed)
    codec = cast_fp16_init(KimiMusicCodec(cfg), device)
    disc = cast_fp16_init(CodecDiscriminator(cfg), device)
    if args.codec_ckpt and os.path.exists(args.codec_ckpt):
        ck = torch.load(args.codec_ckpt, map_location=device)
        codec.load_state_dict(ck["codec"])
        log(f"codec: loaded checkpoint from {args.codec_ckpt}")
    bad = assert_fp16_layout(codec) + assert_fp16_layout(disc)
    log(f"codec fp16 layout violations: {len(bad)}" + ("" if not bad else f" {bad[:3]}"))

    stft_loss = MultiScaleSTFTLoss(lw["stft_scales"]).to(device)
    mel_loss = MelSpectrogramLoss(sr, lw["mel_n_fft"], lw["mel_hop"],
                                  lw["mel_n_mels"]).to(device)
    for m in (stft_loss, mel_loss):
        m.float()                      # measurement losses stay fp32

    ds = SynthMusicDataset(args.songs, sample_rate=sr,
                           frame_rate=round(codec.frame_rate), seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate_wave(b, clip),
                    drop_last=True, num_workers=0)

    p_g = [p for p in codec.parameters() if p.requires_grad]
    p_d = [p for p in disc.parameters() if p.requires_grad]
    opt_g = KahanAdamW8bit(p_g, lr=args.codec_lr, betas=(0.8, 0.99),
                           bits=args.bits, kahan=args.kahan, weight_decay=0.0)
    opt_d = KahanAdamW8bit(p_d, lr=args.codec_lr, betas=(0.8, 0.99),
                           bits=args.bits, kahan=args.kahan, weight_decay=0.0)
    scaler_g = FP16GradScaler(init_scale=args.init_scale, device=device)
    scaler_d = FP16GradScaler(init_scale=args.init_scale, device=device)
    log("codec memory ledger (steady): " + " ".join(
        f"{k.replace('_GiB','')}={v:.2f}" for k, v in
        ledger(codec, bits=args.bits, kahan=args.kahan).items()) + " GiB")

    codec.train(); disc.train()
    step, t0, nan_skips, oom_skips = 0, time.time(), 0, 0
    out_dir = os.path.join(args.out, "codec")
    os.makedirs(out_dir, exist_ok=True)
    last = {}
    while step < args.steps:
        for wav in dl:
            if step >= args.steps:
                break
            wav = wav.to(device)
            try:
                m = codec_step(codec, disc, wav, lw, stft_loss, mel_loss,
                               opt_g, opt_d, scaler_g, scaler_d,
                               args.clip_grad, p_g, p_d)
            except torch.OutOfMemoryError:
                oom_skips += 1
                opt_g.zero_grad(set_to_none=True)
                opt_d.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                log(f"[WARN] CUDA OOM, batch skipped (total {oom_skips})")
                continue
            if m is None:
                nan_skips += 1
                log(f"[WARN] non-finite g_loss / overflow, batch skipped "
                    f"(total {nan_skips})")
                continue
            last = m
            step += 1
            if step % args.log_every == 0:
                log(f"step {step:5d} | g {m['g']:.4f} d {m['d']:.4f} "
                    f"| wave {m['wave']:.4f} stft {m['stft']:.4f} mel {m['mel']:.4f} "
                    f"| commit {m['commit']:.4f} adv {m['adv']:.4f} "
                    f"| gnorm {m['gnorm']:.2f} | scale {scaler_g.get_scale():.0f} "
                    f"| {mem_line()} | {time.time()-t0:.0f}s")
            if step % args.save_every == 0 or step == args.steps:
                torch.save({"codec": codec.state_dict(),
                            "disc": disc.state_dict(),
                            "optimizer_g": opt_g.state_dict(),
                            "optimizer_d": opt_d.state_dict(),
                            "step": step,
                            "codec_config_path": args.codec_config,
                            "precision": "fp16"},
                           os.path.join(out_dir, "codec_latest.pt"))
                log(f"  saved codec checkpoint at step {step}")

    torch.cuda.reset_peak_memory_stats()
    summary = {"stage": "codec", "steps": step, "last": last,
               "nan_skips": nan_skips, "oom_skips": oom_skips,
               "peak_GiB": torch.cuda.max_memory_allocated() / GiB,
               "param_M": param_dtype_report(codec)}
    log(f"codec done: {mem_line('final ')} | nan_skips={nan_skips}")
    return codec, os.path.join(out_dir, "codec_latest.pt")


# --------------------------------------------------------------------------- #
# stage 2: gpt (1.5B backbone) in pure fp16
# --------------------------------------------------------------------------- #
def build_gpt_batch(model, codec, bpe, items, max_frames, device):
    sr = codec.sample_rate
    max_samples = max_frames * codec.hop_length
    wavs, metas = [], []
    for it in items:
        w = it["waveform"]
        if w.shape[-1] > max_samples:
            w = w[..., :max_samples]
        wavs.append(nn.functional.pad(w, (0, max_samples - w.shape[-1])))
        metas.append(it)
    wav = torch.stack(wavs).to(device)
    with torch.no_grad():
        tokens = codec.encode(wav.to(torch.float16))     # [B, T, Q] (fp16 codec)

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
        stream[i, :n] = streams[i]; slot[i, :n] = slots[i]; frame[i, :n] = frames[i]
    struct = {k: torch.stack(v) for k, v in structs.items()}
    ids = [bpe.encode(t, add_bos=True, add_eos=True)[:model.config.music["condition"]["text_max_len"]]
           for t in (m["text"] for m in metas)]
    Lt = max(len(x) for x in ids)
    text_ids = torch.full((len(ids), Lt), bpe.pad_id, dtype=torch.long)
    for i, x in enumerate(ids):
        text_ids[i, :len(x)] = torch.tensor(x)
    style_ids = torch.tensor([m["style_id"] for m in metas], dtype=torch.long)
    return {"stream": stream.to(device), "slots": slot.to(device),
            "src_frames": frame.to(device),
            "struct": {k: v.to(device) for k, v in struct.items()},
            "text_ids": text_ids, "style_ids": style_ids, "wav": wav}


def lr_at(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def run_gpt_stage(args, device, log, codec=None, codec_ckpt=None):
    with open(args.codec_config) as f:
        codec_cfg = json.load(f)
    if codec is None:
        codec = cast_fp16_init(KimiMusicCodec(codec_cfg), device)
        if codec_ckpt and os.path.exists(codec_ckpt):
            codec.load_state_dict(torch.load(codec_ckpt, map_location=device)["codec"])
            log(f"gpt: loaded codec checkpoint {codec_ckpt}")
    codec.eval()
    for p in codec.parameters():
        p.requires_grad_(False)

    sr = codec_cfg["sample_rate"]
    if args.wav_dir:
        from kimi_musicgpt.data import WavMusicDataset
        clip_seconds = args.max_frames / round(codec.frame_rate)
        ds = WavMusicDataset(args.wav_dir, sample_rate=sr,
                             frame_rate=round(codec.frame_rate),
                             clip_seconds=clip_seconds)
        log(f"wav corpus: {len(ds)} files from {args.wav_dir}")
    else:
        ds = SynthMusicDataset(args.songs, sample_rate=sr,
                               frame_rate=round(codec.frame_rate), seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: b, drop_last=True, num_workers=0)

    if args.bpe and os.path.exists(args.bpe):
        bpe = ByteLevelBPETokenizer.load(args.bpe)
    else:
        bpe = ByteLevelBPETokenizer()
        bpe.train((ds[i]["text"] for i in range(min(len(ds), 512))),
                  vocab_size=2000, min_pair_count=2)
        if args.bpe:
            bpe.save(args.bpe)
    log(f"bpe vocab: {bpe.vocab_size}")

    cfg = MusicGPTConfig.from_json(args.gpt_config)
    Q = cfg.music["n_codebooks"]
    token_cap = args.max_tokens or min(8192, cfg.backbone.max_position_embeddings)
    n_tok = args.max_frames * Q + 2
    if n_tok > token_cap:
        raise SystemExit(f"--max-frames {args.max_frames} -> {n_tok} tokens "
                         f"exceeds the {token_cap} token budget")

    torch.manual_seed(args.seed)
    with torch.device("meta"):
        model = KimiMusicGPT(cfg, text_vocab_size=bpe.vocab_size)

    def _gpt_init_(m):
        # Linear/Embedding -> the musicGPT init (normal(0, r)); everything else
        # (ShortConv Conv1d, RMSNorm, KDA gates, LayerNorm) -> _default_init_
        if isinstance(m, (nn.Linear, nn.Embedding)):
            model._init(m)
        else:
            _default_init_(m)

    materialize_init(model, device, init_fn=_gpt_init_)
    model.gradient_checkpointing = args.grad_checkpoint

    # condition encoders: frozen by default (they are the difference between
    # fitting and not fitting the 1.5B model into 16 GiB once grads, 8-bit
    # states and Kahan compensation are accounted for).
    frozen = []
    if not args.train_conditions:
        for name in ("text_encoder", "melody_encoder"):
            mod = getattr(model, name, None)
            if mod is not None:
                mod.requires_grad_(False)
                frozen.append(name)
    params = [p for p in model.parameters() if p.requires_grad]
    bad = assert_fp16_layout(model)
    log(f"gpt params {model.num_parameters()/1e6:.1f}M "
        f"(trainable {sum(p.numel() for p in params)/1e6:.1f}M, "
        f"frozen: {frozen + ['codec']})")
    log("gpt dtype: " + ", ".join(f"{k}={v:.1f}M" for k, v in
                                  param_dtype_report(model).items()))
    log(f"gpt fp16 layout violations: {len(bad)}" + ("" if not bad else f" {bad[:3]}"))

    opt = KahanAdamW8bit(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                         weight_decay=args.weight_decay, bits=args.bits,
                         kahan=args.kahan)
    scaler = FP16GradScaler(init_scale=args.init_scale, device=device)
    rep = ledger(model, bits=args.bits, kahan=args.kahan,
                 frozen_modules=[codec,
                                 getattr(model, "text_encoder", None),
                                 getattr(model, "melody_encoder", None)])
    log("gpt memory ledger (steady): " + " ".join(
        f"{k.replace('_GiB','')}={v:.2f}" for k, v in rep.items()) + " GiB")

    step, t0, nan_skips = 0, time.time(), 0
    out_dir = os.path.join(args.out, "gpt")
    os.makedirs(out_dir, exist_ok=True)
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        step = int(ck.get("step", 0))
        log(f"resumed from {args.resume} at step {step}")

    model.train()
    first_loss = last_loss = None
    consec_skips = 0
    while step < args.steps:
        for items in dl:
            if step >= args.steps:
                break
            batch = build_gpt_batch(model, codec, bpe, items, args.max_frames,
                                    device)
            model.train()
            prefix = model.encode_conditions(batch["text_ids"].to(device),
                                             style_ids=batch["style_ids"].to(device))
            out = model(batch["stream"], batch["slots"], batch["struct"],
                        batch["src_frames"], cond_prefix=prefix,
                        labels=batch["stream"])
            loss = out["loss"]
            scaler.scale_loss(loss).backward()
            if scaler.unscale_(params):
                nan_skips += 1
                consec_skips += 1
                opt.zero_grad(set_to_none=True)
                scaler.update(found_inf=True)
                log(f"[WARN] non-finite grads at step {step+1}, step dropped "
                    f"(total {nan_skips}), scale -> {scaler.get_scale():.0f}")
                if consec_skips >= args.max_skips:
                    raise SystemExit(
                        f"[FATAL] {consec_skips} consecutive dropped steps: "
                        f"the gradients are genuinely non-finite, this is a "
                        f"bug (or data corruption), not fp16 noise")
                continue
            consec_skips = 0
            gnorm = torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
            opt.step()
            scaler.update(found_inf=False)
            opt.zero_grad(set_to_none=True)
            step += 1
            if first_loss is None:
                first_loss = float(loss)
            last_loss = float(loss)
            if step % args.log_every == 0:
                log(f"step {step:5d} | loss {float(loss):.4f} "
                    f"| gnorm {float(gnorm):.2f} "
                    f"| lr {opt.param_groups[0]['lr']:.2e} "
                    f"| scale {scaler.get_scale():.0f} | {mem_line()} "
                    f"| {time.time()-t0:.0f}s")
            if args.check_every and step % args.check_every == 0:
                badp = find_nonfinite(params)
                if badp:
                    raise SystemExit(f"non-finite parameters: {badp[:3]}")
                log(f"  [check] step {step}: all trainable params finite")
            if step % args.save_every == 0 or step == args.steps:
                torch.save({"model": model.state_dict(),
                            "optimizer": opt.state_dict(), "step": step,
                            "gpt_config_path": args.gpt_config,
                            "text_vocab_size": bpe.vocab_size,
                            "precision": "fp16",
                            "optimizer_name": "KahanAdamW8bit"},
                           os.path.join(out_dir, "gpt_latest.pt"))
                log(f"  saved gpt checkpoint at step {step}")
            if torch.cuda.max_memory_allocated() / GiB > args.mem_guard:
                log(f"[WARN] peak {torch.cuda.max_memory_allocated()/GiB:.2f} GiB "
                    f"> guard {args.mem_guard} GiB: stop before thrashing")
                step = args.steps
                break

    peak = torch.cuda.max_memory_allocated() / GiB
    summary = {"stage": "gpt", "steps": args.steps,
               "first_loss": first_loss, "last_loss": last_loss,
               "nan_skips": nan_skips, "peak_GiB": peak,
               "param_M": param_dtype_report(model)}
    log(f"gpt done: peak {peak:.2f} GiB / "
        f"{torch.cuda.get_device_properties(0).total_memory/GiB:.2f} GiB | "
        f"nan_skips={nan_skips} | loss {first_loss} -> {last_loss}")
    if peak > 15.5:
        log("[WARN] peak is within 0.5 GiB of the 16 GiB limit")

    # optional smoke generation (proves the fp16 inference path end-to-end)
    if args.gen_frames > 0:
        try:
            with torch.no_grad():
                frames, _ = model.generate(
                    args.gen_frames,
                    text_ids=batch["text_ids"].to(device)[:1],
                    style_ids=batch["style_ids"].to(device)[:1],
                    temperature=0.9, top_p=0.92, device=device,
                    dtype=torch.float16)
            log(f"smoke generation: {tuple(frames.shape)} frames ok")
        except Exception as e:                                # pragma: no cover
            log(f"[WARN] smoke generation failed: {type(e).__name__}: {e}")

    with open(os.path.join(args.out, "gpt_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return model


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="both", choices=["codec", "gpt", "both"])
    ap.add_argument("--gpt-config", default="kimi_musicgpt/configs/musicgpt_1p5b.json")
    ap.add_argument("--codec-config", default="kimi_musicgpt/configs/codec_tiny_16k.json")
    ap.add_argument("--codec-ckpt", default=None)
    ap.add_argument("--bpe", default=None)
    ap.add_argument("--songs", type=int, default=64)
    ap.add_argument("--wav-dir", default=None)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--clip-seconds", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--codec-lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--no-kahan", dest="kahan", action="store_false")
    ap.set_defaults(kahan=True)
    ap.add_argument("--train-conditions", action="store_true",
                    help="also train the text/melody condition encoders "
                         "(needs ~2.4 GiB more on the 1.5B config)")
    ap.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--init-scale", type=float, default=256.0,
                    help="initial GradScaler scale; the musicGPT loss gives "
                         "fp16 grad norms up to ~180 at init, so scale must "
                         "stay below 65504/180 ~ 364 to avoid any overflow; "
                         "the scaler grows back once gradients settle")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--check-every", type=int, default=25)
    ap.add_argument("--mem-guard", type=float, default=15.5)
    ap.add_argument("--max-skips", type=int, default=20,
                    help="abort after this many consecutively dropped steps")
    ap.add_argument("--gen-frames", type=int, default=0,
                    help="after training, generate N frames as an fp16 smoke test")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train_log.txt")
    t_start = time.time()

    def log(msg):
        line = f"[{time.time()-t_start:7.1f}s] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    device = args.device
    configure_cuda_backends(device)
    log(describe_device(device, torch.float16) +
        f" | cudnn.benchmark={torch.backends.cudnn.benchmark}")
    log(f"args: {vars(args)}")

    codec, codec_ckpt = None, args.codec_ckpt
    if args.stage in ("codec", "both"):
        log("=== stage 1/2: codec (GAN, pure fp16) ===")
        codec, codec_ckpt = run_codec_stage(args, device, log)
    if args.stage in ("gpt", "both"):
        log("=== stage 2/2: gpt (1.5B Kimi Linear backbone, pure fp16) ===")
        if codec is None:
            default_ck = os.path.join(args.out, "codec", "codec_latest.pt")
            codec_ckpt = codec_ckpt or (default_ck if os.path.exists(default_ck) else None)
        run_gpt_stage(args, device, log, codec=codec, codec_ckpt=codec_ckpt)
    log("ALL DONE.")


if __name__ == "__main__":
    main()
