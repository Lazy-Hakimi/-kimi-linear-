"""End-to-end music generation (design doc §10.1).

Conditions (text / style / structure / optional prompt audio) -> prefix ->
cached autoregressive generation with temperature / top-k / top-p,
slot-constrained sampling and optional classifier-free guidance ->
delay-pattern rearrange -> Kimi-Music-Codec decode -> WAV.

Mixed precision (V100 / FP16)
-----------------------------
Inference needs no GradScaler and no master weights -- the whole decode runs
under autocast in the device's tensor-core dtype (fp16 on sm_70).  Sampling is
kept in fp32 (``logits.float()`` inside the model) so top-p/top-k cut-offs are
not decided by fp16 rounding.

Example:
    python -m kimi_musicgpt.generate \
        --gpt-config kimi_musicgpt/configs/musicgpt_local.json \
        --gpt-ckpt runs/gpt_local/gpt_latest.pt \
        --codec-config kimi_musicgpt/configs/codec_tiny_16k.json \
        --codec-ckpt runs/codec_tiny/codec_latest.pt \
        --bpe runs/gpt_local/bpe.json \
        --text "a 120 bpm pop song in C major" --style-id 0 --bpm 120 \
        --seconds 4 --cfg 2.0 --precision auto --out song.wav
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from .codec import KimiMusicCodec
from .text_tokenizer import ByteLevelBPETokenizer
from .model import KimiMusicGPT, MusicGPTConfig
from .audio_io import write_wav, read_wav
from kimi_linear.precision import (PRECISION_CHOICES, resolve_precision,
                                   configure_cuda_backends, autocast,
                                   describe_device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt-config", required=True)
    ap.add_argument("--gpt-ckpt", required=True)
    ap.add_argument("--codec-config", required=True)
    ap.add_argument("--codec-ckpt", required=True)
    ap.add_argument("--bpe", required=True)
    ap.add_argument("--text", default="a pop song")
    ap.add_argument("--style-id", type=int, default=0)
    ap.add_argument("--bpm", type=int, default=120)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--prompt-wav", default=None)
    ap.add_argument("--prompt-seconds", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--precision", default="auto", choices=list(PRECISION_CHOICES))
    ap.add_argument("--out", default="generated.wav")
    args = ap.parse_args()
    if args.seed:
        torch.manual_seed(args.seed)
    device = args.device
    dtype = resolve_precision(args.precision, device)
    configure_cuda_backends(device)
    print(describe_device(device, dtype))

    with open(args.codec_config) as f:
        codec_cfg = json.load(f)
    codec = KimiMusicCodec(codec_cfg).to(device)
    ck = torch.load(args.codec_ckpt, map_location=device)
    codec.load_state_dict(ck["codec"])
    codec.eval()

    bpe = ByteLevelBPETokenizer.load(args.bpe)
    ckpt = torch.load(args.gpt_ckpt, map_location=device)
    cfg = MusicGPTConfig.from_json(args.gpt_config)
    model = KimiMusicGPT(cfg, text_vocab_size=ckpt["text_vocab_size"])
    if any(t.dtype == torch.float16 for t in ckpt["model"].values()):
        # pure-fp16 checkpoint: keep the weights fp16 in memory as well
        # (2 bytes/param resident instead of 4) -- only normalisation weights
        # stay fp32, matching the training layout exactly
        from kimi_linear.fp16 import cast_for_fp16_training
        cast_for_fp16_training(model, torch.float16, verbose=False)
    model = model.to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    frame_rate = round(codec.frame_rate)
    n_frames = int(args.seconds * frame_rate)
    assert n_frames * cfg.music["n_codebooks"] + 2 <= cfg.backbone.max_position_embeddings, \
        "requested length exceeds the model context budget"

    text_ids = torch.tensor(
        [bpe.encode(args.text, add_bos=True, add_eos=True)],
        dtype=torch.long, device=device)
    style_ids = torch.tensor([args.style_id], dtype=torch.long, device=device)

    amp = autocast(device, dtype)
    prompt_tokens = None
    if args.prompt_wav:
        wav, sr = read_wav(args.prompt_wav)
        wav = wav.unsqueeze(0).to(device)
        n = int(args.prompt_seconds * sr)
        wav = torch.nn.functional.pad(wav, (0, max(0, n - wav.shape[-1])))[..., :n]
        with torch.no_grad(), amp:
            prompt_tokens = codec.encode(wav)

    with torch.no_grad(), amp:
        frames, stream = model.generate(
            n_frames=n_frames, bpm=args.bpm, text_ids=text_ids,
            style_ids=style_ids, prompt_tokens=prompt_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            cfg_scale=args.cfg, device=device, dtype=dtype)
    print(f"generated {frames.shape[1]} frames ({frames.shape[1] / frame_rate:.1f}s), "
          f"stream length {stream.shape[1]}")

    with torch.no_grad(), amp:
        wav = codec.decode(frames)
    write_wav(args.out, wav[0, 0], codec.sample_rate)
    print(f"saved {args.out} ({wav.shape[-1] / codec.sample_rate:.1f}s @ "
          f"{codec.sample_rate}Hz)")


if __name__ == "__main__":
    main()
