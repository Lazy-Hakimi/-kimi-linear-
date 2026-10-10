"""Dependency-free WAV I/O and audio utilities (pure stdlib + torch)."""

from __future__ import annotations

import struct
import wave

import torch


def write_wav(path: str, waveform: torch.Tensor, sample_rate: int):
    """waveform: [T], [1, T] or [B, 1, T] float in [-1, 1] -> 16-bit PCM WAV."""
    # fp32 before quantising to int16: fp16 only resolves ~5e-4 in [-1,1],
    # coarser than the 16-bit PCM grid (3e-5).
    wav = waveform.detach().cpu().reshape(-1).float()
    wav = wav.clamp(-1.0, 1.0)
    pcm = (wav * 32767.0).round().to(torch.int16).numpy().tobytes()
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm)


def read_wav(path: str):
    """-> (waveform [1, T] float32 in [-1, 1], sample_rate)."""
    with wave.open(path, "rb") as f:
        sr = f.getframerate()
        n_ch = f.getnchannels()
        width = f.getsampwidth()
        raw = f.readframes(f.getnframes())
    if width != 2:
        raise ValueError(f"only 16-bit PCM WAV supported, got {width * 8}-bit")
    n = len(raw) // 2
    samples = struct.unpack(f"<{n}h", raw)
    wav = torch.tensor(samples, dtype=torch.float32) / 32768.0
    if n_ch > 1:
        wav = wav.view(-1, n_ch).mean(dim=1)
    return wav.unsqueeze(0), sr


def resample_linear(waveform: torch.Tensor, orig_sr: int, target_sr: int):
    """[B, T] linear-interpolation resampling."""
    if orig_sr == target_sr:
        return waveform
    T = waveform.shape[-1]
    T_new = int(round(T * target_sr / orig_sr))
    x = torch.linspace(0, T - 1, T_new, device=waveform.device)
    x0 = x.floor().long().clamp(0, T - 1)
    x1 = (x0 + 1).clamp(0, T - 1)
    frac = (x - x0.float()).unsqueeze(0)
    return waveform[:, x0] * (1 - frac) + waveform[:, x1] * frac


def normalize_loudness(waveform: torch.Tensor, target_rms: float = 0.1):
    """Simple RMS normalization (stand-in for -14 LUFS in the doc §9.1.2)."""
    rms = waveform.pow(2).mean().sqrt().clamp(min=1e-8)
    wav = waveform * (target_rms / rms)
    return wav.clamp(-1.0, 1.0)
