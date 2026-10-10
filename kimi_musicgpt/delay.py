"""Delay (interleave) pattern for multi-codebook streams (design doc §5.1).

A ``(T, Q)`` codebook matrix is flattened into a single autoregressive
stream of length ``(T + max_delay) * Q`` plus BOS/EOS:

* stream frame ``s`` codebook slot ``q`` sits at stream position ``s*Q + q``
  and carries the token of *source frame* ``s - delays[q]``;
* source frames ``< 0`` are filled with PAD (the diagonal delay of
  MusicGen); ``delays == zeros`` reduces to the doc's plain interleave
  ``c0(t) c1(t) ... c7(t) c0(t+1) ...``;
* BOS is prepended, EOS appended.

``build_delay_stream`` produces the stream together with per-position
``slots`` (which codebook a position belongs to) and ``src_frames`` (which
audio frame it belongs to, -1 for BOS/EOS) needed by ``MusicEmbedding``
(structure embeddings are per-frame) and by slot-constrained sampling.
``rearrange_delay_pattern`` is the exact inverse used at inference.
"""

from __future__ import annotations

import torch


def build_delay_stream(tokens, delays, pad_id, bos_id, eos_id):
    """tokens: [B, T, Q] -> dict(stream, slots, src_frames) each [B, L].

    L = 1 + (T + max_delay) * Q + 1  (BOS + body + EOS).
    """
    B, T, Q = tokens.shape
    device = tokens.device
    delays = torch.as_tensor(delays, device=device, dtype=torch.long)
    assert delays.numel() == Q
    max_d = int(delays.max())
    S = T + max_d

    body = torch.full((B, S * Q), pad_id, dtype=torch.long, device=device)
    slots = torch.arange(S * Q, device=device) % Q                  # [S*Q]
    stream_frames = torch.arange(S * Q, device=device) // Q         # [S*Q]
    src = stream_frames.unsqueeze(0) - delays[slots].unsqueeze(0)   # [1, S*Q]
    valid = (src >= 0) & (src < T)                                   # [1, S*Q]
    src_c = src.clamp(0, T - 1).expand(B, -1)                        # [B, S*Q]
    flat = tokens.reshape(B, T * Q)
    idx = src_c * Q + slots.unsqueeze(0)                             # [B, S*Q]
    gathered = flat.gather(1, idx)
    body = torch.where(valid.expand(B, -1), gathered, body)

    bos = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    eos = torch.full((B, 1), eos_id, dtype=torch.long, device=device)
    stream = torch.cat([bos, body, eos], dim=1)

    body_slots = slots.unsqueeze(0).expand(B, -1)
    full_slots = torch.cat([torch.zeros_like(bos), body_slots,
                            torch.full_like(eos, Q - 1)], dim=1)
    src_frames = torch.cat([
        torch.full((B, 1), -1, dtype=torch.long, device=device),
        src.expand(B, -1).clamp(-1, T - 1),
        torch.full((B, 1), -1, dtype=torch.long, device=device)], dim=1)
    return {"stream": stream, "slots": full_slots, "src_frames": src_frames,
            "n_frames": T}


def rearrange_delay_pattern(stream, delays, n_frames, q, pad_id):
    """Inverse: stream [B, L] -> tokens [B, T, Q] (PAD where unset).

    Accepts a stream with or without leading BOS / trailing EOS; extra or
    missing tail positions (e.g. truncated generation) are tolerated.
    """
    B, L = stream.shape
    delays = torch.as_tensor(delays, device=stream.device, dtype=torch.long)
    out = torch.full((B, n_frames, q), pad_id, dtype=torch.long,
                     device=stream.device)
    for i in range(L):
        # skip leading BOS if present: body starts at index 1 when the first
        # token is not part of the (frame, slot) grid -- caller convention:
        # streams produced by build_delay_stream / generate always have BOS.
        p = i - 1
        if p < 0:
            continue
        s, slot = divmod(p, q)
        f = s - int(delays[slot])
        if 0 <= f < n_frames:
            out[:, f, slot] = stream[:, i]
    return out


def stream_length(n_frames, q, delays):
    """Body length (without BOS/EOS) for n_frames source frames."""
    return (n_frames + int(max(delays))) * q
