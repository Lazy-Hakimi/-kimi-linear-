"""MusicEmbedding (design doc §5.1): multi-codebook input adapter.

Each stream position carries (token, slot, src_frame). The input vector is

    codebook_embed[slot](token) + slot_embed[slot] (+ structure_embed[frame])

The structure term (§7.2) is per audio frame and broadcast over the frame's
Q codebook slots; the caller passes per-position frame indices
(``src_frames`` from ``delay.build_delay_stream``) so delayed positions get
the structure of their *source* frame.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .structure import StructureConditionEncoder


class MusicEmbedding(nn.Module):
    def __init__(self, d_model: int, n_codebooks: int, codebook_size: int,
                 n_special: int, structure_cfg: dict = None):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.vocab_size = codebook_size + n_special
        self.codebook_embed = nn.ModuleList(
            nn.Embedding(self.vocab_size, d_model)
            for _ in range(n_codebooks)
        )
        self.slot_embed = nn.Embedding(n_codebooks, d_model)
        self.struct_encoder = (StructureConditionEncoder(d_model, **structure_cfg)
                               if structure_cfg is not None else None)

    def forward(self, tokens, slots, structure=None, src_frames=None):
        """tokens/slots: [B, L]; structure: optional dict of [B, F] label
        tensors (F = n_frames); src_frames: [B, L] frame index per position.
        Returns [B, L, d_model].
        """
        x = 0.0
        for q in range(self.n_codebooks):
            mask = slots == q                                   # [B, L]
            if mask.any():
                emb = self.codebook_embed[q](tokens.clamp(0, self.vocab_size - 1))
                x = x + emb * mask.unsqueeze(-1).to(emb.dtype)
        x = x + self.slot_embed(slots)
        if structure is not None and self.struct_encoder is not None:
            s = self.struct_encoder(structure)                  # [B, F, d]
            if src_frames is not None:
                frames = src_frames.clamp(0, s.shape[1] - 1)
            else:
                frames = (torch.arange(tokens.shape[1], device=tokens.device)
                          // self.n_codebooks).clamp(max=s.shape[1] - 1)
                frames = frames.unsqueeze(0).expand(tokens.shape[0], -1)
            x = x + torch.gather(s, 1, frames.unsqueeze(-1)
                                 .expand(-1, -1, s.shape[-1]))
        return x
