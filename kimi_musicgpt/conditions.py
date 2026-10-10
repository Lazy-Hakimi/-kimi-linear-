"""Multimodal condition encoders (design doc §6), Phase-1 prefix injection.

* ``TextConditionEncoder``   -- a real, trainable transformer encoder over
  BPE tokens (byte-level BPE of ``text_tokenizer.py``) + learned positional
  embeddings + projection to the model dimension. (The doc's T5/CLAP slot
  is an external dependency; this self-contained encoder plays the same
  role and trains end-to-end on device.)
* ``MelodyConditionEncoder`` -- reference audio -> codec semantic codebook
  (codebook 0) -> embedding -> projection (§6.2.2).
* ``StyleConditionEncoder``  -- learned style/artist embeddings (1-4 prefix
  tokens).
* ``ConditionPrefixEncoder`` -- concatenates enabled conditions into one
  prefix embedding sequence + final projection (§6.1), with per-condition
  dropout for classifier-free guidance at inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# fp32-weight LayerNorm that accepts low-precision activations
# --------------------------------------------------------------------------- #
class Fp32LayerNorm(nn.LayerNorm):
    """LayerNorm whose **weights** are fp32 but which accepts any input dtype.

    The pure-fp16 layout stores normalisation parameters in fp32 (they sit on
    the residual path of every block and their update/value ratio is the worst
    in the network), while activations are fp16.  ``nn.LayerNorm`` then fails
    with "expected scalar type Half but found Float", so the normalisation is
    computed in fp32 and the result cast back -- exactly what
    ``kimi_linear.kda.RMSNorm`` does.
    """

    def forward(self, x):
        if self.weight is not None and self.weight.dtype == torch.float32 \
                and x.dtype != torch.float32:
            return F.layer_norm(x.float(), self.normalized_shape,
                                self.weight, self.bias, self.eps).to(x.dtype)
        return super().forward(x)


# --------------------------------------------------------------------------- #
# Text condition
# --------------------------------------------------------------------------- #
class _TextBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1 = Fp32LayerNorm(d)
        self.n2 = Fp32LayerNorm(d)
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, x, key_padding_mask=None):
        B, T, D = x.shape
        h = self.n1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, T, self.heads, D // self.heads).transpose(1, 2)
        k = k.view(B, T, self.heads, D // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, D // self.heads).transpose(1, 2)
        attn_mask = None
        if key_padding_mask is not None:                    # [B, T] True=pad
            attn_mask = (~key_padding_mask)[:, None, None, :]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o = o.transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(o)
        x = x + self.mlp(self.n2(x))
        return x


class TextConditionEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_cond: int, n_layers: int = 2,
                 n_heads: int = 4, max_len: int = 64, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len
        self.tok_embed = nn.Embedding(vocab_size, d_cond)
        self.pos_embed = nn.Embedding(max_len, d_cond)
        self.blocks = nn.ModuleList(_TextBlock(d_cond, n_heads)
                                    for _ in range(n_layers))
        self.norm = Fp32LayerNorm(d_cond)
        self.projector = nn.Sequential(nn.Linear(d_cond, d_cond),
                                       Fp32LayerNorm(d_cond))

    def forward(self, text_ids):
        """text_ids: LongTensor [B, L] (padded with pad_id) -> [B, L, d]."""
        B, L = text_ids.shape
        L = min(L, self.max_len)
        text_ids = text_ids[:, :L]
        pad_mask = text_ids == self.pad_id
        pos = torch.arange(L, device=text_ids.device).unsqueeze(0).expand(B, -1)
        x = self.tok_embed(text_ids) + self.pos_embed(pos)
        for blk in self.blocks:
            x = blk(x, pad_mask)
        return self.projector(self.norm(x))


# --------------------------------------------------------------------------- #
# Melody condition
# --------------------------------------------------------------------------- #
class MelodyConditionEncoder(nn.Module):
    """Reference-audio conditioning through the codec's semantic codebook."""

    def __init__(self, codebook_size: int, semantic_dim: int, d_cond: int):
        super().__init__()
        self.semantic_embed = nn.Embedding(codebook_size, semantic_dim)
        self.melody_proj = nn.Sequential(nn.Linear(semantic_dim, d_cond),
                                         Fp32LayerNorm(d_cond))

    def forward(self, semantic_tokens):
        """semantic_tokens: [B, T_f] codec codebook-0 ids -> [B, T_f, d]."""
        return self.melody_proj(self.semantic_embed(semantic_tokens))


# --------------------------------------------------------------------------- #
# Prefix encoder
# --------------------------------------------------------------------------- #
class ConditionPrefixEncoder(nn.Module):
    """Unify heterogeneous conditions into one prefix embedding sequence."""

    def __init__(self, d_model: int, n_styles: int, max_prefix_len: int = 256,
                 condition_dropout: float = 0.0):
        super().__init__()
        self.max_prefix_len = max_prefix_len
        self.condition_dropout = condition_dropout
        self.style_embed = nn.Embedding(n_styles, d_model)
        self.prefix_proj = nn.Sequential(nn.Linear(d_model, d_model),
                                         Fp32LayerNorm(d_model))
        self.norm = Fp32LayerNorm(d_model)
        # fp32 exception (same class as the normalisation layers): the prefix
        # is prepended to the stream, so *every* subsequent token's gradient
        # flows through this projection (MLA attends to it, KDA absorbs it
        # into the recurrent state), and the backward is amplified by the
        # following LayerNorm's 1/std factor.  Measured on the 1.5B model at
        # init: the fp16-stored weight gradient overflows to Inf on some
        # batches.  The projection is 4.2M params (~0.25% of the model), so
        # keeping it fp32 costs ~8 MiB and removes the overflow entirely
        # (fp32 gradients cannot overflow the fp16 storage range).
        self.prefix_proj[0]._keep_fp32_params = True

    def _keep(self, kind: str, batch_size: int, device):
        """Per-batch condition dropout decision (True = keep)."""
        if not self.training or self.condition_dropout <= 0:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        return torch.rand(batch_size, device=device) >= self.condition_dropout

    def forward(self, text_emb=None, melody_emb=None, style_ids=None):
        """Each input optional; at least one must be given.

        text_emb:   [B, Lt, d] output of TextConditionEncoder
        melody_emb: [B, Lm, d] output of MelodyConditionEncoder
        style_ids:  [B] or [B, K] style indices
        Returns prefix embeddings [B, L_prefix, d].
        """
        B = next(t for t in (text_emb, melody_emb) if t is not None).shape[0] \
            if style_ids is None else style_ids.shape[0]
        device = (text_emb if text_emb is not None else
                  melody_emb if melody_emb is not None else
                  self.style_embed.weight).device
        parts = []
        if style_ids is not None:
            s = self.style_embed(style_ids)                      # [B, K, d]
            keep = self._keep("style", B, device).view(B, 1, 1)
            parts.append(s * keep)
        if text_emb is not None:
            keep = self._keep("text", B, device).view(B, 1, 1)
            parts.append(text_emb * keep)
        if melody_emb is not None:
            keep = self._keep("melody", B, device).view(B, 1, 1)
            parts.append(melody_emb * keep)
        if not parts:
            raise ValueError("at least one condition must be provided")
        prefix = torch.cat(parts, dim=1)[:, :self.max_prefix_len]
        out_dtype = prefix.dtype
        # the projection runs in its (fp32) weight dtype -- see __init__ --
        # and the result is cast back to the activation dtype
        prefix = prefix.to(self.prefix_proj[0].weight.dtype)
        return self.norm(self.prefix_proj(prefix)).to(out_dtype)
