"""Kimi-Music-Codec: vector quantization.

Complete (non-simplified) implementations:

* ``VectorQuantizer`` -- VQ-VAE quantizer with
  - exact L2 nearest-neighbour lookup (||z||^2 - 2 z.e + ||e||^2),
  - straight-through gradient estimator,
  - commitment loss  beta * ||z_e - sg(e)||^2  (encoder pulled to codebook),
  - codebook loss    ||sg(z_e) - e||^2         (codebook pulled to encoder),
  - optional EMA codebook updates (VQ-VAE-v2 style),
  - dead-code revival: entries whose EMA usage falls below a threshold are
    re-initialised to a random encoder output from the current batch.
* ``ResidualVectorQuantizer`` -- sequential residual quantization with
  train-time quantizer dropout (design doc §2.2.2 "RVQ Dropout"), enabling
  inference-time quality/speed trade-offs with fewer codebooks.

Tensors are ``[B, T, D]`` (time-major), matching the codec internals.

Mixed-precision note (FP16 on the V100 target)
----------------------------------------------
The quantizer always runs on the **fp32 path** (``torch.autocast(enabled=False)``
around its forward).  Two reasons:

* the codebook lookup is an ``argmin`` over ``C=2048`` distances -- a decision,
  not a smooth function, so a half-precision tie-break silently changes the
  token stream the whole GPT stage is trained on;
* the EMA accumulators (``ema_embed_sum`` / ``ema_cluster_size``) are
  ``x *= 0.99; x += ...`` recursions that stall completely in fp16
  (``1 - 0.99 = 0.01`` is representable, but the *increment* underflows).

The cost is one extra cast of ``[B*T, D]`` per codebook -- irrelevant next to
the encoder/decoder conv stacks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, embedding_dim: int,
                 commitment_cost: float = 0.25, ema: bool = False,
                 ema_decay: float = 0.99, dead_code_revival: bool = True,
                 usage_threshold: float = 1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema = ema
        self.ema_decay = ema_decay
        self.dead_code_revival = dead_code_revival
        self.usage_threshold = usage_threshold

        embed = torch.randn(codebook_size, embedding_dim) / (embedding_dim ** 0.5)
        if ema:
            self.register_buffer("codebook", embed)
            self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
            self.register_buffer("ema_embed_sum", embed.clone())
        else:
            self.codebook = nn.Parameter(embed)
        self.register_buffer("usage_ema", torch.ones(codebook_size))

    # ------------------------------------------------------------------ #
    def _distances(self, z):                    # z: [N, D] -> [N, C]
        # The codebook is stored in fp16 in the pure-fp16 layout; the distance
        # (and therefore the assignment decision) is still computed in the
        # caller's dtype -- fp32 in the quantizer's own fp32 path -- so the
        # storage dtype does not change which code is selected.
        cb = self.codebook.to(z.dtype)
        return (z.pow(2).sum(-1, keepdim=True)
                - 2.0 * z @ cb.t()
                + cb.pow(2).sum(-1))

    # ------------------------------------------------------------------ #
    def forward(self, z):
        """z: [B, T, D] -> (z_q [B,T,D], tokens [B,T], losses dict)."""
        B, T, D = z.shape
        in_dtype = z.dtype
        # fp32 path: see the mixed-precision note in the module docstring.
        with torch.autocast(device_type=z.device.type, enabled=False):
            flat = z.reshape(-1, D).float()                      # [N, D]
            tokens = self._distances(flat).argmin(-1)            # [N]
            z_q = self.codebook[tokens].float()                  # [N, D]

            commitment = F.mse_loss(flat, z_q.detach())
            codebook = F.mse_loss(flat.detach(), z_q)

            if self.training:
                if self.ema:
                    self._ema_update(flat, tokens)
                if self.dead_code_revival:
                    self._revive_dead_codes(flat)
                # usage tracking happens after the codebook mutations above so
                # the EMA reflects the entries that were actually selected
                with torch.no_grad():
                    tokens_now = self._distances(flat).argmin(-1)
                    counts = torch.bincount(tokens_now, minlength=self.codebook_size)
                self.usage_ema.mul_(0.99).add_(counts.float(), alpha=0.01)

            z_q_st = flat + (z_q - flat).detach()                # straight-through
            losses = {
                "commitment": self.commitment_cost * commitment,
                "codebook": codebook,
            }
        return z_q_st.view(B, T, D).to(in_dtype), tokens.view(B, T), losses

    # ------------------------------------------------------------------ #
    def _ema_update(self, flat, tokens):
        one_hot = F.one_hot(tokens, self.codebook_size).type(flat.dtype)   # [N, C]
        cluster = one_hot.sum(0)                                           # [C]
        embed_sum = one_hot.t() @ flat                                     # [C, D]
        self.ema_cluster_size.mul_(self.ema_decay).add_(
            cluster, alpha=1 - self.ema_decay)
        self.ema_embed_sum.mul_(self.ema_decay).add_(
            embed_sum, alpha=1 - self.ema_decay)
        n = self.ema_cluster_size.sum()
        smoothed = ((self.ema_cluster_size + 1e-5)
                    / (n + self.codebook_size * 1e-5) * n)
        self.codebook.copy_(self.ema_embed_sum / smoothed.unsqueeze(-1).clamp(min=1e-6))

    # ------------------------------------------------------------------ #
    def _revive_dead_codes(self, flat):
        dead = self.usage_ema < self.usage_threshold
        n_dead = int(dead.sum())
        if n_dead == 0 or flat.shape[0] == 0:
            return
        idx = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
        with torch.no_grad():
            replacement = flat[idx].detach().clone()
            # codebook / EMA storage dtypes may differ from the fp32 working
            # dtype of the distance math (pure-fp16 layout)
            self.codebook[dead] = replacement.to(self.codebook.dtype)
            if self.ema:
                self.ema_embed_sum[dead] = replacement.to(self.ema_embed_sum.dtype)
                self.ema_cluster_size[dead] = 1.0
            self.usage_ema[dead] = 1.0

    # ------------------------------------------------------------------ #
    def decode(self, tokens):
        """tokens: [B, T] -> embeddings [B, T, D]."""
        return self.codebook[tokens]

    # ------------------------------------------------------------------ #
    def usage(self):
        """Perplexity-style usage stats for logging."""
        p = self.usage_ema / self.usage_ema.sum().clamp(min=1e-9)
        perplexity = torch.exp(-(p * (p + 1e-9).log()).sum())
        return {"perplexity": perplexity.item(),
                "n_active": int((self.usage_ema >= self.usage_threshold).sum())}


class ResidualVectorQuantizer(nn.Module):
    """Residual stack of :class:`VectorQuantizer` with quantizer dropout.

    ``n_quantizers`` quantizers successively quantize the residual left by
    the previous ones. During training a random count ``n_q`` in
    ``[1, n_quantizers]`` is applied with probability ``dropout_p`` (per
    forward), matching the design doc's RVQ-dropout requirement; the full
    stack is always used in eval mode.
    """

    def __init__(self, n_quantizers: int, codebook_size: int, embedding_dim: int,
                 commitment_cost: float = 0.25, ema: bool = False,
                 dropout_p: float = 0.5, dead_code_revival: bool = True,
                 usage_threshold: float = 1e-5):
        super().__init__()
        self.n_quantizers = n_quantizers
        self.dropout_p = dropout_p
        self.quantizers = nn.ModuleList(
            VectorQuantizer(codebook_size, embedding_dim, commitment_cost,
                            ema=ema, dead_code_revival=dead_code_revival,
                            usage_threshold=usage_threshold)
            for _ in range(n_quantizers)
        )

    # ------------------------------------------------------------------ #
    def _active_count(self):
        if self.training and self.dropout_p > 0 and torch.rand(()) < self.dropout_p:
            return int(torch.randint(1, self.n_quantizers + 1, (1,)))
        return self.n_quantizers

    # ------------------------------------------------------------------ #
    def forward(self, z, n_quantizers=None):
        """z: [B, T, D] -> (z_q, tokens [B,T,Q], losses dict)."""
        n_q = n_quantizers or self._active_count()
        residual = z
        z_q_total = 0.0
        tokens, losses = [], {"commitment": 0.0, "codebook": 0.0}
        for i in range(n_q):
            z_q_i, tok_i, l_i = self.quantizers[i](residual)
            residual = residual - z_q_i.detach() if i < n_q - 1 else residual - z_q_i
            z_q_total = z_q_total + z_q_i
            tokens.append(tok_i)
            losses["commitment"] = losses["commitment"] + l_i["commitment"]
            losses["codebook"] = losses["codebook"] + l_i["codebook"]
        return z_q_total, torch.stack(tokens, dim=-1), losses

    # ------------------------------------------------------------------ #
    def decode(self, tokens):
        """tokens: [B, T, Q<=n] -> summed embeddings [B, T, D]."""
        z = 0.0
        for i in range(tokens.shape[-1]):
            z = z + self.quantizers[i].decode(tokens[..., i])
        return z

    # ------------------------------------------------------------------ #
    def usage(self):
        return {f"q{i}": q.usage() for i, q in enumerate(self.quantizers)}
