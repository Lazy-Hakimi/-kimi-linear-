"""Multi-head Latent Attention (MLA) with optional NoPE.

Implements the MLA token mixer used by the full-attention layers of Kimi
Linear (paper Section 4): the DeepSeek-style latent attention where keys and
values are compressed into a shared latent ``c_KV``. Following the paper, the
MLA layers use **NoPE** (no positional encoding), configured by
``full_attn.qk_rope_head_dim = 0``; a rope part is still supported when the
config sets a non-zero rope head dim.

The kernel is written in the *absorbed* form: queries are projected into the
latent space with ``W_kv_b^T`` so that attention reads the compressed cache
directly (this is the form that gives MLA its small KV cache):

    score[t,s] = (W_nope q_nope[t]) . c_KV[s] + q_rope[t] . k_rope[s]
    out[t]     = W_v ( sum_s p[t,s] c_KV[s] )

which is algebraically identical to expanding k/v first.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kda import RMSNorm
from .precision import cached_cast


class RopeCache:
    """Precomputed cos/sin tables for the (optional) rope dimensions."""

    def __init__(self, head_dim: int, theta: float, max_pos: int, device, dtype=torch.float32):
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        t = torch.arange(max_pos, device=device).float()
        freqs = torch.outer(t, inv_freq)                     # [max_pos, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)              # [max_pos, head_dim]
        self.cos = emb.cos().to(dtype)
        self.sin = emb.sin().to(dtype)

    @staticmethod
    def apply(x, cos, sin, offset: int = 0):
        """Rotate-half rope. Time is always at dim 1:
        x: [B, T, D] or [B, T, H, D]; cos/sin: [max_pos, D]."""
        T = x.shape[1]
        cos = cos[offset : offset + T].to(x.dtype)   # [T, D]
        sin = sin[offset : offset + T].to(x.dtype)
        if x.dim() == 4:
            cos = cos.unsqueeze(1)                   # [T, 1, D] broadcast over heads
            sin = sin.unsqueeze(1)
        else:
            cos = cos.unsqueeze(0)                   # [1, T, D]
            sin = sin.unsqueeze(0)
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin


class MLA(nn.Module):
    """DeepSeek-style Multi-head Latent Attention (NoPE by default)."""

    def __init__(self, config):
        super().__init__()
        fa = config.full_attn_config
        d = config.hidden_size
        self.num_heads = H = fa["num_heads"]
        self.q_lora_rank = ql = fa["q_lora_rank"]
        self.kv_lora_rank = kvl = fa["kv_lora_rank"]
        self.qk_nope_head_dim = dn = fa["qk_nope_head_dim"]
        self.qk_rope_head_dim = dr = fa["qk_rope_head_dim"]
        self.v_head_dim = dv = fa["v_head_dim"]
        self.q_head_dim = qd = dn + dr
        self.scale = qd ** -0.5
        self.rope_theta = fa["rope_theta"]
        self.max_position_embeddings = config.max_position_embeddings

        self.q_a_proj = nn.Linear(d, ql, bias=fa["qkv_bias"])
        self.q_a_layernorm = RMSNorm(ql, config.rms_norm_eps)
        self.q_b_proj = nn.Linear(ql, H * qd, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(d, kvl + dr, bias=fa["qkv_bias"])
        self.kv_a_layernorm = RMSNorm(kvl, config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(kvl, H * (dn + dv), bias=False)
        self.o_proj = nn.Linear(H * dv, d, bias=fa["o_bias"])

        self._rope = None
        if dr > 0:
            self._rope = RopeCache(dr, self.rope_theta, self.max_position_embeddings,
                                   torch.device("cpu"))
        self._cast_cache = {}                  # see precision.cached_cast

    # ------------------------------------------------------------------ #
    def _rope_to(self, device):
        if self._rope is not None and self._rope.cos.device != device:
            self._rope.cos = self._rope.cos.to(device)
            self._rope.sin = self._rope.sin.to(device)
        return self._rope

    # ------------------------------------------------------------------ #
    def _rope_at(self, x, rope, pos):
        """Rotate-half rope at a *device-side* position tensor.

        ``RopeCache.apply`` takes a Python int offset (fine for eager runs),
        but a captured CUDA graph must read the position from device memory.
        ``index_select`` with a tensor index does exactly that.
        x: [B,T,H,dr] or [B,T,dr]; pos: int64 [1].
        """
        cos = rope.cos.index_select(0, pos).to(x.dtype)             # [1, dr]
        sin = rope.sin.index_select(0, pos).to(x.dtype)
        if x.dim() == 4:
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)           # [1,1,dr]
        else:
            cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)           # [1,1,dr]
        x1, x2 = x.chunk(2, dim=-1)
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    # ------------------------------------------------------------------ #
    def _decode_step_static(self, x, q_nope, q_rope, c_kv, k_rope, state, use_cache):
        """Single-token decode over pre-allocated (graph-capturable) buffers.

        ``state`` holds ``{"c_kv_buf": [B, L, kvl], "pos": int64 [1]}`` where
        ``pos`` is the number of valid latents already in the buffer and ``L``
        is the fixed capacity (prompt + max_new_tokens).  The new latent is
        written at position ``pos`` with ``index_copy_`` -- the index is read
        from device memory when the kernel executes, so a replayed CUDA graph
        picks up the current value.  Attention runs over the whole ``L``-long
        buffer with a validity mask derived from ``pos`` (masked positions get
        ``-inf`` scores, i.e. exactly zero attention weight), which keeps every
        tensor shape static.  The extra FLOPs of attending over ``L`` instead
        of ``pos + 1`` slots are negligible: the decode step is launch-bound,
        not compute-bound.  ``pos`` is advanced in place at the end.

        Numerically this is the same attention as the eager decode path below
        (same absorbed-form algebra, same fp32 score/softmax reduction); only
        the reduction is taken over ``L`` entries of which the invalid ones
        contribute exact zeros.
        """
        B, T, _ = x.shape
        assert T == 1, "the static-buffer path is single-token decode only"
        H, dn, dr, dv, kvl = (self.num_heads, self.qk_nope_head_dim,
                              self.qk_rope_head_dim, self.v_head_dim,
                              self.kv_lora_rank)
        buf, pos = state["c_kv_buf"], state["pos"]
        L = buf.shape[1]

        buf.index_copy_(1, pos, c_kv)                                # write at pos
        if dr > 0:
            kr_buf = state["k_rope_buf"]
            rope = self._rope_to(x.device)
            kr_buf.index_copy_(1, pos, self._rope_at(k_rope, rope, pos))
            q_rope = self._rope_at(q_rope, rope, pos)

        W = cached_cast(self.kv_b_proj.weight, self._cast_cache,
                        dtype=x.dtype).view(H, dn + dv, kvl)
        W_nope, W_v = W[:, :dn, :], W[:, dn:, :]                     # [H,dn,kvl], [H,dv,kvl]

        q_latent = torch.matmul(q_nope.float().permute(0, 2, 1, 3),
                                W_nope.float().unsqueeze(0))         # [B,H,1,kvl]
        scores = torch.matmul(q_latent,
                              buf.float().transpose(1, 2).unsqueeze(1))  # [B,H,1,L]
        if dr > 0:
            scores = scores + torch.matmul(
                q_rope.float().permute(0, 2, 1, 3),
                kr_buf.float().transpose(1, 2).unsqueeze(1))         # [B,H,1,L]
        scores = scores * self.scale

        # validity mask read from the device-side position tensor at replay time
        invalid = torch.arange(L, device=x.device).unsqueeze(0) > pos   # [1,L]
        scores = scores.masked_fill(invalid.unsqueeze(0).unsqueeze(0),
                                    float("-inf"))

        p = torch.softmax(scores.float(), dim=-1)                     # [B,H,1,L]
        mixed = torch.matmul(p.to(buf.dtype), buf.unsqueeze(1))      # [B,H,1,kvl]
        attn_out = torch.matmul(mixed.float(),
                                W_v.float().permute(0, 2, 1).unsqueeze(0))  # [B,H,1,dv]
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, H * dv)
        out = self.o_proj(attn_out.to(x.dtype))

        pos.add_(1)                                                  # advance in place
        if use_cache:
            return out, state
        return out

    def forward(self, x, state=None, use_cache=False):
        """x: [B, T, d]. ``state`` (optional) is a dict {"c_kv": [B,S,kvl],
        "k_rope": [B,S,dr], "offset": int}; when ``use_cache`` the updated
        state is returned."""
        B, T, _ = x.shape
        H, dn, dr, dv, kvl = self.num_heads, self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim, self.kv_lora_rank

        # ---- projections -------------------------------------------------
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))       # [B,T,H*qd]
        q = q.view(B, T, H, self.q_head_dim)
        q_nope, q_rope = q.split([dn, dr], dim=-1)

        kv_a = self.kv_a_proj_with_mqa(x)                             # [B,T,kvl+dr]
        c_kv, k_rope = kv_a.split([kvl, dr], dim=-1)
        c_kv = self.kv_a_layernorm(c_kv)                              # [B,T,kvl]

        # CUDA-Graph static-buffer decode path (see kimi_linear.graph_decode):
        # a state carrying a pre-allocated ``c_kv_buf`` takes the branch below,
        # where every shape is static and the write position is a device-side
        # tensor read by the kernels at execution time -- the prerequisite for
        # ``torch.cuda.graph`` capture of the incremental-decode step.
        if state is not None and "c_kv_buf" in state:
            return self._decode_step_static(x, q_nope, q_rope, c_kv, k_rope,
                                            state, use_cache)

        if dr > 0:
            rope = self._rope_to(x.device)
            q_rope = RopeCache.apply(q_rope, rope.cos, rope.sin, offset=(state["offset"] if state else 0))
            k_rope = RopeCache.apply(k_rope, rope.cos, rope.sin, offset=(state["offset"] if state else 0))

        # ---- absorbed attention over the latent --------------------------
        # kv_b_proj's weight is re-viewed and re-cast on every call; cache the
        # cast view (see precision.cached_cast) -- it is the single largest
        # per-token cast on the incremental-decode path.
        #
        # The target dtype is the **activation** dtype, not a hard-coded fp32:
        # in the pure-fp16 layout (and under autocast) ``q_nope``/``c_kv`` are
        # fp16, and the fused path below feeds them straight into einsum /
        # scaled_dot_product_attention, which require matching dtypes.  Casting
        # the weight to fp32 here raised
        # "expected scalar type Float but found Half" on every fp16 prefill.
        W = cached_cast(self.kv_b_proj.weight, self._cast_cache,
                        dtype=x.dtype).view(H, dn + dv, kvl)
        W_nope, W_v = W[:, :dn, :], W[:, dn:, :]                      # [H,dn,kvl], [H,dv,kvl]

        if state is not None:
            c_kv_all = torch.cat([state["c_kv"], c_kv], dim=1)        # [B,S+T,kvl]
            k_rope_all = torch.cat([state["k_rope"], k_rope], dim=1) if dr > 0 else None
            offset = state["offset"]
        else:
            c_kv_all, k_rope_all, offset = c_kv, k_rope, 0

        # ---- Prefill fast path: NoPE + fresh (aligned) context ----------
        # In the absorbed form, attention over the shared latent c_KV is a
        # single fused QK^T-softmax-V (head dim = kvl), so it maps exactly onto
        # scaled_dot_product_attention. The dot-product scale stays the paper's
        # q_head_dim^-0.5 (NOT kvl^-0.5), hence the explicit ``scale``.
        # Decode, chunked-extension and RoPE keep the explicit path below.
        S = c_kv_all.shape[1]
        if dr == 0 and offset == 0 and T > 1 and S == T:
            q_latent = torch.einsum("bthn,hnc->bhtc", q_nope, W_nope)     # [B,H,T,kvl]
            K = c_kv_all.unsqueeze(1).expand(B, H, S, kvl).contiguous()
            Vh = torch.einsum("bsc,hvc->bhsv", c_kv_all, W_v)             # [B,H,S,dv]
            attn_out = F.scaled_dot_product_attention(
                q_latent, K, Vh, is_causal=True, scale=self.scale,
            )                                                             # [B,H,T,dv]
            attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, H * dv)
        else:
            # Same algebra as the absorbed form above, written with plain
            # batched matmuls instead of einsum: on the incremental-decode path
            # (T == 1) this runs once per generated token per MLA layer, and
            # einsum's parse/permute/reduce chain costs ~0.8 ms of *CPU* per
            # call on the V100 profile -- an order of magnitude more than the
            # matmul it lowers to.  Numerically identical (same reductions).
            q_latent = torch.matmul(q_nope.float().permute(0, 2, 1, 3),
                                    W_nope.float().unsqueeze(0))       # [B,H,T,kvl]
            ckv = c_kv_all.float()
            scores = torch.matmul(q_latent, ckv.transpose(1, 2).unsqueeze(1))  # [B,H,T,S]
            if dr > 0:
                scores = scores + torch.matmul(
                    q_rope.float().permute(0, 2, 1, 3),
                    k_rope_all.float().transpose(1, 2).unsqueeze(1))   # [B,H,T,S]
            scores = scores * self.scale

            # causal mask over global positions
            pos_q = torch.arange(offset, offset + T, device=x.device).unsqueeze(-1)
            pos_k = torch.arange(S, device=x.device)
            scores = scores.masked_fill(pos_k.unsqueeze(0) > pos_q, float("-inf"))

            p = torch.softmax(scores.float(), dim=-1)                 # [B,H,T,S]
            mixed = torch.matmul(p.to(c_kv_all.dtype),
                                 c_kv_all.unsqueeze(1))               # [B,H,T,kvl]
            attn_out = torch.matmul(mixed.float(),
                                    W_v.float().permute(0, 2, 1).unsqueeze(0))  # [B,H,T,dv]
            attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, H * dv)

        out = self.o_proj(attn_out.to(x.dtype))

        if use_cache:
            new_state = {
                "c_kv": c_kv_all,
                "k_rope": k_rope_all if dr > 0 else c_kv_all.new_zeros(B, S, 0),
                "offset": offset + T,
            }
            return out, new_state
        return out
