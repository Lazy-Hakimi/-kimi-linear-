"""Kimi Delta Attention (KDA).

Reference PyTorch implementation of the token mixer proposed in
"Kimi Linear: An Expressive, Efficient Attention Architecture"
(arXiv:2510.26692).

Recurrent form (paper Eq. 1):

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T q_t

where ``alpha_t = exp(g_t) <= 1`` is a *per-channel* (fine-grained) forget
gate -- the key difference to Gated DeltaNet, which uses a scalar gate per
head -- and ``beta_t in (0,1)`` is the scalar delta-rule learning rate.

The chunkwise kernel below is a faithful implementation of the paper's
specialized DPLR algorithm (Eq. 8/9 and Appendix C, Listing 1): the UT
transform yields ``M = (I + L)^{-1}`` with the strictly-lower-triangular

    L[c, i] = beta_c * <k_c , exp(gc_c - gc_i) * k_i>,      i < c

``gc`` being the within-chunk cumulative log-decay, so that

    W = M (exp(gc) * K),      U = M V                      (Eq. 6/7)
    O = (exp(gc) * Q) S_0 + Tril[(Q exp(gc)) (K exp(-gc))^T] (U - W S_0)
    S_1 = exp(gc_C) S_0 + (K * exp(gc_C - gc))^T (U - W S_0)   (Eq. 8/9)

The ``a = b = k`` binding of the DPLR transition removes the need for a
second-level chunking of the reciprocal decay (paper Section 6.2); the
per-column loop kept below follows the paper's pseudo code and preserves
numerical stability in fp32.

Mixed precision (V100 / FP16)
-----------------------------
Both kernels accept ``state_dtype``.  All the delta-rule mathematics is done
in fp32 either way, but the *carried* recurrent state ``S`` is what accumulates
error across chunks: stored in fp16 it is rounded once per chunk (relative
error ~5e-4) and its small entries lose all precision below 6e-5.  Keeping ``S``
in fp32 (an fp32 master state -- the same trick AMP uses for weights) costs
``B*H*K*V*4`` bytes (1 MiB/layer for the 1.5B config) and removes that drift
entirely.  :class:`KimiDeltaAttention` turns it on automatically whenever the
activations are fp16/bf16.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fp16 import FP16_MAX_SAFE
from .precision import cached_cast


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
def kda_chunkwise(q, k, v, g, beta, initial_state=None, chunk_size=64, tile_size=0,
                  state_dtype=None):
    """Chunkwise KDA kernel (paper Eq. 6-9, Appendix C Listing 1) -- optimized.

    This is the pure-PyTorch-fast reference: the two O(C) per-column Python
    loops of the naive formulation are replaced by tiled log-space reductions
    (still computing ``exp(gc_c - gc_i)`` exactly as the paper does for
    numerical stability, never the overflow-prone ``exp(gc)*exp(-gc)`` product),
    and the hand-rolled forward substitution is replaced by the fused batched
    ``torch.linalg.solve_triangular``. The inter-block recurrence remains a
    sequential loop, matching the paper's own "inter-block recurrent /
    intra-block parallel" strategy (Eq. 8/9).

    V100 (sm_70) fp16 fast path
    ---------------------------
    The kernel is HBM-bandwidth-bound on Volta, so the fp16 path is engineered
    around *cast traffic*, not FLOPs:

    * fp32 inputs run **bit-identically** to the plain fp32 kernel (verified);
    * fp16 inputs run the transient tensors (exp masks, masked products,
      ``Aout``, ``w_n``/``u_n``/``pseudo_v``) in fp16 -- half the bytes of the
      dominant masked-sum traffic -- with all reductions accumulated in fp32
      (``.sum(dtype=fp32)``, fp32-accumulating cuBLAS GEMMs);
    * the pair-difference ``gc_c - gc_i`` is computed in fp32 (it is the
      difference of two large cumulative decays; rounding ``gc`` to fp16 first
      would inject >10% error into ``exp``), cast to fp16 *once*, then
      ``exp_`` in fp16.  No clamping is needed: only strictly-upper-triangular
      entries can exceed fp16 range and they are zeroed by the ``tril`` masks
      right after the per-pair reduction (inf/NaN in masked positions can
      never leak into kept pairs -- the reduction is per ``(c, i)``);
    * the intra-chunk masked sums for ``L`` and ``Aout`` share one ``D`` per
      tile (the naive structure computed it twice);
    * the carried state ``S`` stays fp32 (fp32 master state) and is cast to
      fp16 once per chunk for the two state GEMMs; measured vs fp32 on V100:
      out rel-err ~9e-4, state ~7e-4, kernel speed-up ~1.8-1.9x on fp16 inputs.

    Args:
        q, k: ``[B, T, H, K]`` query/key (expected l2-normalized per head).
        v:    ``[B, T, H, V]`` values.
        g:    ``[B, T, H, K]`` log decay (``<= 0``), i.e. ``log alpha``.
        beta: ``[B, T, H]``    delta-rule rates in ``(0, 1)``.
        initial_state: optional ``[B, H, K, V]`` recurrent state.
        chunk_size: chunk length ``C`` (paper uses 64).
        tile_size: row-tile size for the masked-sum buffers; ``0`` (default)
            uses the full chunk -- fastest on the V100's 16GB.  Set 16/32 to
            bound the ``[tile, C, K]`` transients on smaller GPUs.
        state_dtype: dtype of the returned recurrent state; ``None`` keeps the
            value dtype (pass ``torch.float32`` for an fp32 master state when
            the activations are fp16).
    Returns:
        o: ``[B, T, H, V]`` outputs;  state: ``[B, H, K, V]`` final state.
    ``T`` must be a multiple of ``chunk_size`` (the module pads / splits).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = int(chunk_size)
    NT = T // C
    out_dtype = v.dtype
    td = torch.float16 if out_dtype == torch.float16 else torch.float32

    # Gates stay fp32 (stability-critical); q/k/v stay in their storage dtype.
    g = g.float()
    beta = beta.float()

    # [B, T, H, D] -> [B, H, N, C, D]
    def ch(x):
        return x.reshape(B, NT, C, H, x.shape[-1]).permute(0, 3, 1, 2, 4)

    q, k, v, g = ch(q), ch(k), ch(v), ch(g)   # q,k,v stay in input dtype
    beta = beta.reshape(B, NT, C, H).permute(0, 3, 1, 2)  # [B, H, N, C]

    gc = g.cumsum(-2)  # [B, H, N, C, K] cumulative log decay inside chunk (fp32)
    TS = min(C, max(int(tile_size), 1)) if tile_size else C
    eye = torch.eye(C, device=q.device, dtype=torch.float32)
    q_scale = K ** -0.5

    if initial_state is None:
        S = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)
    else:
        S = initial_state.float().clone()
    o = torch.empty(B, H, NT, C, V, device=q.device, dtype=td)

    for n in range(NT):
        q_n = q[:, :, n] * q_scale                     # td, scaled once
        k_n, v_n = k[:, :, n], v[:, :, n]              # td
        g_n, b_n = gc[:, :, n], beta[:, :, n]          # fp32
        exp_g = g_n.exp()                              # fp32, in (0, 1]

        # ---- M = (I + L)^{-1} Diag(beta), L strictly lower; Aout (Eq. 9) ---
        # Both masked sums share the same decay tensor D per tile.
        L = torch.empty(B, H, C, C, device=q.device, dtype=torch.float32)
        Aout = torch.empty(B, H, C, C, device=q.device, dtype=td)
        for c0 in range(0, C, TS):
            ce = min(c0 + TS, C)
            g_row = g_n[..., c0:ce, :]                       # [B,H,TS',K] fp32
            # fp32 difference (precision) -> one fp16 cast -> fp16 exp.
            # The fp16 clamp is *backward*-critical: only strictly-upper-
            # triangular entries (zeroed by tril below) can exceed fp16's
            # exp range, and clamping them changes no kept value -- but an
            # inf here would make exp's backward compute inf*0 = NaN and
            # poison every gradient downstream (measured: 572/~700 params).
            D = g_row.unsqueeze(-2) - g_n.unsqueeze(-3)
            if td == torch.float16:
                D = D.clamp(max=11.0)
            D = D.to(td).exp_()                              # td, (0,1]
            k_row = k_n[..., c0:ce, :]
            blk = (k_row.unsqueeze(-2) * k_n.unsqueeze(-3) * D).sum(-1, dtype=torch.float32)
            L[..., c0:ce, :] = (blk * b_n[..., c0:ce, None]).tril(diagonal=c0 - 1)
            q_row = q_n[..., c0:ce, :]
            blk = (q_row.unsqueeze(-2) * k_n.unsqueeze(-3) * D).sum(-1, dtype=torch.float32)
            Aout[..., c0:ce, :] = blk.tril(diagonal=c0).to(td)

        M = torch.linalg.solve_triangular(eye + L, eye, upper=False, unitriangular=True)
        M = (M * b_n[..., None, :]).to(td)                   # (I+L)^{-1} Diag(beta)
        w_n = M @ (exp_g.to(td) * k_n)                       # Eq. 7
        u_n = M @ v_n

        # ---- intra-block attention + inter-block recurrence (Eq. 8/9) ----
        S_h = S.to(td)                                       # one cast per chunk
        pseudo_v = u_n - w_n @ S_h                           # U - W S_0
        o[:, :, n] = (q_n * exp_g.to(td)) @ S_h + Aout @ pseudo_v
        g_last = g_n[..., -1, :]
        S = S * g_last.exp().unsqueeze(-1)
        S = S + (((g_n[..., -1:, :] - g_n).exp().to(td) * k_n).transpose(-1, -2)
                 @ pseudo_v).float()                         # fp32 master state

    o = o.permute(0, 2, 3, 1, 4).reshape(B, NT * C, H, V)
    return o.to(out_dtype), S.to(state_dtype or out_dtype)


def kda_recurrent(q, k, v, g, beta, state=None, state_dtype=None):
    """Step-by-step KDA recurrence (paper Eq. 1) -- ground truth / decoding.

    Same arguments as :func:`kda_chunkwise`; ``T`` may be any length.
    ``state_dtype`` controls the dtype of the returned state (fp32 master
    state when the activations are fp16).
    Returns outputs ``[B, T, H, V]`` and the final state ``[B, H, K, V]``.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    out_dtype = v.dtype
    q = q.float() * (K ** -0.5)
    k = k.float()
    v = v.float()
    g = g.float()
    beta = beta.float()

    S = torch.zeros(B, H, K, V, device=q.device) if state is None else state.float().clone()
    o = torch.zeros(B, T, H, V, device=q.device)
    for t in range(T):
        alpha_t = g[:, t].exp().unsqueeze(-1)          # [B,H,K,1] Diag(alpha) rows
        S = S * alpha_t                                # Diag(alpha_t) S_{t-1}
        k_t = k[:, t]                                  # [B,H,K]
        b_t = beta[:, t].unsqueeze(-1)                 # [B,H,1]
        kS = (k_t.unsqueeze(-1) * S).sum(-2)           # k_t^T S  -> [B,H,V]
        S = S - b_t.unsqueeze(-1) * k_t.unsqueeze(-1) * kS.unsqueeze(-2)
        S = S + b_t.unsqueeze(-1) * k_t.unsqueeze(-1) * v[:, t].unsqueeze(-2)
        q_t = q[:, t]                                  # [B,H,K]
        o[:, t] = (q_t.unsqueeze(-1) * S).sum(-2)      # q_t^T S_t
    if out_dtype is torch.float16:
        o = o.clamp(-FP16_MAX_SAFE, FP16_MAX_SAFE)     # see kda_chunkwise
    return o.to(out_dtype), S.to(state_dtype or out_dtype)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """Root-mean-square normalisation.

    Marked ``_keep_fp32_params``: in the pure-fp16 training layout
    (:func:`kimi_linear.fp16.cast_for_fp16_training`) every other parameter of
    the network is stored in fp16 but normalisation weights stay fp32.  Reasons:

    * a norm weight multiplies an already-normalised activation, so its value
      (~1) is much larger than a typical update (~lr); fp16 would round most
      updates away even with Kahan compensation;
    * the norm gain sits directly on the residual path of *every* layer, so a
      relative error there is amplified by the depth of the network.

    The cost is negligible (a few thousand parameters in total).
    """

    _keep_fp32_params = True

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)
        self._cast_cache = {}                  # see precision.cached_cast

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        w = self.weight if self.weight.dtype == torch.float32 \
            else cached_cast(self.weight, self._cast_cache)
        return (x * w).to(dtype)


class HeadwiseRMSNorm(nn.Module):
    """RMSNorm applied per head with a per-head weight ``[H, D]``
    (paper: "head-wise RMSNorm" before the KDA output projection).

    fp32-stored for the same reasons as :class:`RMSNorm`.
    """

    _keep_fp32_params = True

    def __init__(self, heads: int, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(heads, dim))
        self.eps = float(eps)
        self._cast_cache = {}                  # see precision.cached_cast

    def forward(self, x):  # x: [B, T, H, D]
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        w = self.weight if self.weight.dtype == torch.float32 \
            else cached_cast(self.weight, self._cast_cache)
        return (x * w).to(dtype)


class ShortConv(nn.Module):
    """Causal depthwise (per-channel) convolution, kernel size ``conv_size``."""

    def __init__(self, dim: int, conv_size: int, bias: bool = True):
        super().__init__()
        self.dim = dim
        self.conv_size = conv_size
        self.conv = nn.Conv1d(dim, dim, conv_size, groups=dim, bias=bias)
        self._cast_cache = {}                  # see precision.cached_cast

    def forward(self, x, conv_state=None, use_cache=False):
        """x: [B, T, D]. If ``use_cache``, ``x`` is [B, 1, D] and ``conv_state``
        holds the last ``conv_size - 1`` inputs ``[B, conv_size-1, D]``."""
        if use_cache:
            window = torch.cat([conv_state, x], dim=1)  # [B, conv_size, D]
            w = cached_cast(self.conv.weight, self._cast_cache).squeeze(1).t()  # [C, D]
            # elementwise mul + reduce: same math as einsum("bcd,dc->bd") but
            # one kernel launch instead of einsum's parse/permute/reduce chain
            # (this path runs once per generated token, so launch overhead --
            # ~0.8 ms per einsum call on the V100 profile -- dominates decode)
            out = (window.float() * w).sum(1)           # [B, D]
            if self.conv.bias is not None:
                out = out + self.conv.bias.float()
            out = out.to(x.dtype).unsqueeze(1)          # [B, 1, D]
            new_state = window[:, 1 - self.conv_size :] if self.conv_size > 1 else window[:, :0]
            return out, new_state
        x = x.transpose(1, 2)                            # [B, D, T]
        x = F.pad(x, (self.conv_size - 1, 0))            # left pad -> causal
        return self.conv(x).transpose(1, 2)


def l2norm(x, eps: float):
    """L2-normalize the last dimension (paper: eigenvalue stability for q,k).

    The reduction is done in fp32 (fp16 would underflow ``x*x`` for the many
    components below 2^-11) but the result is returned in the **input** dtype.
    (The reference version returned fp32 unconditionally -- ``.to(x.dtype)``
    after ``x = x.float()`` is a no-op -- which silently pinned q/k to fp32 and
    kept the whole KDA chunkwise kernel off the fp16 tensor cores.)
    """
    src_dtype = x.dtype
    x = x.float()
    rstd = (x * x).sum(-1, keepdim=True).rsqrt().clamp(max=1.0 / eps)
    return (x * rstd).to(src_dtype)


# --------------------------------------------------------------------------- #
# KDA attention module (paper Section 4, "Neural Parameterization")
# --------------------------------------------------------------------------- #
class KimiDeltaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        la = config.linear_attn_config
        d = config.hidden_size
        self.num_heads = la["num_heads"]
        self.head_dim = la["head_dim"]          # K
        self.value_head_dim = la["value_head_dim"]  # V
        self.conv_size = la["conv_size"]
        self.chunk_size = la["chunk_size"]
        # Row-tile size for the [tile, C, K] masked-sum buffers.  ``0`` uses the
        # full chunk (one big [C, C, K] tensor); a smaller tile trades more
        # Python iterations for a working set that fits in L2, which on the V100
        # is where the kernel is actually won or lost.
        self.tile_size = int(la.get("tile_size", 0))
        self.l2norm_eps = config.l2norm_eps
        H, K, V = self.num_heads, self.head_dim, self.value_head_dim
        bias = la["proj_bias"]

        self.in_proj_q = nn.Linear(d, H * K, bias=bias)
        self.in_proj_k = nn.Linear(d, H * K, bias=bias)
        self.in_proj_v = nn.Linear(d, H * V, bias=bias)
        conv_bias = la["conv_bias"]
        self.conv_q = ShortConv(H * K, self.conv_size, bias=conv_bias)
        self.conv_k = ShortConv(H * K, self.conv_size, bias=conv_bias)
        self.conv_v = ShortConv(H * V, self.conv_size, bias=conv_bias)

        # fine-grained forget gate: low-rank W_up @ W_down (rank = head_dim by
        # default) + per-channel static A_log / dt_bias, f(x) as in GDN/Mamba
        self.gk_proj = nn.Sequential(
            nn.Linear(d, la["gate_low_rank"], bias=bias),
            nn.Linear(la["gate_low_rank"], H * K, bias=bias),
        )
        # config-driven static init values, stored so that a *meta-device*
        # build + materialization (kimi_linear.fp16.materialize_fp16_) can
        # reproduce the constructor init afterwards: on meta tensors
        # ``torch.zeros + init`` carries shape only, and to_empty leaves the
        # storage uninitialised.  KimiLinearForCausalLM._init_weights uses
        # these to reset A_log/dt_bias after materialization.
        self._a_log_init = float(la["a_log_init"])
        self._dt_bias_init = float(la["dt_bias_init"])
        self.A_log = nn.Parameter(torch.zeros(H, K) + self._a_log_init)
        self.dt_bias = nn.Parameter(torch.zeros(H, K) + self._dt_bias_init)

        self.beta_proj = nn.Linear(d, H, bias=bias)

        # sigmoid output gate, low-rank parameterization (paper Eq. 10)
        self.og_proj = nn.Sequential(
            nn.Linear(d, la["output_gate_low_rank"], bias=bias),
            nn.Linear(la["output_gate_low_rank"], H * V, bias=bias),
        )
        self.g_norm = HeadwiseRMSNorm(H, V, config.rms_norm_eps)  # per-head weight [H, V]
        self.o_proj = nn.Linear(H * V, d, bias=bias)
        self._cast_cache = {}                  # see precision.cached_cast

        # fp32 master recurrent state -- see the module docstring.  On for
        # fp16/bf16 activations (the V100 / AMP path), off in fp32 where it
        # would be a no-op anyway.  Cost: B*H*K*V*4 bytes per layer.
        self.fp32_recurrent_state = True

    # ------------------------------------------------------------------ #
    def _state_dtype(self, x):
        """dtype used for the carried recurrent state ``S``."""
        if self.fp32_recurrent_state and x.dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return None                                  # -> kernel keeps v.dtype

    # ------------------------------------------------------------------ #
    def _projections(self, x, conv_cache=None, use_cache=False):
        """Shared projection path for prefill and decoding."""
        B, T, _ = x.shape
        H, K, V = self.num_heads, self.head_dim, self.value_head_dim
        keep = max(self.conv_size - 1, 0)

        new_conv_cache = None
        if use_cache and T == 1:
            # single-token incremental path driven by the conv history
            q_conv_in = self.in_proj_q(x)
            k_conv_in = self.in_proj_k(x)
            v_conv_in = self.in_proj_v(x)
            q_conv_in, cq_s = self.conv_q(q_conv_in, conv_cache["q"], use_cache=True)
            k_conv_in, ck_s = self.conv_k(k_conv_in, conv_cache["k"], use_cache=True)
            v_conv_in, cv_s = self.conv_v(v_conv_in, conv_cache["v"], use_cache=True)
            new_conv_cache = {"q": cq_s, "k": ck_s, "v": cv_s}
        else:
            # batch path: causal conv over the whole window
            q_in = self.in_proj_q(x)
            k_in = self.in_proj_k(x)
            v_in = self.in_proj_v(x)
            has_history = use_cache and conv_cache is not None and keep > 0
            if has_history:
                # chunked-prefill extension: continue from the cached conv inputs
                q_full = torch.cat([conv_cache["q"], q_in], dim=1)
                k_full = torch.cat([conv_cache["k"], k_in], dim=1)
                v_full = torch.cat([conv_cache["v"], v_in], dim=1)
                q_conv_in = self.conv_q(q_full)[:, keep:]
                k_conv_in = self.conv_k(k_full)[:, keep:]
                v_conv_in = self.conv_v(v_full)[:, keep:]
            else:
                q_conv_in = self.conv_q(q_in)
                k_conv_in = self.conv_k(k_in)
                v_conv_in = self.conv_v(v_in)
            if use_cache:
                # remember the last (conv_size - 1) projected inputs
                if has_history:
                    new_conv_cache = {
                        "q": q_full[:, -keep:], "k": k_full[:, -keep:], "v": v_full[:, -keep:],
                    }
                else:
                    new_conv_cache = {
                        "q": q_in[:, -keep:] if keep else q_in[:, :0],
                        "k": k_in[:, -keep:] if keep else k_in[:, :0],
                        "v": v_in[:, -keep:] if keep else v_in[:, :0],
                    }

        q = F.silu(q_conv_in).view(B, T, H, K)
        k = F.silu(k_conv_in).view(B, T, H, K)
        v = F.silu(v_conv_in).view(B, T, H, V)
        q = l2norm(q, self.l2norm_eps)
        k = l2norm(k, self.l2norm_eps)

        beta = torch.sigmoid(self.beta_proj(x).float())          # [B,T,H]
        gk = self.gk_proj(x).view(B, T, H, K)
        # f(.): per-channel log decay as in GDN/Mamba: g = -exp(A_log) * softplus(.)
        # A_log / dt_bias are static per-channel parameters: cache their fp32
        # view (recomputed only if the parameter is updated) so the decode
        # path stops re-casting them every generated token.
        a_log = cached_cast(self.A_log, self._cast_cache)
        dt_bias = cached_cast(self.dt_bias, self._cast_cache)
        g = -torch.exp(a_log) * F.softplus(gk.float() + dt_bias)
        return q, k, v, g, beta, new_conv_cache

    # ------------------------------------------------------------------ #
    def _output(self, o_core, x):
        """Head-wise RMSNorm + sigmoid output gate + out projection (Eq. 10)."""
        B, T, H, V = o_core.shape
        o = self.g_norm(o_core)                                   # norm over V per head
        gate = torch.sigmoid(self.og_proj(x).view(B, T, H, V).float())
        o = (o.float() * gate).to(o.dtype)
        return self.o_proj(o.reshape(B, T, H * V))

    # ------------------------------------------------------------------ #
    def forward(self, x, state=None, use_cache=False):
        """x: [B, T, d].

        ``state`` is a dict ``{"conv": {"q","k","v"}, "recurrent": [B,H,K,V]}``
        used for incremental decoding / chunked prefill. When ``use_cache`` is
        True the updated state dict is returned as the second element.
        """
        if use_cache and state is None:
            B = x.shape[0]
            state = {
                "conv": {
                    name: x.new_zeros(B, max(self.conv_size - 1, 0), dim)
                    for name, dim in (
                        ("q", self.num_heads * self.head_dim),
                        ("k", self.num_heads * self.head_dim),
                        ("v", self.num_heads * self.value_head_dim),
                    )
                },
                "recurrent": None,
            }
        conv_cache = state["conv"] if state is not None else None
        rec_state = state["recurrent"] if state is not None else None

        q, k, v, g, beta, new_conv = self._projections(
            x, conv_cache=conv_cache, use_cache=use_cache
        )

        B, T = x.shape[0], x.shape[1]
        sdtype = self._state_dtype(x)
        if use_cache and T == 1:
            o_core, new_rec = kda_recurrent(q, k, v, g, beta, state=rec_state,
                                            state_dtype=sdtype)
        else:
            # pad T to a multiple of the chunk size, chunkwise on the body and
            # run the remainder through the exact recurrent kernel.
            # If the sequence is shorter than one chunk, fall back to a single
            # chunk covering it: with body == 0 every position would go through
            # the token-by-token recurrent loop, which on the 1.5B model turns a
            # 64-token prefill into ~1 s instead of ~30 ms.
            C = self.chunk_size
            if 0 < T < C:
                C = T
            rem = T % C
            body = T - rem
            o_core = x.new_zeros(B, T, self.num_heads, self.value_head_dim)
            if body > 0:
                o_body, rec_state = kda_chunkwise(
                    q[:, :body], k[:, :body], v[:, :body], g[:, :body], beta[:, :body],
                    initial_state=rec_state, chunk_size=C, state_dtype=sdtype,
                    tile_size=self.tile_size,
                )
                o_core[:, :body] = o_body
            if rem > 0:
                o_tail, rec_state = kda_recurrent(
                    q[:, body:], k[:, body:], v[:, body:], g[:, body:], beta[:, body:],
                    state=rec_state, state_dtype=sdtype,
                )
                o_core[:, body:] = o_tail
            new_rec = rec_state

        out = self._output(o_core, x)
        if use_cache:
            return out, {"conv": new_conv, "recurrent": new_rec}
        return out
