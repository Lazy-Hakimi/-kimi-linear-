"""Pure-FP16 weight training for the sm_70 (Tesla V100-SXM2-16GB) target.

Why this module exists
----------------------
The V100 has FP16 tensor cores but *no* BF16 and *no* TF32 (measured:
2048x2048 GEMM = 0.29 ms in fp16 vs 1.50 ms in fp32 -- 5.1x).  The usual way
to exploit that is AMP: keep fp32 master weights and let autocast cast them to
fp16 for every matmul.  That costs

    fp32 master   6.04 GB
    fp32 grads    6.04 GB   (or 3.02 GB with fp16 grads)
    AdamW m+v    12.08 GB
    --------------------
    >= 21 GB  -> does not fit in 16 GB

This module implements the alternative the project asks for: **the weights
themselves are fp16**.  Storing parameters in fp16 halves both the weight and
the gradient footprint and -- equally important -- *removes the per-step cast*
that AMP pays on every single GEMM (autocast has to materialise an fp16 copy
of each fp32 weight before each matmul; with fp16 weights that copy does not
exist).  Only LayerNorm-style normalisation parameters stay fp32, because a
normalisation weight multiplies a *normalised* residual and its gradient is
tiny relative to its value -- exactly the regime where fp16 stalls.

The price of dropping fp32 masters: an AdamW update is ~1e-4 relative to the
weight, and fp16 has 10 mantissa bits (~5e-4 relative resolution), so a naive
``p.add_(-lr * update)`` rounds to *zero* on most steps and the model stops
learning.  Two standard, algorithm-preserving fixes are provided:

1. **Kahan summation** (``kahan=True``, default).  Each fp16 parameter carries
   an fp16 *compensation* tensor holding the part of the update that fp16
   could not represent; the next step adds it back before rounding.  This is
   the same trick as a fp32 master weight but costs 2 bytes/param instead of 4
   and keeps the *stored* model in fp16.  Mathematically the optimiser still
   performs ``p <- p - lr * update``; only the rounding is compensated.

2. **8-bit block-wise optimiser state** (``bits=8``, default).  AdamW's second
   moment does not need 32 bits.  Following bitsandbytes, the state is
   quantised per 2048-element block with an absmax scale, giving
   ~1.002 bytes/param for ``m`` and the same for ``v`` instead of 8.

Memory ledger for the 1.5B config (1.51e9 params, ``bits=8``, ``kahan=True``):

    fp16 weights                 3.02 GB
    fp16 gradients               3.02 GB
    8-bit m + v (+ scales)       3.02 GB
    fp16 Kahan compensation      3.02 GB
    -------------------------------------
    optimiser + weights         12.08 GB   -> ~3.9 GB left for activations

Dropping Kahan (``kahan=False``) gives 9.06 GB; using Adafactor instead of
AdamW (:class:`KahanAdafactor`) gives ~9.2 GB.  See ``memory_report``.

Numerical-safety invariants
---------------------------
* every fp32->fp16 store is clamped to +-60000 (fp16 max is 65504) so a
  diverging update saturates instead of producing inf/NaN;
* the optimiser never writes a non-finite value: if the update is not finite
  the parameter is left untouched and counted in ``overflow_stats``;
* ``bits=8`` scales are clamped to >=1e-12 so an all-zero state block cannot
  produce 0/0 in the dequantised division.

Everything here is pure PyTorch -- no bitsandbytes / triton / apex, none of
which exist in the target environment.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "FP32_NORM_MODULES",
    "KahanAdamW8bit",
    "KahanAdafactor",
    "FP16GradScaler",
    "kahan_apply_",
    "quantize_blockwise",
    "dequantize_blockwise",
    "cast_for_fp16_training",
    "materialize_fp16_",
    "assert_fp16_layout",
    "param_dtype_report",
    "memory_report",
    "chunked_linear_ce",
    "find_nonfinite",
    "FP16_MAX_SAFE",
]


# fp16 max normal is 65504; clamp a bit below so a later accumulation of the
# gradient cannot push a stored weight into inf.
FP16_MAX_SAFE = 60000.0
_QMAX = 127.0
_DEFAULT_BLOCK = 2048
# Elements processed per optimiser pass.  Measured sweep on the 1.5B model
# (V100): 0.25M -> 10.7 s, 1M -> 3.2 s, 8M -> 1.06 s, 16M -> 0.92 s, 32M ->
# 0.94 s.  The step is launch-bound, not L2-bound, so bigger is better until
# the fp32 scratch (about 8 x chunk x 4 B) starts to hurt, which it does at 32M.
_DEFAULT_CHUNK = 16 * 1024 * 1024

# Modules whose parameters must stay fp32 even in a pure-fp16 model.
FP32_NORM_MODULES = (nn.LayerNorm,)


def _keep_fp32_module(module: nn.Module) -> bool:
    """True when every parameter of ``module`` must remain fp32."""
    if isinstance(module, FP32_NORM_MODULES):
        return True
    return bool(getattr(module, "_keep_fp32_params", False))


# --------------------------------------------------------------------------- #
# 8-bit block-wise absmax quantisation (bitsandbytes scheme, pure PyTorch)
# --------------------------------------------------------------------------- #
def quantize_blockwise(x: torch.Tensor, blocksize: int = _DEFAULT_BLOCK
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """``[N] fp32 -> ([N_pad] int8, [N_pad/blocksize] fp32)``.

    ``N_pad`` is ``N`` rounded up to a whole number of blocks; the padding is
    zero in both the codes and (because the scale is an *absmax*) harmless on
    dequantisation: a zero-padded element dequantises to 0.
    """
    n = x.numel()
    n_pad = ((n + blocksize - 1) // blocksize) * blocksize
    if n_pad != n:
        x = torch.cat([x.reshape(-1), x.new_zeros(n_pad - n)])
    xb = x.reshape(-1, blocksize)
    absmax = xb.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    codes = (xb * (_QMAX / absmax)).round_().clamp_(-128.0, 127.0).to(torch.int8)
    return codes.reshape(-1), absmax.squeeze(1).to(torch.float32)


def dequantize_blockwise(codes: torch.Tensor, scale: torch.Tensor,
                         blocksize: int = _DEFAULT_BLOCK,
                         numel: Optional[int] = None) -> torch.Tensor:
    """Inverse of :func:`quantize_blockwise` (returns fp32)."""
    x = codes.to(torch.float32).reshape(-1, blocksize) * (scale.to(torch.float32).unsqueeze(1) / _QMAX)
    x = x.reshape(-1)
    return x if numel is None else x[:numel]


# --------------------------------------------------------------------------- #
# Kahan-compensated in-place update
# --------------------------------------------------------------------------- #
def kahan_apply_(p: torch.Tensor, comp: torch.Tensor, delta: torch.Tensor,
                 max_abs: float = FP16_MAX_SAFE) -> None:
    """``p <- p - delta`` with fp16 Kahan compensation (in-place).

    ``comp`` holds the residual ``P_exact - p_stored`` of the *previous* step,
    so the exact value we want to store is ``p_stored + comp - delta``.  After
    rounding to fp16 the new residual is written back to ``comp``.

    This is a numerically compensated implementation of the plain AdamW update
    -- the optimiser's algorithm is unchanged, only the rounding error is
    carried forward instead of being discarded.
    """
    new = p.float().add_(comp.float()).sub_(delta)
    new.clamp_(-max_abs, max_abs)                    # saturate, never inf
    p.copy_(new.to(p.dtype))
    comp.copy_((new - p.float()).to(comp.dtype))


# --------------------------------------------------------------------------- #
# AdamW: fp16 params + optional fp16 Kahan + optional 8-bit state
# --------------------------------------------------------------------------- #
class KahanAdamW8bit(torch.optim.Optimizer):
    """AdamW for **fp16 parameters**, with 8-bit state and Kahan compensation.

    Args:
        bits: 8 -> block-wise int8 moment storage (default, ~1.002 B/param each);
              32 -> plain fp32 moments (4 B/param each), for A/B comparisons.
        kahan: keep an fp16 compensation tensor per low-precision parameter
              (2 B/param).  Disable only if you have measured that you do not
              need it -- without it most AdamW updates round away to nothing.
        blocksize: quantisation block length for ``bits=8``.
        chunk: elements processed per inner pass.  Bounding this keeps the
              fp32 scratch buffers (5 x chunk x 4 B) small enough that a
              268 M-parameter embedding table cannot spike the allocator.
              It is always rounded to a multiple of ``blocksize``.
        max_abs: saturation limit for fp16 stores.
    """

    def __init__(self, params, lr: float = 1e-4, betas: Sequence[float] = (0.9, 0.95),
                 eps: float = 1e-8, weight_decay: float = 0.1,
                 bits: int = 8, kahan: bool = True,
                 blocksize: int = _DEFAULT_BLOCK,
                 chunk: int = _DEFAULT_CHUNK,
                 max_abs: float = FP16_MAX_SAFE):
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid betas: {betas}")
        if bits not in (8, 32):
            raise ValueError("bits must be 8 or 32")
        if blocksize <= 0 or chunk <= 0:
            raise ValueError("blocksize/chunk must be positive")
        defaults = dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay,
                        bits=bits, kahan=kahan, blocksize=blocksize,
                        max_abs=max_abs)
        super().__init__(params, defaults)
        self.chunk = chunk
        self._nonfinite = 0          # updates rejected because the delta blew up
        self._saturated = 0          # parameters hit +-max_abs (informational)

    # ------------------------------------------------------------------ #
    def _init_state(self, p: torch.Tensor, st: dict, group: dict) -> None:
        n = p.numel()
        nb = group["blocksize"]
        n_pad = ((n + nb - 1) // nb) * nb
        st["numel"] = n
        st["n_pad"] = n_pad
        st["step"] = 0
        if group["bits"] == 8:
            st["m"] = torch.zeros(n_pad, dtype=torch.int8, device=p.device)
            st["v"] = torch.zeros(n_pad, dtype=torch.int8, device=p.device)
            st["m_scale"] = torch.zeros(n_pad // nb, dtype=torch.float32, device=p.device)
            st["v_scale"] = torch.zeros(n_pad // nb, dtype=torch.float32, device=p.device)
        else:
            st["m"] = torch.zeros(n_pad, dtype=torch.float32, device=p.device)
            st["v"] = torch.zeros(n_pad, dtype=torch.float32, device=p.device)
            st["m_scale"] = st["v_scale"] = None
        # Kahan compensation only makes sense for a low-precision parameter;
        # fp32 parameters (LayerNorms) already hold every bit of the update.
        if group["kahan"] and p.dtype in (torch.float16, torch.bfloat16):
            st["comp"] = torch.zeros(n, dtype=p.dtype, device=p.device)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Non-finite bookkeeping is accumulated on the GPU and read back **once
        # per step**.  A per-chunk ``isfinite().all()`` would cost two host
        # syncs per chunk -- ~380 per step at 1.5B -- and every sync drains the
        # CUDA queue, which alone made the optimiser 70x slower than its memory
        # traffic requires.
        finite_chk = None

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            nb = group["blocksize"]
            use8 = group["bits"] == 8
            max_abs = group["max_abs"]
            chunk = max(int(self.chunk // nb), 1) * nb

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError(f"{type(self).__name__} does not support sparse gradients")
                st = self.state[p]
                if len(st) == 0:
                    self._init_state(p, st, group)
                if p.numel() != st["numel"]:
                    raise RuntimeError("parameter was resized after optimiser construction")
                st["step"] += 1
                t = st["step"]
                bc1 = 1.0 - b1 ** t
                bc2 = 1.0 - b2 ** t
                p_contig = p.is_contiguous()
                p_flat = p.data.reshape(-1) if p_contig else p.data.flatten()
                g_flat = p.grad.reshape(-1) if p.grad.is_contiguous() else p.grad.flatten()
                comp = st.get("comp")
                n, n_pad = st["numel"], st["n_pad"]

                for s in range(0, n_pad, chunk):
                    e = min(s + chunk, n_pad)
                    L = e - s
                    ve = min(e, n)
                    vl = ve - s
                    nblk = L // nb               # chunk and n_pad are block aligned
                    b0, b1i = s // nb, e // nb

                    g = g_flat[s:ve].to(torch.float32)
                    ps = p_flat[s:ve].to(torch.float32)
                    if vl < L:                   # tail block: zero-pad
                        g = torch.cat([g, g.new_zeros(L - vl)])
                        ps = torch.cat([ps, ps.new_zeros(L - vl)])

                    # Work in [nblk, nb] so per-block scales broadcast without
                    # extra copies.
                    g = g.reshape(nblk, nb)
                    ps = ps.reshape(nblk, nb)
                    if use8:
                        ms = st["m_scale"][b0:b1i].unsqueeze(1)
                        vs = st["v_scale"][b0:b1i].unsqueeze(1)
                        # int8 * fp32 promotes to fp32 in ONE kernel; the
                        # two-step ``.to(float32).mul_(scale)`` costs an extra
                        # full read+write of a 64 MB fp32 buffer per chunk.
                        mm = torch.mul(st["m"][b0 * nb:b1i * nb].reshape(nblk, nb),
                                       ms / _QMAX)
                        vv = torch.mul(st["v"][b0 * nb:b1i * nb].reshape(nblk, nb),
                                       vs / _QMAX)
                    else:
                        mm = st["m"][s:e].reshape(nblk, nb).clone()
                        vv = st["v"][s:e].reshape(nblk, nb).clone()
                        ms = vs = None

                    mm.mul_(b1).add_(g, alpha=1.0 - b1)
                    vv.mul_(b2).addcmul_(g, g, value=1.0 - b2)

                    # Adam step.  ``mm``/``vv`` must go back into the state
                    # *uncorrected* -- the bias-corrected copies live in their
                    # own buffers.  (Dividing them in place and then storing is
                    # the classic way to destroy an Adam: the moments drift by
                    # 1/bc per step and the update collapses to sign(g).)
                    upd = mm.div(bc1)
                    vh = vv.div(bc2).clamp_(min=0.0).sqrt_()
                    if use8:
                        # 8-bit floor for the denominator.  int8 resolves v only
                        # down to one quantisation step (block_absmax/127), so a
                        # small-but-real v can dequantise to *exactly* zero;
                        # 1/sqrt(0+eps) is then a 1/1e-8 spike that blows the
                        # weights up in a single step.  Clamping sqrt(v) to the
                        # square root of one step means "below the resolution of
                        # the state, treat v as the smallest value it can hold".
                        vh = torch.maximum(vh, (vs / (_QMAX * bc2)).clamp_(min=0.0).sqrt())
                    upd.div_(vh.add_(eps))
                    if wd != 0.0:
                        upd.add_(ps, alpha=wd)
                    upd.mul_(lr)

                    if comp is not None:
                        c = comp[s:ve].to(torch.float32)
                        if vl < L:
                            c = torch.cat([c, c.new_zeros(L - vl)])
                        new = ps.add_(c.reshape(nblk, nb)).sub_(upd)
                    else:
                        new = ps.sub_(upd)

                    # Guarantee the stored value is finite *without* a host sync:
                    # nan_to_num makes inf/nan unreachable whatever the update
                    # produced.  (nan survives clamp(), so it must be handled
                    # explicitly -- one diverging chunk would otherwise write
                    # NaN into the weights and kill the model.)  Non-finite
                    # *detection* lives in the trainer's periodic sweep.
                    new = new.reshape(-1).nan_to_num_(nan=0.0, posinf=max_abs,
                                                      neginf=-max_abs)
                    p_flat[s:ve].copy_(new[:vl])          # copy_ also casts
                    if comp is not None:
                        # residual = exact target - what fp16 could store; read
                        # p back so the residual uses the *rounded* value.
                        comp[s:ve].copy_(new[:vl] - p_flat[s:ve].to(torch.float32))

                    # re-quantise the moments for this chunk
                    if use8:
                        s_m = mm.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
                        st["m"][b0 * nb:b1i * nb] = (
                            (mm * (_QMAX / s_m)).round_().clamp_(-128.0, 127.0)
                            .to(torch.int8).reshape(-1))
                        st["m_scale"][b0:b1i] = s_m.squeeze(1)
                        s_v = vv.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
                        st["v"][b0 * nb:b1i * nb] = (
                            (vv * (_QMAX / s_v)).round_().clamp_(-128.0, 127.0)
                            .to(torch.int8).reshape(-1))
                        st["v_scale"][b0:b1i] = s_v.squeeze(1)
                    else:
                        st["m"][s:e] = mm.reshape(-1)
                        st["v"][s:e] = vv.reshape(-1)

                if not p_contig:
                    p.data.copy_(p_flat.reshape_as(p))
        return loss

    # ------------------------------------------------------------------ #
    def capture_graph(self, warmup: int = 3):
        """Capture one ``step()`` into a :class:`torch.cuda.CUDAGraph`.

        The step is ~4300 tiny elementwise kernels at 1.5B, so launch overhead
        is a large fraction of its cost.  A graph replays them with a single
        launch.  Requires

        * no host sync inside ``step`` (hence the async finite-value handling),
        * every tensor address to stay valid: call ``zero_grad(set_to_none=False)``
          so autograd reuses the same gradient buffers, and do not resize or
          replace parameters between capture and replay.

        Returns the graph; the caller runs ``graph.replay()`` instead of
        ``opt.step()``.  Returns None on non-CUDA or if capture fails.
        """
        if not torch.cuda.is_available():
            return None
        for _ in range(warmup):
            self.step()
        torch.cuda.synchronize()
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.step()
        except Exception as exc:                                   # noqa: BLE001
            print(f"[fp16] CUDA graph capture failed ({type(exc).__name__}: {exc}); "
                  f"falling back to eager step")
            return None
        return graph

    # ------------------------------------------------------------------ #
    @property
    def overflow_stats(self) -> Dict[str, int]:
        return {"nonfinite_updates": self._nonfinite, "saturated_params": self._saturated}


# --------------------------------------------------------------------------- #
# Adafactor: fp16 params + Kahan.  Zero second-moment state (rank-1 factor).
# --------------------------------------------------------------------------- #
class KahanAdafactor(torch.optim.Optimizer):
    """Adafactor (Shazeer & Stern, 2018) adapted to fp16 parameters.

    Same algorithm as ``kimi_musicgpt.optim.Adafactor`` -- rank-1
    factorisation of the second moment for every 2-D parameter, update-RMS
    clipping, decoupled weight decay -- but writes its update through
    :func:`kahan_apply_` so it can drive fp16 weights without a master copy.

    State cost is ~0 for 2-D parameters (two row/col vectors of fp32), which
    makes this the *lowest memory* option: ~9.2 GB total for the 1.5B model
    including the Kahan compensation.
    """

    def __init__(self, params, lr: float = 1e-3, eps=(1e-30, 1e-3),
                 clip_threshold: float = 1.0, decay_rate: float = -0.8,
                 beta1: Optional[float] = None, weight_decay: float = 0.0,
                 kahan: bool = True, max_abs: float = FP16_MAX_SAFE,
                 factored: bool = True, max_factored_numel: int = 64_000_000):
        if lr <= 0.0:
            raise ValueError(f"invalid lr: {lr}")
        defaults = dict(lr=lr, eps=eps, clip_threshold=clip_threshold,
                        decay_rate=decay_rate, beta1=beta1,
                        weight_decay=weight_decay, kahan=kahan,
                        max_abs=max_abs, factored=factored,
                        max_factored_numel=int(max_factored_numel))
        super().__init__(params, defaults)
        self._nonfinite = 0

    # ------------------------------------------------------------------ #
    @staticmethod
    def _rms(x: torch.Tensor) -> torch.Tensor:
        return x.norm() / (x.numel() ** 0.5)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps1, eps2 = group["eps"]
            factored = group["factored"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("KahanAdafactor does not support sparse gradients")
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    # Factoring is only a win while the *transient* [r, c] fp32
                    # buffers stay small.  For the 131072x2048 embedding (268 M
                    # params) even the factored form has to build a full-size
                    # ``update``, and the rank-1 reconstruction was costing two
                    # extra [r, c] fp32 temporaries (2.1 GB) on top of that --
                    # measured peak 17.5 GiB, i.e. off the 16 GB card.
                    use_factored = (factored and p.dim() >= 2
                                    and p.numel() <= group["max_factored_numel"])
                    st["factored"] = use_factored
                    if use_factored:
                        st["row"] = torch.zeros(p.shape[:-1], dtype=torch.float32, device=p.device)
                        st["col"] = torch.zeros(p.shape[:-2] + p.shape[-1:],
                                                dtype=torch.float32, device=p.device)
                    else:
                        st["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    if group["beta1"] is not None:
                        st["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    if group["kahan"] and p.dtype in (torch.float16, torch.bfloat16):
                        st["comp"] = torch.zeros_like(p)
                st["step"] += 1
                t = st["step"]
                beta2_t = 1.0 - t ** group["decay_rate"]

                g = grad.to(torch.float32)
                if not bool(torch.isfinite(g).all()):
                    self._nonfinite += 1
                    continue
                g2 = g * g + eps1

                if st.get("factored"):
                    row, col = st["row"], st["col"]
                    row.mul_(beta2_t).add_(g2.mean(dim=-1), alpha=1.0 - beta2_t)
                    col.mul_(beta2_t).add_(g2.mean(dim=-2), alpha=1.0 - beta2_t)
                    # v = (row (x) col) / mean(row).  Build sqrt(v) straight from
                    # the two broadcast factors instead of materialising v and
                    # then its sqrt: sqrt(a (x) b) == sqrt(a) (x) sqrt(b), and
                    # both factors are non-negative.  Saves two full [r, c] fp32
                    # buffers (1.07 GB each for the 1.5B embedding table).
                    r_hat = row / row.mean(dim=-1, keepdim=True).clamp(min=eps1)
                    denom = (r_hat.clamp(min=0.0).sqrt().unsqueeze(-1)
                             * col.clamp(min=0.0).sqrt().unsqueeze(-2))
                    update = g / (denom + eps2)
                else:
                    v = st["exp_avg_sq"]
                    v.mul_(beta2_t).add_(g2, alpha=1.0 - beta2_t)
                    update = g / (v.clamp(min=0.0).sqrt_() + eps2)
                if group["clip_threshold"] > 0.0:
                    rms = self._rms(update)
                    update = update / (rms / group["clip_threshold"]).clamp(min=1.0)
                if group["beta1"] is not None:
                    m = st["exp_avg"]
                    m.mul_(group["beta1"]).add_(update, alpha=1.0 - group["beta1"])
                    update = m

                lr = group["lr"]
                delta = update * lr
                if not bool(torch.isfinite(delta).all()):
                    self._nonfinite += 1
                    continue
                if group["weight_decay"] != 0.0:
                    pf = p.float().mul_(1.0 - lr * group["weight_decay"])
                    p.copy_(pf.to(p.dtype))
                if "comp" in st:
                    kahan_apply_(p, st["comp"], delta, max_abs=group["max_abs"])
                else:
                    p.copy_((p.float() - delta).clamp(-group["max_abs"],
                                                      group["max_abs"]).to(p.dtype))
        return loss

    # ------------------------------------------------------------------ #
    @property
    def overflow_stats(self) -> Dict[str, int]:
        return {"nonfinite_updates": self._nonfinite, "saturated_params": 0}


# --------------------------------------------------------------------------- #
# Model layout helpers
# --------------------------------------------------------------------------- #
def cast_for_fp16_training(model: nn.Module, dtype: torch.dtype = torch.float16,
                           verbose: bool = True) -> Dict[str, float]:
    """Put ``model`` in the pure-fp16 training layout.

    Every parameter becomes ``dtype`` **except** those of normalisation
    modules (``nn.LayerNorm`` and anything flagged ``_keep_fp32_params``),
    which are forced back to fp32 -- a normalisation weight multiplies a
    normalised activation, so its update/value ratio is the worst in the
    network and it is the one place fp16 demonstrably stalls.

    Returns a small report (parameter counts in millions per dtype).
    """
    model.to(dtype)
    fp32_params = 0
    for module in model.modules():
        if _keep_fp32_module(module):
            module.float()
            fp32_params += sum(q.numel() for q in module.parameters(recurse=False))

    total = sum(p.numel() for p in model.parameters())
    low = total - fp32_params
    report = {
        "total_M": total / 1e6,
        "low_precision_M": low / 1e6,
        "fp32_norm_M": fp32_params / 1e6,
        "bytes_MiB": (low * torch.empty(0, dtype=dtype).element_size()
                      + fp32_params * 4) / 2 ** 20,
    }
    if verbose:
        print(f"[fp16 layout] {report['low_precision_M']:.1f}M params @ "
              f"{str(dtype).replace('torch.', '')}, "
              f"{report['fp32_norm_M']:.3f}M @ fp32 (norms) | "
              f"weights {report['bytes_MiB']:.1f} MiB")
    return report


def materialize_fp16_(module: nn.Module, device, dtype: torch.dtype = torch.float16
                      ) -> nn.Module:
    """Realise a ``meta``-device module directly in the pure-fp16 layout.

    Building a 1.5B model normally, then calling ``.half()``, transiently needs
    **6 GB** of fp32 storage on the target device (or 6 GB of host RAM if it is
    built on CPU) on top of the 3 GB fp16 copy.  On a 16 GB card that alone can
    be the difference between fitting and OOM.  Setting the *meta* dtype first
    costs nothing -- meta tensors carry only shape/dtype -- so the real
    allocation is exactly the bytes the final layout needs:

        with torch.device("meta"):
            model = KimiLinearForCausalLM(cfg)
        materialize_fp16_(model, "cuda")
        model.apply(model._init_weights)

    Tied weights need care.  ``Module.to_empty`` lowers to ``Module._apply``,
    which builds a **fresh** ``nn.Parameter`` object per module and installs it
    in that module's ``_parameters``.  A shared parameter
    (``tie_word_embeddings``: ``lm_head.weight is embed_tokens.weight``) is
    therefore visited twice and comes out as two independent tensors.  On the
    1.5B config that silently adds 268 M duplicated parameters -- 0.5 GB of
    weights, 0.5 GB of gradients, 0.5 GB of Kahan state -- and, worse, unties
    the output head from the input embedding.  The sharing is re-established
    after materialisation by pointing every duplicate slot at the canonical one.

    Parameters are uninitialised after this call; the caller runs its init.
    """
    for m in module.modules():
        want = torch.float32 if _keep_fp32_module(m) else dtype
        for _, q in m.named_parameters(recurse=False):
            if q.dtype != want:
                q.data = q.data.to(want)      # meta: metadata only
        for _, b in m.named_buffers(recurse=False):
            if b.is_floating_point() and b.dtype != want:
                b.data = b.data.to(want)

    # record sharing BEFORE _apply replaces the Parameter objects
    seen: Dict[int, tuple] = {}
    dupes: List[tuple] = []
    for m in module.modules():
        for name, q in m.named_parameters(recurse=False):
            key = id(q)
            if key in seen:
                dupes.append((m, name, True) + seen[key])
            else:
                seen[key] = (m, name)
        for name, b in m.named_buffers(recurse=False):
            key = id(b)
            if key in seen:
                dupes.append((m, name, False) + seen[key])
            else:
                seen[key] = (m, name)

    module.to_empty(device=device)

    for m, name, is_param, cm, cn in dupes:
        table = cm._parameters if is_param else cm._buffers
        src = table[cn]
        (m._parameters if is_param else m._buffers)[name] = src
    return module


# --------------------------------------------------------------------------- #
# Loss scaling for pure-fp16 parameters
# --------------------------------------------------------------------------- #
class FP16GradScaler:
    """GradScaler that works when the **parameters themselves** are fp16.

    ``torch.amp.GradScaler.unscale_`` hard-codes ``allow_fp16=False`` and raises
    ``ValueError("Attempting to unscale FP16 gradients.")`` -- it assumes fp16
    gradients can only mean a misconfigured AMP setup.  Here they are the whole
    point, so the check-and-unscale is driven directly through the fused CUDA
    kernel ``GradScaler`` itself uses internally.  That kernel

    * handles fp16 gradients (it is templated on the scalar type), and
    * does the non-finite scan *and* the division in **one** pass, so there is
      no per-parameter CPU sync (a naive ``isfinite().all()`` loop costs ~200
      host round-trips per step at 1.5B).

    Protocol is the standard AMP one:

        scaler.scale_loss(loss).backward()
        if scaler.unscale_(params):        # True => inf/nan found, grads dropped
            scaler.update(found_inf=True); opt.zero_grad(); continue
        clip_grad_norm_(params, 1.0)       # always AFTER the unscale
        opt.step(); scaler.update(False)
    """

    def __init__(self, init_scale: float = 2048.0, growth_factor: float = 2.0,
                 backoff_factor: float = 0.5, growth_interval: int = 2000,
                 min_scale: float = 1.0, max_scale: float = 65504.0,
                 device="cuda"):
        self._scale = float(init_scale)
        self._growth_factor = float(growth_factor)
        self._backoff_factor = float(backoff_factor)
        self._growth_interval = int(growth_interval)
        self._min_scale = float(min_scale)
        self._max_scale = float(max_scale)
        self._growth_tracker = 0
        self._device = device
        self.overflows = 0
        self.underflows = 0

    # ------------------------------------------------------------------ #
    def get_scale(self) -> float:
        return self._scale

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self._scale

    # ------------------------------------------------------------------ #
    def unscale_(self, params) -> bool:
        """Divide every gradient by the scale in place.

        Returns True when a non-finite gradient was found (the step must be
        skipped); the division still happened, but the grads are discarded.
        """
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            return False
        dev = grads[0].device
        found_inf = torch.zeros((), dtype=torch.float32, device=dev)
        inv_scale = torch.tensor(1.0 / self._scale, dtype=torch.float32, device=dev)
        try:
            torch._amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv_scale)
        except (RuntimeError, AttributeError):
            # CPU or an older build: kernel-per-tensor fallback.
            bad = False
            for g in grads:
                if not bool(torch.isfinite(g).all()):
                    bad = True
                    break
            if not bad:
                for g in grads:
                    g.div_(self._scale)
            return bad
        return bool(found_inf.item() > 0.0)

    # ------------------------------------------------------------------ #
    def update(self, found_inf: bool) -> None:
        if found_inf:
            self.overflows += 1
            self._scale = max(self._scale * self._backoff_factor, self._min_scale)
            self._growth_tracker = 0
            if self._scale <= self._min_scale:
                self.underflows += 1
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale = min(self._scale * self._growth_factor, self._max_scale)
                self._growth_tracker = 0

    def state_dict(self):
        return {"scale": self._scale, "growth_tracker": self._growth_tracker}

    def load_state_dict(self, sd):
        self._scale = float(sd.get("scale", self._scale))
        self._growth_tracker = int(sd.get("growth_tracker", 0))


def assert_fp16_layout(model: nn.Module, dtype: torch.dtype = torch.float16) -> List[str]:
    """Return the list of parameters that violate the fp16 layout.

    Empty list == the model is stored the way this module intends: every
    parameter is ``dtype`` except normalisation weights, which are fp32.
    """
    bad = []
    for module in model.modules():
        keep = _keep_fp32_module(module)
        for name, q in module.named_parameters(recurse=False):
            want = torch.float32 if keep else dtype
            if q.dtype != want:
                bad.append(f"{type(module).__name__}.{name}: {q.dtype} (expected {want})")
    return bad


def param_dtype_report(model: nn.Module) -> Dict[str, float]:
    """Parameter count (millions) grouped by storage dtype."""
    out: Dict[str, float] = {}
    for p in model.parameters():
        key = str(p.dtype).replace("torch.", "")
        out[key] = out.get(key, 0.0) + p.numel() / 1e6
    return out


def memory_report(model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                  bits: int = 8, kahan: bool = True) -> Dict[str, float]:
    """Estimated steady-state GPU memory in GiB (weights+grads+optimiser)."""
    total = sum(p.numel() for p in model.parameters())
    low = sum(p.numel() for p in model.parameters()
              if p.dtype in (torch.float16, torch.bfloat16))
    fp32 = total - low
    grads = sum(p.numel() * p.dtype.itemsize for p in model.parameters()) / 2 ** 30
    weights = (low * 2 + fp32 * 4) / 2 ** 30
    state = 0.0
    if optimizer is not None:
        state = sum(sum(t.numel() * t.element_size() for t in s.values()
                        if torch.is_tensor(t)) for s in optimizer.state.values()) / 2 ** 30
    elif bits == 8:
        state = (low * 2 * 1.002) / 2 ** 30
    else:
        state = (low * 8) / 2 ** 30
    kahan_bytes = (low * 2) / 2 ** 30 if kahan else 0.0
    return {
        "weights_GiB": weights,
        "grads_GiB": grads,
        "optimizer_state_GiB": state,
        "kahan_GiB": kahan_bytes,
        "total_GiB": weights + grads + state + kahan_bytes,
    }


# --------------------------------------------------------------------------- #
# Chunked lm_head + cross-entropy: kills the [B, T, V] fp32 logits spike
# --------------------------------------------------------------------------- #
def chunked_linear_ce(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor,
                      token_weight: Optional[torch.Tensor] = None,
                      ignore_index: int = -100,
                      chunk_tokens: int = 4096,
                      use_checkpoint: bool = True):
    """``F.linear`` + cross-entropy, computed in token chunks.

    With vocab 131072 a single fp32 logits tensor for ``N`` tokens costs
    ``N * 131072 * 4`` bytes -- 1.07 GB at N=2048, and ``cross_entropy`` needs
    two more of those (log_softmax output + saved softmax for backward), i.e.
    more than the entire remaining budget of a 16 GB card.  Slicing over tokens
    and wrapping each slice in gradient checkpointing keeps only
    ``chunk_tokens`` rows alive at a time; the logits are recomputed in
    backward rather than stored.

    The result is **mathematically identical** to computing the whole
    ``[N, V]`` logits block and reducing the weighted CE over it.

    Returns ``(weighted_sum, denominator)``; the caller divides.
    """
    d = hidden.shape[-1]
    h = hidden.reshape(-1, d)
    lab = labels.reshape(-1)
    tw = None if token_weight is None else token_weight.reshape(-1)
    N = h.shape[0]
    if N != lab.shape[0]:
        raise ValueError(f"hidden rows {N} != labels {lab.shape[0]}")
    chunk = max(int(chunk_tokens), 1)

    def _block(hs, ls, ws):
        logits = F.linear(hs, weight)                    # autocast keeps this fp16
        ce = F.cross_entropy(logits.float(), ls, ignore_index=ignore_index,
                             reduction="none")
        mask = (ls != ignore_index).to(ce.dtype)
        w = mask if ws is None else ws * mask
        # numerator sums weighted CE, denominator sums the same weights, so
        # ``total / denom`` is the weighted mean CE over un-ignored tokens --
        # identical to the single-block reduction the chunks replace.
        return (ce * w).sum(), w.sum()

    total = hidden.new_zeros((), dtype=torch.float32)
    denom = hidden.new_zeros((), dtype=torch.float32)
    grad_on = torch.is_grad_enabled()
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        hs, ls = h[s:e], lab[s:e]
        ws = None if tw is None else tw[s:e]
        if use_checkpoint and grad_on and hs.requires_grad:
            ts, ds = torch.utils.checkpoint.checkpoint(_block, hs, ls, ws,
                                                       use_reentrant=False)
        else:
            ts, ds = _block(hs, ls, ws)
        total = total + ts
        denom = denom + ds
    return total, denom


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def find_nonfinite(params: Sequence[torch.Tensor]) -> List[str]:
    """Names/indices of parameters or gradients containing NaN/Inf.

    One GPU->CPU sync per tensor, so call this sparsely (e.g. every 50 steps
    or when the loss stops being finite), never inside the hot loop.
    """
    bad = []
    for i, p in enumerate(params):
        if p is None:
            continue
        if not bool(torch.isfinite(p).all()):
            bad.append(f"param[{i}] {tuple(p.shape)} {p.dtype}")
        if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
            bad.append(f"grad[{i}] {tuple(p.shape)} {p.dtype}")
    return bad
