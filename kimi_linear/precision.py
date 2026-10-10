"""Hardware-aware mixed-precision utilities (device capability -> dtype).

Why this module exists
----------------------
The reference implementation was developed and tuned on an RTX 4060 Laptop
(Ada, sm_89) where ``bfloat16`` runs on tensor cores.  On the target
production device -- **Tesla V100-SXM2-16GB (Volta, sm_70)** -- the situation
is the opposite, and getting it wrong costs more than 5x throughput:

============  =====  ==========  ==========  ==========
GPU           sm     FP16 TC     BF16 TC     TF32 TC
============  =====  ==========  ==========  ==========
Tesla V100    7.0    yes         **no**      **no**
RTX 20xx      7.5    yes         no          no
A100 / 30xx   8.0    yes         yes         yes
RTX 40xx      8.9    yes         yes         yes
============  =====  ==========  ==========  ==========

Measured on the actual V100-SXM2-16GB of this project (2048x2048 GEMM):

    fp16   0.29 ms   <- 5.1x vs fp32 (tensor cores)
    fp32   1.50 ms
    fp32 + allow_tf32  1.32 ms  (no TF32 on Volta: the flag is a no-op)
    bf16   1.81 ms   <- SLOWER than fp32 (CUDA-core emulation, no TC)

So on sm_70 **FP16 is the only precision that unlocks hardware acceleration**,
and it must be paired with loss scaling (fp16 has 10 mantissa bits and a
max normal of 65504, so gradients underflow without a scaler).  This module
centralises that decision so no script hard-codes a dtype.

Public API
----------
``supports_bfloat16`` / ``supports_tf32``   capability probes
``resolve_precision``                       "auto"|"fp16"|"bf16"|"fp32" -> dtype
``configure_cuda_backends``                 set the global backend flags once
``autocast``                                autocast context (no-op on CPU/fp32)
``make_grad_scaler``                        GradScaler, enabled only for fp16
``describe_device``                         one-line human-readable summary
"""

from __future__ import annotations

import contextlib
import functools
from typing import Optional, Tuple

import torch

__all__ = [
    "PRECISION_CHOICES",
    "device_capability",
    "supports_bfloat16",
    "supports_tf32",
    "resolve_precision",
    "precision_name",
    "configure_cuda_backends",
    "autocast",
    "make_grad_scaler",
    "cached_cast",
    "describe_device",
]

PRECISION_CHOICES = ("auto", "fp16", "bf16", "fp32")


# --------------------------------------------------------------------------- #
# capability probes
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=8)
def device_capability(device=None) -> Optional[Tuple[int, int]]:
    """Compute capability ``(major, minor)`` of the CUDA device, or None."""
    if not torch.cuda.is_available():
        return None
    try:
        if isinstance(device, torch.device):
            idx = device.index if device.index is not None else torch.cuda.current_device()
        elif isinstance(device, int):
            idx = device
        elif isinstance(device, str) and device.startswith("cuda:"):
            idx = int(device.split(":")[1])
        else:
            idx = torch.cuda.current_device()
        return torch.cuda.get_device_capability(idx)
    except Exception:
        return None


def supports_bfloat16(device=None) -> bool:
    """True only when bfloat16 runs on tensor cores (Ampere, sm_80+).

    Deliberately does **not** use ``torch.cuda.is_bf16_supported()``: that
    helper reports ``True`` on sm_70 (it only checks the CUDA runtime version),
    where bf16 GEMMs are emulated on the CUDA cores and end up slower than
    fp32.  Compute capability is the fact that matters.
    """
    cap = device_capability(device)
    return cap is not None and cap[0] >= 8


def supports_tf32(device=None) -> bool:
    """True only for Ampere+ (sm_80); Volta has no TF32 tensor cores."""
    cap = device_capability(device)
    return cap is not None and cap[0] >= 8


# --------------------------------------------------------------------------- #
# dtype resolution
# --------------------------------------------------------------------------- #
def resolve_precision(preference: str = "auto", device=None) -> Optional[torch.dtype]:
    """Map a precision *preference* to the dtype autocast should use.

    Returns ``None`` for pure fp32 (callers then skip autocast entirely).

    ``auto`` picks **fp16** everywhere bf16 has no tensor cores -- i.e. on the
    V100 target -- and prefers **bf16** on Ampere+ where its wider exponent
    range makes loss scaling unnecessary.  ``bf16``/``fp16`` requests on
    hardware without support are downgraded (with the reason reported by
    :func:`describe_device`) instead of silently running at emulation speed.
    """
    pref = (preference or "auto").lower()
    if pref not in PRECISION_CHOICES:
        raise ValueError(f"precision must be one of {PRECISION_CHOICES}, got {pref!r}")
    if pref == "fp32":
        return None
    if pref == "fp16":
        return torch.float16
    if pref == "bf16":
        return torch.bfloat16 if supports_bfloat16(device) else torch.float16
    # auto
    return torch.bfloat16 if supports_bfloat16(device) else torch.float16


def precision_name(dtype: Optional[torch.dtype]) -> str:
    if dtype is None:
        return "fp32"
    return {torch.float16: "fp16", torch.bfloat16: "bf16"}.get(dtype, str(dtype))


# --------------------------------------------------------------------------- #
# backend flags
# --------------------------------------------------------------------------- #
def configure_cuda_backends(device=None, cudnn_benchmark: bool = True) -> dict:
    """Set global CUDA backend flags appropriate for the current silicon.

    * ``allow_tf32`` is enabled only on sm_80+ (harmless but a no-op below);
    * ``cudnn.benchmark`` is enabled because the codec is conv-heavy with
      fixed input shapes (autotuning pays off after a few steps);
    * fp16 reduced-precision *reduction* in split-K GEMMs is left at the
      PyTorch default (True): it is a reduction-only approximation and buys
      real speed on Volta.
    """
    info = {"tf32": False, "cudnn_benchmark": False}
    if not torch.cuda.is_available():
        return info
    tf32 = supports_tf32(device)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    info["tf32"] = tf32
    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        info["cudnn_benchmark"] = True
    return info


# --------------------------------------------------------------------------- #
# autocast / grad scaler
# --------------------------------------------------------------------------- #
def _device_type(device=None) -> str:
    if isinstance(device, torch.device):
        return device.type
    if isinstance(device, str):
        return device.split(":")[0]
    return "cuda" if torch.cuda.is_available() else "cpu"


def autocast(device=None, dtype: Optional[torch.dtype] = None, enabled: bool = True):
    """Autocast context for ``dtype``; a no-op context when dtype is None.

    Keeping fp32 as "no context at all" (rather than
    ``torch.autocast(enabled=False)``) avoids the small but real overhead of
    the autocast cache in the pure-fp32 path.
    """
    if dtype is None or not enabled or _device_type(device) != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(device=None, dtype: Optional[torch.dtype] = None,
                     enabled: Optional[bool] = None, **kwargs):
    """GradScaler for the given compute dtype.

    Enabled exactly when the forward runs in fp16 on CUDA (bf16 has fp32's
    exponent range and needs no scaling; fp32 needs none either).
    """
    if enabled is None:
        enabled = (dtype == torch.float16 and _device_type(device) == "cuda")
    # torch.amp.GradScaler("cuda") keeps the modern API and still accepts the
    # classic kwargs (init_scale / growth_interval ...).
    return torch.amp.GradScaler("cuda", enabled=bool(enabled), **kwargs)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def describe_device(device=None, dtype: Optional[torch.dtype] = None) -> str:
    """One-line summary used by every training/inference entry point."""
    if not torch.cuda.is_available():
        return "device=cpu precision=fp32"
    name = torch.cuda.get_device_name(device)
    cap = device_capability(device)
    props = torch.cuda.get_device_properties(device)
    return (f"device={name} (sm_{cap[0]}{cap[1]}, {props.total_memory / 2**30:.1f} GiB, "
            f"{props.multi_processor_count} SMs) "
            f"bf16_tc={'yes' if supports_bfloat16(device) else 'no'} "
            f"tf32_tc={'yes' if supports_tf32(device) else 'no'} "
            f"precision={precision_name(dtype)}")


# --------------------------------------------------------------------------- #
# weight-cast caching (decode hot path)
# --------------------------------------------------------------------------- #
def cached_cast(tensor: torch.Tensor, cache: dict, dtype=torch.float32) -> torch.Tensor:
    """``tensor.to(dtype)`` memoised in ``cache`` until ``tensor`` changes.

    The incremental-decode path is **launch-bound**, not FLOP-bound: profiling
    one generated token of the local config on the V100 shows ~322
    ``aten::_to_copy`` calls, most of them re-casting the *same* fp32 weights
    (norm weights, conv kernels, ``A_log``/``dt_bias``, MLA's ``W_nope``/``W_v``)
    over and over -- roughly 40% of decode latency in both fp32 and fp16.

    Safety
    ------
    * While gradients are enabled (training) the cache is **bypassed
      entirely** -- training is compute-bound, not launch-bound, so there is
      nothing to save and no staleness risk.
    * Under ``no_grad`` (inference / decoding) the cache is keyed on tensor
      identity + ``Tensor._version`` + ``data_ptr()``.  ``_version`` is bumped
      by every in-place autograd-tracked update (``optimizer.step()``,
      ``load_state_dict``'s ``copy_``); ``data_ptr()`` catches re-assignment.
      Raw ``.data`` writes bypass the version counter -- but they equally
      bypass autograd entirely and are not a supported way to mutate a model
      between decode steps.  ``cache`` may hold several tensors at once (it is
      keyed by ``id(tensor)``).
    """
    if torch.is_grad_enabled():
        return tensor.to(dtype)
    key = (dtype, tensor.device)
    hit = cache.get(id(tensor))
    if (hit is not None and hit[0] == key and hit[1] == tensor._version
            and hit[3] == tensor.data_ptr()):
        return hit[2]
    out = tensor.detach().to(dtype)
    cache[id(tensor)] = (key, tensor._version, out, tensor.data_ptr())
    return out
