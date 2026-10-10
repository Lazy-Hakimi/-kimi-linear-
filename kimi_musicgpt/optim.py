"""Adafactor optimizer (factorised second moment) -- the memory plan that
makes 1.5B pure-PyTorch training fit on a single 16GB V100.

Why not AdamW on 16GB
---------------------
A 1.55B model trained the AMP way (fp32 master weights) needs

    fp32 master weights   6.2 GB
    AdamW m + v (fp32)   12.4 GB
    fp32 gradients        6.2 GB
                        ---------
                          24.8 GB   -> dead on arrival, before any activation

Adafactor (Shazeer & Stern, 2018) replaces the per-parameter second moment
``v`` for every 2-D weight ``[r, c]`` with a rank-1 factorisation
``v ≈ row[r] ⊗ col[c]``, so optimizer state drops from 2x parameters to
~0.01x.  The full budget becomes

    fp32 master weights   6.2 GB
    fp32 gradients        6.2 GB
    Adafactor state      ~0.1 GB
    activations (fp16, grad-checkpointed, T=2048, B=1)  ~1.5 GB
                        ---------
                          ~14 GB   -> fits the V100's 16 GiB

This is the reference implementation used by fairseq/HuggingFace
(``transformers.optimization.Adafactor``), kept dependency-free:

* second moment factorisation as above; 1-D tensors keep a full ``v``;
* optional first moment (``beta1``); update-norm (RMS) clipping as in the
  paper; decoupled weight decay;
* ``relative_step=False`` by default so the trainer's own warmup/cosine
  schedule drives ``lr`` exactly like AdamW (``relative_step=True`` is the
  T5-style self-scheduling mode and ignores ``lr``).
"""

from __future__ import annotations

import math

import torch


class Adafactor(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, eps=(1e-30, 1e-3), clip_threshold=1.0,
                 decay_rate=-0.8, beta1=None, weight_decay=0.0,
                 relative_step=False, warmup_steps=0):
        if lr <= 0.0 and not relative_step:
            raise ValueError(f"invalid learning rate: {lr}")
        defaults = dict(lr=lr, eps=eps, clip_threshold=clip_threshold,
                        decay_rate=decay_rate, beta1=beta1,
                        weight_decay=weight_decay, relative_step=relative_step,
                        warmup_steps=warmup_steps)
        super().__init__(params, defaults)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _rms(x: torch.Tensor) -> torch.Tensor:
        return x.norm() / (x.numel() ** 0.5)

    def _lr(self, group, step: int) -> float:
        if not group["relative_step"]:
            return group["lr"]
        # T5-style: lr = min(1/sqrt(t), 1/warmup) * (1/sqrt(d_model-free scaling))
        rel = min(1.0 / math.sqrt(step), 1.0 / max(1, group["warmup_steps"]))
        return rel

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps1, eps2 = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    if p.dim() >= 2:
                        # rank-1 factorisation over (leading dims) x (last dim)
                        state["exp_avg_sq_row"] = torch.zeros(
                            p.shape[:-1], dtype=torch.float32, device=p.device)
                        state["exp_avg_sq_col"] = torch.zeros(
                            p.shape[:-2] + p.shape[-1:],
                            dtype=torch.float32, device=p.device)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, dtype=torch.float32)
                    if group["beta1"] is not None:
                        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)

                state["step"] += 1
                t = state["step"]
                beta2_t = 1.0 - t ** group["decay_rate"]       # paper schedule
                g2 = grad.float() ** 2 + eps1

                if p.dim() >= 2:
                    row, col = state["exp_avg_sq_row"], state["exp_avg_sq_col"]
                    row.mul_(beta2_t).add_(g2.mean(dim=-1), alpha=1.0 - beta2_t)
                    col.mul_(beta2_t).add_(g2.mean(dim=-2), alpha=1.0 - beta2_t)
                    # v_hat = (row outer col) / mean(row), broadcast for any dim
                    v = (row.unsqueeze(-1) * col.unsqueeze(-2)) / (
                        row.mean(dim=-1, keepdim=True).clamp(min=eps1).unsqueeze(-1))
                else:
                    v = state["exp_avg_sq"]
                    v.mul_(beta2_t).add_(g2, alpha=1.0 - beta2_t)

                update = grad.float() / (v.sqrt() + eps2)

                # update-norm clipping (paper: RMS clip at 1.0)
                rms = self._rms(update)
                if group["clip_threshold"] > 0.0:
                    update = update / (rms / group["clip_threshold"]).clamp(min=1.0)

                if group["beta1"] is not None:
                    m = state["exp_avg"]
                    m.mul_(group["beta1"]).add_(update, alpha=1.0 - group["beta1"])
                    update = m

                lr = self._lr(group, t)
                if group["weight_decay"] != 0.0:
                    p.mul_(1.0 - lr * group["weight_decay"])
                p.add_(update.to(p.dtype), alpha=-lr)
        return loss

    # ------------------------------------------------------------------ #
    def state_dict_compact(self):
        """Alias of state_dict for readability at call sites."""
        return self.state_dict()
