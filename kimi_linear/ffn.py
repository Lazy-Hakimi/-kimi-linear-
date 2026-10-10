"""Feed-forward network with a configurable Dense / MoE branch.

The paper's Kimi Linear stack follows Moonlight and uses a MoE channel-mixing
layer (sigmoid-scored routed experts plus always-on shared experts, first
layers dense). Per the project requirements the FFN is implemented as a
branch selected purely by the configuration document:

    "ffn": {
        "mode": "dense" | "moe",
        "intermediate_size": ...,          # used by the dense branch
        "moe": { ... }                     # used by the moe branch
    }

No branch decision is hard-coded anywhere else.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fp16 import FP16_MAX_SAFE
from .kda import RMSNorm


_ACTS = {
    "silu": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
}


class SwiGLUExpert(nn.Module):
    """Gated MLP (SwiGLU): down( silu(gate(x)) * up(x) )."""

    def __init__(self, hidden_size: int, intermediate_size: int, act: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = _ACTS[act]

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class DenseFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        ffn = config.ffn_config
        self.expert = SwiGLUExpert(config.hidden_size, ffn["intermediate_size"], ffn["hidden_act"])

    def forward(self, x):
        return self.expert(x)


class MoEFFN(nn.Module):
    """Moonlight/DeepSeek-style mixture of experts.

    Routing: sigmoid scores, top-k selection, optional renormalization of the
    top-k weights and a global scaling factor. ``n_shared_experts`` are always
    active. An optional Switch-style load-balancing auxiliary loss is exposed
    via the ``aux_loss`` attribute when ``aux_loss_coeff > 0``.
    """

    def __init__(self, config):
        super().__init__()
        ffn = config.ffn_config
        moe = config.moe_config
        d = config.hidden_size
        self.n_routed = moe["n_routed_experts"]
        self.top_k = moe["num_experts_per_tok"]
        self.n_shared = moe["n_shared_experts"]
        self.norm_topk_prob = moe["norm_topk_prob"]
        self.routed_scaling_factor = moe["routed_scaling_factor"]
        self.scoring_func = moe["scoring_func"]
        self.aux_loss_coeff = moe["aux_loss_coeff"]
        self.act = ffn["hidden_act"]

        self.gate = nn.Linear(d, self.n_routed, bias=False)
        self.experts = nn.ModuleList(
            SwiGLUExpert(d, moe["moe_intermediate_size"], self.act) for _ in range(self.n_routed)
        )
        self.shared_experts = nn.ModuleList(
            SwiGLUExpert(d, moe["moe_intermediate_size"], self.act) for _ in range(self.n_shared)
        )
        self.aux_loss = None

    def forward(self, x):
        B, T, d = x.shape
        flat = x.reshape(B * T, d)
        out_dtype = x.dtype

        # ---- routing: fp32 (a *discrete* decision, not a smooth function) ---
        # sigmoid/softmax over 64 logits is a top-k selection; an fp16 rounding
        # of the scores silently changes which experts a token is sent to, and
        # the route is non-differentiable anyway, so there is nothing to gain
        # from fp16 here and a lot to lose.
        logits = self.gate(flat)
        if self.scoring_func == "sigmoid":
            scores = torch.sigmoid(logits.float())
        elif self.scoring_func == "softmax":
            scores = torch.softmax(logits.float(), dim=-1)
        else:
            raise ValueError(f"unsupported scoring_func: {self.scoring_func}")

        weights, topk_idx = torch.topk(scores, self.top_k, dim=-1)     # [N, k]
        if self.norm_topk_prob and self.top_k > 1:
            weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-6)
        weights = weights * self.routed_scaling_factor

        # ---- expert compute: pure fp16 (this is the FLOP bulk) -------------
        # Every expert GEMM runs in the storage dtype so it hits the V100's fp16
        # tensor cores.  Only the *accumulation* buffer stays fp32: index_add_
        # accumulates top_k (=6) expert outputs per token, and doing that in
        # fp16 injects ~sqrt(k)*5e-4 relative error straight onto the residual
        # path.  [N, d] fp32 costs 4 bytes/token/dim -- 64 MiB at N=8192, d=2048.
        y = torch.zeros(flat.shape[0], d, dtype=torch.float32, device=x.device)
        flat_weights = weights.reshape(-1)                             # [N*k]
        flat_idx = topk_idx.reshape(-1)                                # [N*k]
        token_ids = torch.arange(flat.shape[0], device=x.device).repeat_interleave(self.top_k)

        # route tokens to their experts (reference loop over experts)
        sort_order = torch.argsort(flat_idx)
        token_ids = token_ids[sort_order]
        flat_weights = flat_weights[sort_order]
        flat_idx = flat_idx[sort_order]
        boundaries = torch.searchsorted(
            flat_idx, torch.arange(self.n_routed + 1, device=x.device)
        )
        for e in range(self.n_routed):
            lo, hi = int(boundaries[e]), int(boundaries[e + 1])
            if hi > lo:
                sel = token_ids[lo:hi]
                ye = self.experts[e](flat[sel])                       # fp16 in/out
                y.index_add_(0, sel, ye.float() * flat_weights[lo:hi, None])

        for shared in self.shared_experts:
            y = y + shared(flat).float()

        if self.aux_loss_coeff > 0 and self.training:
            # Switch-style balance loss over the (token, slot) assignments
            with torch.no_grad():
                frac = torch.zeros(self.n_routed, device=x.device)
                frac.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=frac.dtype))
                frac = frac / flat_idx.numel()
            mean_prob = scores.mean(dim=0)
            self.aux_loss = self.aux_loss_coeff * self.n_routed * (frac * mean_prob).sum()

        if out_dtype is torch.float16:
            y = y.clamp(-FP16_MAX_SAFE, FP16_MAX_SAFE)
        return y.to(out_dtype).view(B, T, d)


def build_ffn(config, layer_idx: int):
    """FFN factory implementing the dense/moe branch switch.

    In MoE mode the first ``first_k_dense_replace`` layers stay dense
    (paper: "The first layer is implemented as a dense layer without MoE").
    """
    ffn = config.ffn_config
    if ffn["mode"] == "dense":
        return DenseFFN(config)
    if layer_idx < ffn["moe"]["first_k_dense_replace"]:
        return DenseFFN(config)
    return MoEFFN(config)
