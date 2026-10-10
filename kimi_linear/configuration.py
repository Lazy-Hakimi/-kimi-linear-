"""Kimi Linear configuration.

Every model hyper-parameter lives in a single JSON document (see
``configs/kimi_linear_1p5b_dense.json``); nothing is hard-coded in the model
code.  :class:`KimiLinearConfig` loads that document, validates it and exposes
read-only attribute access plus an analytic parameter counter.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field


class ConfigError(KeyError):
    pass


@dataclass
class KimiLinearConfig:
    """Strict container for one Kimi Linear model configuration document."""

    values: dict

    # ------------------------------------------------------------------ io
    @classmethod
    def from_json(cls, path: str) -> "KimiLinearConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = cls(data)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict:
        return self.values

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.values, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------- access
    def __getattr__(self, name: str):
        # only called when normal attribute lookup fails -> strict key access
        try:
            return self.__dict__["values"][name]
        except KeyError:
            raise ConfigError(
                f"missing config key '{name}': all model parameters must be "
                f"declared in the configuration document"
            ) from None

    def get(self, name: str, default=None):
        return self.values.get(name, default)

    # --------------------------------------------------------- shortcuts
    @property
    def linear_attn_config(self) -> dict:
        return self.values["linear_attn"]

    @property
    def full_attn_config(self) -> dict:
        return self.values["full_attn"]

    @property
    def ffn_config(self) -> dict:
        return self.values["ffn"]

    @property
    def moe_config(self) -> dict:
        return self.values["ffn"]["moe"]

    @property
    def layer_types(self) -> list:
        """Fully expanded per-layer type list, e.g. ['kda','kda','kda','mla', ...]."""
        pattern = self.values["layer_pattern"]
        repeat = self.values["layer_pattern_repeat"]
        expanded = pattern * repeat
        if len(expanded) != self.values["num_hidden_layers"]:
            raise ConfigError(
                f"layer_pattern ({pattern}) x layer_pattern_repeat ({repeat}) "
                f"gives {len(expanded)} layers but num_hidden_layers is "
                f"{self.values['num_hidden_layers']}"
            )
        return expanded

    # -------------------------------------------------------- validation
    def validate(self) -> None:
        v = self.values
        for key in (
            "model_type", "vocab_size", "hidden_size", "num_hidden_layers",
            "layer_pattern", "layer_pattern_repeat", "rms_norm_eps",
            "l2norm_eps", "initializer_range", "torch_dtype",
        ):
            if key not in v:
                raise ConfigError(f"missing top-level config key '{key}'")
        for section in ("linear_attn", "full_attn", "ffn"):
            if section not in v:
                raise ConfigError(f"missing config section '{section}'")

        la, fa, ffn = v["linear_attn"], v["full_attn"], v["ffn"]
        if la["num_heads"] * la["head_dim"] <= 0:
            raise ConfigError("linear_attn heads/head_dim must be positive")
        if la["value_head_dim"] <= 0:
            raise ConfigError("linear_attn value_head_dim must be positive")
        if la["chunk_size"] <= 0:
            raise ConfigError("linear_attn chunk_size must be positive")
        if fa["num_heads"] <= 0 or fa["kv_lora_rank"] <= 0:
            raise ConfigError("full_attn num_heads/kv_lora_rank must be positive")
        if fa["qk_rope_head_dim"] < 0:
            raise ConfigError("full_attn qk_rope_head_dim must be >= 0 (0 = NoPE)")
        if ffn["mode"] not in ("dense", "moe"):
            raise ConfigError("ffn.mode must be 'dense' or 'moe'")
        if ffn["mode"] == "moe":
            moe = ffn["moe"]
            if moe["num_experts_per_tok"] > moe["n_routed_experts"]:
                raise ConfigError("num_experts_per_tok cannot exceed n_routed_experts")
        # layer pattern consistency (raises on mismatch)
        self.layer_types

    # --------------------------------------------------- parameter count
    def num_parameters(self, detailed: bool = False):
        """Analytic parameter count of the model described by this config.

        Must stay exactly in sync with the modules built in ``modeling.py``
        (cross-checked by scripts/count_params.py on the meta device).
        """
        v = self.values
        d = v["hidden_size"]
        vocab = v["vocab_size"]

        la = v["linear_attn"]
        H, K, V = la["num_heads"], la["head_dim"], la["value_head_dim"]
        r_gk = la["gate_low_rank"]
        r_og = la["output_gate_low_rank"]
        cs = la["conv_size"]
        conv_bias = 1 if la["conv_bias"] else 0
        proj_bias = 1 if la["proj_bias"] else 0

        # KDA layer: q/k/v input projections + short convs + beta/gate/output
        # gate projections + A_log/dt_bias + headwise norm + output projection
        kda = 0
        kda += 3 * d * (H * K) + 3 * proj_bias * (H * K)      # in_proj q,k,v
        kda += 3 * ((H * K) * cs + conv_bias * (H * K))       # depthwise conv w+b
        kda += d * H + proj_bias * H                          # beta projection
        kda += d * r_gk + r_gk * (H * K) + 2 * proj_bias * r_gk   # forget gate low-rank
        kda += 2 * (H * K)                                    # A_log + dt_bias
        kda += d * r_og + r_og * (H * V) + 2 * proj_bias * r_og   # output gate low-rank
        kda += H * V                                          # headwise RMSNorm
        kda += (H * V) * d + proj_bias * d                    # output projection

        fa = v["full_attn"]
        Hf = fa["num_heads"]
        qk_rope = fa["qk_rope_head_dim"]
        q_head_dim = fa["qk_nope_head_dim"] + qk_rope
        ql, kvl = fa["q_lora_rank"], fa["kv_lora_rank"]
        vh = fa["v_head_dim"]
        qkv_bias = 1 if fa["qkv_bias"] else 0
        o_bias = 1 if fa["o_bias"] else 0

        # MLA layer (DeepSeek style; NoPE = qk_rope_head_dim == 0)
        mla = 0
        mla += d * ql + qkv_bias * ql                        # q_a_proj
        mla += ql                                            # q_a_layernorm
        mla += ql * (Hf * q_head_dim)                        # q_b_proj
        mla += d * (kvl + qk_rope) + qkv_bias * (kvl + qk_rope)  # kv_a_proj
        mla += kvl                                           # kv_a_layernorm
        mla += kvl * (Hf * (fa["qk_nope_head_dim"] + vh))    # kv_b_proj
        mla += (Hf * vh) * d + o_bias * d                    # o_proj

        act_extra = 0  # silu/swish are parameter free
        ffn = v["ffn"]
        inter = ffn["intermediate_size"]
        dense_ffn = 3 * d * inter + act_extra                # SwiGLU gate/up/down

        moe = ffn["moe"]
        mi = moe["moe_intermediate_size"]
        E = moe["n_routed_experts"]
        n_shared = moe["n_shared_experts"]
        moe_ffn = d * E + (E + n_shared) * 3 * d * mi        # router + experts

        total = vocab * d                                    # embedding
        per_layer = {}
        for i, ltype in enumerate(self.layer_types):
            if ffn["mode"] == "moe" and i < moe["first_k_dense_replace"]:
                ffn_p = dense_ffn
            elif ffn["mode"] == "moe":
                ffn_p = moe_ffn
            else:
                ffn_p = dense_ffn
            layer_p = (kda if ltype == "kda" else mla) + ffn_p + 2 * d  # + 2 RMSNorm
            per_layer[ltype] = per_layer.get(ltype, 0) + layer_p
            total += layer_p
        total += d                                           # final norm
        if not v["tie_word_embeddings"]:
            total += vocab * d                               # lm_head

        if detailed:
            return {
                "total": total,
                "embedding": vocab * d + (0 if v["tie_word_embeddings"] else vocab * d),
                "per_kda_layer": kda,
                "per_mla_layer": mla,
                "per_dense_ffn_layer": dense_ffn,
                "per_moe_ffn_layer": moe_ffn,
                "layers": per_layer,
                "final_norm": d,
            }
        return total


def default_config_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "configs", "kimi_linear_1p5b_dense.json")
