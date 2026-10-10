"""Kimi Linear model: blocks, backbone, causal LM head and inference cache.

Layer stack follows the paper's hybrid design (Section 4): token-mixing
layers alternate KDA (linear attention) and full-attention MLA in a uniform
ratio (3:1 by default), each followed by an FFN channel-mixing layer that is
dense or MoE depending on the single configuration document.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .configuration import KimiLinearConfig
from .kda import RMSNorm, KimiDeltaAttention
from .mla import MLA
from .ffn import build_ffn, MoEFFN


# --------------------------------------------------------------------------- #
# Transformer block
# --------------------------------------------------------------------------- #
class KimiLinearBlock(nn.Module):
    """Pre-norm block: x + Mixer(norm(x)); x + FFN(norm(x))."""

    def __init__(self, config: KimiLinearConfig, layer_idx: int, layer_type: str):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(config.hidden_size, eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps)
        if layer_type == "kda":
            self.mixer = KimiDeltaAttention(config)
        elif layer_type == "mla":
            self.mixer = MLA(config)
        else:
            raise ValueError(f"unknown layer_type '{layer_type}' in layer_pattern")
        self.ffn = build_ffn(config, layer_idx)

    def forward(self, x, state=None, use_cache=False):
        if use_cache or state is not None:
            h, new_state = self.mixer(self.input_layernorm(x), state=state, use_cache=True)
            x = x + h
        else:
            x = x + self.mixer(self.input_layernorm(x))
        x = x + self.ffn(self.post_attention_layernorm(x))
        return x, (new_state if (use_cache or state is not None) else None)


# --------------------------------------------------------------------------- #
# Inference cache
# --------------------------------------------------------------------------- #
class KimiLinearCache:
    """Per-layer recurrent/latent states for incremental decoding.

    KDA layers keep the short-conv input history plus the matrix state S;
    MLA layers keep the compressed latent c_KV (and rope keys if any).
    """

    def __init__(self, config: KimiLinearConfig, batch_size: int, device, dtype=torch.float32):
        la = config.linear_attn_config
        fa = config.full_attn_config
        H, K, V = la["num_heads"], la["head_dim"], la["value_head_dim"]
        conv_in = la["num_heads"] * la["head_dim"]
        conv_in_v = la["num_heads"] * la["value_head_dim"]
        self.layers = []
        for ltype in config.layer_types:
            if ltype == "kda":
                self.layers.append({
                    "conv": {
                        "q": torch.zeros(batch_size, max(la["conv_size"] - 1, 0), conv_in, device=device, dtype=dtype),
                        "k": torch.zeros(batch_size, max(la["conv_size"] - 1, 0), conv_in, device=device, dtype=dtype),
                        "v": torch.zeros(batch_size, max(la["conv_size"] - 1, 0), conv_in_v, device=device, dtype=dtype),
                    },
                    "recurrent": None,
                })
            else:
                self.layers.append({
                    "c_kv": torch.zeros(batch_size, 0, fa["kv_lora_rank"], device=device, dtype=dtype),
                    "k_rope": torch.zeros(batch_size, 0, fa["qk_rope_head_dim"], device=device, dtype=dtype),
                    "offset": 0,
                })

    def __len__(self):
        for st in self.layers:
            if "offset" in st:
                return st["offset"]
        return 0


# --------------------------------------------------------------------------- #
# Backbone
# --------------------------------------------------------------------------- #
class KimiLinearModel(nn.Module):
    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            KimiLinearBlock(config, i, ltype)
            for i, ltype in enumerate(config.layer_types)
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids=None, inputs_embeds=None, cache: KimiLinearCache = None,
                use_cache: bool = False, gradient_checkpointing: bool = False):
        # ``inputs_embeds`` is the sanctioned entry point for wrapper models
        # (e.g. the music adapter) that sum multiple embedding sources before
        # the backbone; exactly one of input_ids / inputs_embeds is required.
        # ``gradient_checkpointing`` (training only) recomputes each block in
        # backward instead of storing its activations -- opt-in, off by
        # default, so existing callers see bit-identical behaviour.
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("KimiLinearModel.forward needs input_ids or inputs_embeds")
            x = self.embed_tokens(input_ids)
        else:
            x = inputs_embeds
        new_cache = None
        if use_cache:
            if cache is None:
                cache = KimiLinearCache(self.config, x.shape[0], x.device, x.dtype)
            new_cache = cache
        for i, layer in enumerate(self.layers):
            # cache participates only when use_cache is requested; otherwise a
            # plain full forward is computed and any cache stays untouched
            layer_state = cache.layers[i] if (use_cache and cache is not None) else None
            if gradient_checkpointing and torch.is_grad_enabled() and not use_cache:
                x, updated = torch.utils.checkpoint.checkpoint(
                    layer, x, layer_state, use_cache, use_reentrant=False)
            else:
                x, updated = layer(x, state=layer_state, use_cache=use_cache)
            if use_cache:
                cache.layers[i] = updated
        return self.final_norm(x), new_cache


# --------------------------------------------------------------------------- #
# Causal LM
# --------------------------------------------------------------------------- #
class KimiLinearForCausalLM(nn.Module):
    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.model = KimiLinearModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.apply(self._init_weights)

    # ------------------------------------------------------------------ #
    def _init_weights(self, module):
        init_range = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=init_range)
            if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)
        # A_log / dt_bias: re-applied from the stored config values.  This
        # line looks redundant for a normal build but is *load-bearing* after
        # a meta-device build + materialize_fp16_ (to_empty leaves the storage
        # uninitialised, and the constructor's zeros+init never ran).
        if isinstance(module, KimiDeltaAttention):
            module.A_log.data.fill_(module._a_log_init)
            module.dt_bias.data.fill_(module._dt_bias_init)

    # ------------------------------------------------------------------ #
    def forward(self, input_ids, labels=None, cache: KimiLinearCache = None, use_cache: bool = False):
        hidden, cache = self.model(input_ids, cache=cache, use_cache=use_cache)
        logits = self.lm_head(hidden).float()

        out = {"logits": logits, "cache": cache}
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            out["loss"] = loss

            # collect optional MoE auxiliary (load-balancing) losses
            aux = 0.0
            for m in self.modules():
                if isinstance(m, MoEFFN) and getattr(m, "aux_loss", None) is not None:
                    aux = aux + m.aux_loss
            if isinstance(aux, torch.Tensor):
                out["aux_loss"] = aux
        return out

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens: int, do_sample: bool = False,
                 temperature: float = 1.0, top_k: int = 0):
        """Minimal greedy/sampling generation using the recurrent + latent cache."""
        self.eval()
        out = self.forward(input_ids, use_cache=True)
        logits = out["logits"][:, -1, :]
        generated = [input_ids]
        cache = out["cache"]
        cur = input_ids
        for _ in range(max_new_tokens):
            if do_sample:
                probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
                if top_k > 0:
                    topv, _ = torch.topk(probs, top_k, dim=-1)
                    probs = torch.where(probs >= topv[..., -1:], probs, torch.zeros_like(probs))
                    probs = probs / probs.sum(-1, keepdim=True)
                nxt = torch.multinomial(probs, 1)
            else:
                nxt = logits.argmax(-1, keepdim=True)
            generated.append(nxt)
            cur = nxt
            out = self.forward(cur, cache=cache, use_cache=True)
            logits = out["logits"][:, -1, :]
        return torch.cat(generated, dim=1)

    # ------------------------------------------------------------------ #
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
