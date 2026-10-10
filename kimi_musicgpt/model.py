"""KimiMusicGPT: music adapter wrapping the Kimi Linear backbone (§4/§5).

Phase-1 architecture (design doc §5.1, 方案A delay pattern):

* the backbone is *wrapped, not modified* -- tokens enter through the
  ``inputs_embeds`` entry point of ``KimiLinearModel``;
* ``MusicEmbedding`` sums per-slot codebook embeddings + slot embeddings +
  per-frame structure embeddings;
* conditions (text / melody / style) are encoded into a prefix embedding
  sequence prepended to the stream (§6.1); MLA layers attend to it fully,
  KDA layers absorb it into the recurrent state;
* a single untied ``lm_head`` over the music vocabulary
  (codebook_size + n_special, e.g. 2056) predicts every slot;
* training loss is shifted CE with per-slot weighting (semantic codebook 0
  up-weighted, §9.1.4) and PAD positions ignored;
* generation supports temperature / top-k / top-p, slot-constrained
  sampling (each position may only emit its own codebook's tokens + EOS on
  the last slot), structure-driven stopping and classifier-free guidance
  (dual cond/uncond batch).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kimi_linear import KimiLinearConfig, KimiLinearModel, KimiLinearCache, ConfigError

from .embedding import MusicEmbedding
from .conditions import (TextConditionEncoder, MelodyConditionEncoder,
                         ConditionPrefixEncoder)
from .structure import create_structure_tokens
from .delay import build_delay_stream, rearrange_delay_pattern


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class MusicGPTConfig:
    """Outer config: validates the 'music' section while keeping the strict
    KimiLinearConfig for the backbone (design doc §5.3 recommendation)."""

    REQUIRED = ("n_codebooks", "codebook_size", "n_special", "frame_rate",
                "io_pattern", "delays", "pad_token_id", "bos_token_id",
                "eos_token_id", "structure", "condition")

    def __init__(self, backbone: KimiLinearConfig):
        self.backbone = backbone
        music = backbone.get("music")
        if music is None:
            raise ConfigError("missing 'music' section in MusicGPT config")
        for k in self.REQUIRED:
            if k not in music:
                raise ConfigError(f"missing music config key '{k}'")
        if music["io_pattern"] != "delay":
            raise ConfigError("only io_pattern='delay' is implemented (Phase 1)")
        if len(music["delays"]) != music["n_codebooks"]:
            raise ConfigError("delays must have n_codebooks entries")
        vocab = music["codebook_size"] + music["n_special"]
        if vocab != backbone.vocab_size:
            raise ConfigError(
                f"backbone vocab_size ({backbone.vocab_size}) must equal "
                f"codebook_size + n_special ({vocab})")
        self.music = music

    @classmethod
    def from_json(cls, path: str) -> "MusicGPTConfig":
        return cls(KimiLinearConfig.from_json(path))

    @property
    def vocab_size(self):
        return self.music["codebook_size"] + self.music["n_special"]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class KimiMusicGPT(nn.Module):
    def __init__(self, config: MusicGPTConfig, text_vocab_size: int = 0):
        super().__init__()
        self.config = config
        m = config.music
        d = config.backbone.hidden_size
        self.n_codebooks = m["n_codebooks"]
        self.codebook_size = m["codebook_size"]
        self.pad_id = m["pad_token_id"]
        self.bos_id = m["bos_token_id"]
        self.eos_id = m["eos_token_id"]
        self.delays = m["delays"]
        self.semantic_slot_weight = m.get("semantic_slot_weight", 1.0)

        self.backbone = KimiLinearModel(config.backbone)
        self.embedding = MusicEmbedding(d, self.n_codebooks,
                                        self.codebook_size, m["n_special"],
                                        m["structure"])
        # share the backbone's (bypassed) embedding table with codebook 0 so
        # no dead parameters exist in the checkpoint
        self.backbone.embed_tokens.weight = self.embedding.codebook_embed[0].weight

        c = m["condition"]
        self.prefix_encoder = ConditionPrefixEncoder(
            d, c["n_styles"], c["max_prefix_len"], c.get("condition_dropout", 0.0))
        self.text_encoder = (TextConditionEncoder(
            text_vocab_size, c["text_dim"], c["text_layers"], c["text_heads"],
            c["text_max_len"]) if text_vocab_size > 0 else None)
        self.melody_encoder = MelodyConditionEncoder(self.codebook_size, d,
                                                     c["melody_dim"])

        self.lm_head = nn.Linear(d, config.vocab_size, bias=False)
        self.gradient_checkpointing = False      # opt-in for long-stream training
        self._graph_decoder = None               # see enable_graph_decode()
        self.apply(self._init)

    # ------------------------------------------------------------------ #
    def _init(self, module):
        r = self.config.backbone.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(0.0, r)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    # ------------------------------------------------------------------ #
    # conditions
    # ------------------------------------------------------------------ #
    def encode_conditions(self, text_ids=None, melody_tokens=None,
                          style_ids=None):
        text_emb = self.text_encoder(text_ids) \
            if (text_ids is not None and self.text_encoder is not None) else None
        melody_emb = self.melody_encoder(melody_tokens) \
            if melody_tokens is not None else None
        return self.prefix_encoder(text_emb=text_emb, melody_emb=melody_emb,
                                   style_ids=style_ids)

    # ------------------------------------------------------------------ #
    # training forward
    # ------------------------------------------------------------------ #
    def forward(self, stream, slots, structure=None, src_frames=None,
                cond_prefix=None, labels=None, cache=None, use_cache=False):
        x = self.embedding(stream, slots, structure, src_frames)
        n_prefix = 0
        if cond_prefix is not None:
            x = torch.cat([cond_prefix.to(x.dtype), x], dim=1)
            n_prefix = cond_prefix.shape[1]
        hidden, new_cache = self.backbone(
            inputs_embeds=x, cache=cache, use_cache=use_cache,
            gradient_checkpointing=self.gradient_checkpointing)
        hidden = hidden[:, n_prefix:]
        logits = self.lm_head(hidden).float()

        out = {"logits": logits, "cache": new_cache}
        if labels is not None:
            shift_logits = logits[:, :-1]
            shift_labels = labels[:, 1:]
            shift_slots = slots[:, 1:]
            ce = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                ignore_index=self.pad_id, reduction="none")
            ce = ce.view_as(shift_labels)
            w = torch.ones_like(ce)
            w = torch.where(shift_slots == 0,
                            torch.full_like(w, self.semantic_slot_weight), w)
            mask = shift_labels != self.pad_id
            out["loss"] = (ce * w * mask).sum() / (w * mask).sum().clamp(min=1.0)
        return out

    # ------------------------------------------------------------------ #
    # helpers for data construction
    # ------------------------------------------------------------------ #
    def build_stream(self, tokens):
        """tokens [B, T, Q] -> delay-stream dict (see delay.py)."""
        return build_delay_stream(tokens, self.delays, self.pad_id,
                                  self.bos_id, self.eos_id)

    # ------------------------------------------------------------------ #
    # CUDA-Graph incremental decoding (see kimi_linear.graph_decode)
    # ------------------------------------------------------------------ #
    def enable_graph_decode(self, max_new_tokens: int = 1024,
                            warmup_iters: int = 3):
        """Capture the backbone's single-token step as a CUDA graph.

        The decode loop below then replays one launch per emitted token
        instead of the thousands of eager launches the KDA reference kernels
        need.  Prefix/prompt prefill stays eager; per-step embedding lookup,
        lm_head sampling and slot masking stay eager too (a handful of
        kernels).  Supported in :meth:`generate` when ``prompt_tokens`` is
        None (prompt continuation falls back to the eager cache path).
        """
        from kimi_linear.graph_decode import GraphedDecoder
        if self._graph_decoder is not None:
            # a captured graph pins a private memory pool; release it before
            # capturing a new one (16 GiB cards have no headroom for two)
            del self._graph_decoder
            self._graph_decoder = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._graph_decoder = GraphedDecoder(
            self.backbone, max_new_tokens=max_new_tokens,
            warmup_iters=warmup_iters, lm_head=self.lm_head,
            input_mode="embeds")
        return self._graph_decoder

    def disable_graph_decode(self):
        self._graph_decoder = None

    # ------------------------------------------------------------------ #
    # generation
    # ------------------------------------------------------------------ #
    def _sample(self, logits, temperature, top_k, top_p):
        logits = logits / max(temperature, 1e-6)
        if top_k and top_k > 0:
            v, _ = torch.topk(logits, top_k, dim=-1)
            logits = torch.where(logits < v[..., -1:],
                                 torch.full_like(logits, float("-inf")), logits)
        probs = torch.softmax(logits, dim=-1)
        if top_p and 0 < top_p < 1.0:
            sp, si = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(sp, dim=-1)
            keep = cum - sp < top_p
            probs = torch.zeros_like(probs).scatter(-1, si, sp * keep)
            probs = probs / probs.sum(-1, keepdim=True).clamp(min=1e-9)
        return torch.multinomial(probs, 1)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, n_frames, bpm=120, chords=None, sections=None,
                 text_ids=None, melody_tokens=None, style_ids=None,
                 prompt_tokens=None, temperature=0.9, top_k=0, top_p=0.92,
                 cfg_scale=1.0, min_frames=0, device=None, dtype=None):
        """Autoregressive music generation with slot-constrained sampling.

        ``prompt_tokens``: optional [B, T_p, Q] codec frames to continue.
        ``cfg_scale`` > 1 enables classifier-free guidance (the unconditional
        branch uses a zeroed prefix).
        Returns (tokens [B, n_frames_total, Q], stream [B, L]).
        """
        self.eval()
        device = device or next(self.parameters()).device
        B = text_ids.shape[0] if text_ids is not None else (
            style_ids.shape[0] if style_ids is not None else
            prompt_tokens.shape[0] if prompt_tokens is not None else 1)
        Q = self.n_codebooks
        delays = torch.as_tensor(self.delays, device=device)

        # ---- condition prefix (+ unconditional twin for CFG) ------------- #
        cond_prefix = self.encode_conditions(text_ids, melody_tokens,
                                             style_ids)
        prefixes = [cond_prefix]
        if cfg_scale and cfg_scale != 1.0:
            prefixes.append(torch.zeros_like(cond_prefix))
        prefix = torch.cat(prefixes, dim=0)                      # [B*g, Lp, d]
        g = len(prefixes)

        # ---- structure labels per frame ---------------------------------- #
        struct = create_structure_tokens(n_frames, bpm=bpm, chords=chords,
                                         sections=sections, device=device)
        struct = {k: v.unsqueeze(0).expand(B, -1) for k, v in struct.items()}

        # ---- prefill: condition prefix, then BOS / prompt ---------------- #
        use_graph = (self._graph_decoder is not None and prompt_tokens is None)
        cache = hidden = None
        graph_logits = None
        if use_graph:
            dec = self._graph_decoder
            graph_logits = dec.prefill(inputs_embeds=prefix)      # [B*g, V]
        else:
            cache = KimiLinearCache(self.config.backbone, B * g, device,
                                    dtype or torch.float32)
            hidden, cache = self.backbone(inputs_embeds=prefix, cache=cache,
                                          use_cache=True)

        struct_g = {k: v.repeat(g, 1) for k, v in struct.items()}
        stream = [torch.full((B, 1), self.bos_id, dtype=torch.long,
                             device=device)]

        p_start = 0
        if prompt_tokens is not None:
            pr = build_delay_stream(prompt_tokens, self.delays, self.pad_id,
                                    self.bos_id, self.eos_id)
            fed = pr["stream"][:, :-1]                           # BOS + prompt body
            slots = pr["slots"][:, :-1]
            frames = pr["src_frames"][:, :-1]
            x = self.embedding(fed, slots, struct, frames).repeat(g, 1, 1)
            hidden, cache = self.backbone(inputs_embeds=x, cache=cache,
                                          use_cache=True)
            stream.append(fed[:, 1:])                            # prompt body
            p_start = fed.shape[1] - 1                           # next body pos
        else:
            bos_g = stream[0].repeat(g, 1)
            x = self.embedding(bos_g, torch.zeros_like(bos_g), struct_g,
                               torch.zeros_like(bos_g))
            if use_graph:
                graph_logits = dec.step(embeds=x).float()   # logits after BOS
            else:
                hidden, cache = self.backbone(inputs_embeds=x, cache=cache,
                                              use_cache=True)

        # ---- decode ------------------------------------------------------- #
        max_d = int(delays.max())
        body_len = (n_frames + max_d) * Q
        generated_frames = n_frames
        stopped = False
        for p in range(p_start, body_len):
            s, q = divmod(p, Q)
            src_frame = s - int(delays[q])
            if src_frame < 0:                                    # delay warm-up
                nxt = torch.full((B, 1), self.pad_id, dtype=torch.long,
                                 device=device)
            else:
                logits = graph_logits if use_graph else \
                    self.lm_head(hidden[:, -1]).float()     # [B*g, V]
                if g == 2:
                    lc, lu = logits[:B], logits[B:]
                    logits = lu + cfg_scale * (lc - lu)
                # slot constraint: only this codebook's tokens; EOS only on
                # the last slot of a frame once min_frames have been made
                mask = torch.full_like(logits, float("-inf"))
                mask[:, :self.codebook_size] = 0.0
                allow_eos = (q == Q - 1 and src_frame >= min_frames)
                if allow_eos:
                    mask[:, self.eos_id] = 0.0
                nxt = self._sample(logits + mask, temperature, top_k, top_p)
                if allow_eos:
                    eos_rows = (nxt == self.eos_id).squeeze(-1)
                    if bool(eos_rows.all()):
                        stream.append(nxt)
                        generated_frames = src_frame + 1
                        stopped = True
                        break
                    if bool(eos_rows.any()):
                        fallback = logits[:, :self.codebook_size].argmax(-1, keepdim=True)
                        nxt = torch.where(nxt == self.eos_id, fallback, nxt)
            stream.append(nxt)
            # feed the emitted token for the next step
            nxt_g = nxt.repeat(g, 1)
            slot_ids = torch.full((B * g, 1), q, dtype=torch.long, device=device)
            fr = torch.full((B * g, 1), max(min(src_frame, n_frames - 1), 0),
                            dtype=torch.long, device=device)
            x = self.embedding(nxt_g, slot_ids, struct_g, fr)
            if use_graph:
                graph_logits = dec.step(embeds=x).float()
            else:
                hidden, cache = self.backbone(inputs_embeds=x, cache=cache,
                                              use_cache=True)

        if not stopped:
            stream.append(torch.full((B, 1), self.eos_id, dtype=torch.long,
                                     device=device))
        stream = torch.cat(stream, dim=1)
        frames = rearrange_delay_pattern(stream, self.delays, generated_frames,
                                         Q, self.pad_id)
        return frames, stream

    # ------------------------------------------------------------------ #
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
