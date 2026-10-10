"""CUDA-Graph-captured incremental decoding for Kimi Linear (pure PyTorch).

Why
---
Profiling one generated token of the 1.5B model on the V100 shows thousands of
tiny kernel launches (and ~322 ``aten::_to_copy`` calls): the decode step is
**launch-bound**, not compute-bound -- 101.8 ms/token while the arithmetic in
it is worth ~1 ms.  ``torch.cuda.CUDAGraph`` is part of core PyTorch and replays
the entire captured step as a single launch, which is the only remaining
order-of-magnitude win available under the "pure PyTorch" constraint.

What makes capture hard here
----------------------------
1. **KDA layers** carry a *fixed-size* state (fp32 master recurrent state
   ``S`` plus the short-conv windows), but the eager modules return *new*
   tensors every step.  Solution: the captured region ends with **copy-back
   kernels** that write the newly produced state back into the static input
   tensors, so one replay == one self-contained decode step (state chaining
   happens on the GPU, inside the graph).

2. **MLA layers** cache grows by one ``torch.cat`` per token -- incompatible
   with the static shapes a graph needs.  Solution:
   :meth:`MLA._decode_step_static <kimi_linear.mla.MLA._decode_step_static>`
   writes into a pre-allocated ``[B, max_len, kvl]`` buffer at a *device-side*
   position tensor and attends over the whole buffer with a validity mask, so
   all shapes are static while the semantics are unchanged.

3. **Warmup and capture each *execute* the step once** (stream capture records
   and runs the ops), which would silently corrupt the state.  Solution: the
   post-prefill state is snapshotted and restored around warmup and capture.

Scope / limitations
-------------------
* ``batch_size`` and the MLA buffer capacity (prompt + max_new_tokens) are
  fixed at capture; a longer prompt or bigger batch transparently re-captures.
* ``MoEFFN`` decoding performs host-side routing decisions (``int()`` on a
  device tensor per expert) and is **not** graph-safe; the constructor rejects
  MoE models.  The dense FFN of the 1.5B config has no such dependency.
* The sampler (argmax / top-k) stays eager -- it is one kernel and keeps
  sampling policy out of the captured graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ffn import MoEFFN
from .mla import MLA


class _WorkCache:
    """Duck-typed stand-in for :class:`KimiLinearCache`.

    ``KimiLinearModel.forward`` only needs ``cache.layers`` (read and
    reassigned per step).  The KDA entries start out as the *static* tensors
    so the captured graph reads them; the model replaces the list entries
    with fresh dicts during capture, and the copy-back kernels inside the
    captured region write those back into the static tensors.
    """

    def __init__(self, layers):
        self.layers = layers


class GraphedDecoder:
    """CUDA-Graph incremental decoder for a :class:`KimiLinearForCausalLM`.

    Usage::

        dec = GraphedDecoder(model, max_new_tokens=256)
        logits = dec.prefill(prompt_ids)          # [B, V] fp32, eager + capture
        for _ in range(n):
            nxt = logits.argmax(-1, keepdim=True)
            logits = dec.step(nxt)                 # one graph replay

    ``step`` returns a view of the static output buffer -- read it (or clone
    it) before the next call.  Weights are never touched by decoding, so one
    capture serves any number of prompts with the same batch size (a longer
    prompt re-captures automatically).
    """

    def __init__(self, model: nn.Module, max_new_tokens: int = 1024,
                 warmup_iters: int = 3, lm_head: nn.Module = None,
                 input_mode: str = "ids"):
        for m in model.modules():
            if isinstance(m, MoEFFN):
                raise NotImplementedError(
                    "GraphedDecoder does not support MoE FFNs: their routing "
                    "reads device tensors on the host per expert, which cannot "
                    "be captured. Use a dense-FFN config."
                )
        # ``model`` may be a KimiLinearForCausalLM (its ``.model`` is the
        # backbone) or a bare KimiLinearModel -- the music adapter feeds
        # embeddings instead of token ids (``input_mode="embeds"``) and brings
        # its own lm_head.
        self.backbone = getattr(model, "model", model)
        self.lm_head = lm_head if lm_head is not None else getattr(model, "lm_head", None)
        if input_mode not in ("ids", "embeds"):
            raise ValueError(f"input_mode must be 'ids' or 'embeds', got {input_mode!r}")
        self.input_mode = input_mode
        # cudnn.benchmark autotuning is fatal to a captured CUDA graph: any
        # eager convolution with a *new* input shape between capture and
        # replay (e.g. a second prefill of a different prompt length) makes
        # cudnn re-run its algorithm search, churning workspace memory and
        # leaving the graph's captured kernels writing to freed storage --
        # observed on the V100 as "illegal memory access" / segfault on the
        # next replay (isolated empirically: the full eager prefill crashes,
        # and the trigger reduces exactly to F.conv1d with a new shape).
        # The decoder therefore forces benchmark off for its lifetime; the
        # decode path itself never touches cudnn (the T=1 ShortConv path is
        # manual mul+sum) so nothing is lost.
        torch.backends.cudnn.benchmark = False
        self.model = model
        self.model.eval()
        self.max_new_tokens = int(max_new_tokens)
        self.warmup_iters = int(warmup_iters)
        cfg = self.backbone.config
        fa = cfg.full_attn_config
        self.kvl = fa["kv_lora_rank"]
        self.dr = fa["qk_rope_head_dim"]
        self.graph: torch.cuda.CUDAGraph | None = None
        self._B = None                # captured batch size
        self._L_max = 0               # MLA buffer capacity
        self._avail = 0               # remaining graph steps before overflow
        self._dev = None

    # ------------------------------------------------------------------ #
    # state plumbing
    # ------------------------------------------------------------------ #
    def _alloc_static(self, B: int, S: int, dev, dtype: torch.dtype):
        """Allocate the static MLA buffers, input/output and id buffers.

        KDA states are shape-invariant (allocated once, refilled by
        :meth:`_fill_static`); only the MLA buffers depend on the capacity
        ``L_max = S + max_new_tokens``.
        """
        self._B, self._L_max = B, S + self.max_new_tokens
        self._dev = dev
        if self.input_mode == "embeds":
            self.input_static = torch.zeros(
                B, 1, self.backbone.config.hidden_size, device=dev, dtype=dtype)
        else:
            self.ids_static = torch.zeros(B, 1, device=dev, dtype=torch.long)
        out_dim = self.lm_head.out_features if self.lm_head is not None \
            else self.backbone.config.hidden_size
        self.logits_static = torch.empty(B, out_dim, device=dev, dtype=dtype)

    def _build_static_from(self, cache, S: int):
        """Copy an eager post-prefill cache into the static state tensors."""
        B = cache.layers[0]["conv"]["q"].shape[0] if "conv" in cache.layers[0] \
            else cache.layers[0]["c_kv"].shape[0]
        dev = self._dev
        self.static_layers = []
        for st in cache.layers:
            if "conv" in st:                                   # KDA layer
                self.static_layers.append({
                    "conv": {nm: t.clone() for nm, t in st["conv"].items()},
                    "recurrent": st["recurrent"].clone(),
                })
            else:                                               # MLA layer
                entry = {
                    "c_kv_buf": torch.zeros(
                        B, self._L_max, self.kvl, device=dev,
                        dtype=st["c_kv"].dtype),
                    "pos": torch.tensor([S], device=dev, dtype=torch.long),
                }
                entry["c_kv_buf"][:, :S].copy_(st["c_kv"])
                if self.dr > 0:
                    entry["k_rope_buf"] = torch.zeros(
                        B, self._L_max, self.dr, device=dev,
                        dtype=st["k_rope"].dtype)
                    entry["k_rope_buf"][:, :S].copy_(st["k_rope"])
                self.static_layers.append(entry)
        # fresh work cache whose KDA entries are the static tensors themselves
        self._rebuild_work_cache()

    def _fill_static(self, cache, S: int):
        """Refill the existing static tensors from a post-prefill cache."""
        for st, src in zip(self.static_layers, cache.layers):
            if "conv" in st:                                   # KDA layer
                for nm in st["conv"]:
                    st["conv"][nm].copy_(src["conv"][nm])
                st["recurrent"].copy_(src["recurrent"])
            else:                                               # MLA layer
                st["c_kv_buf"].zero_()
                st["c_kv_buf"][:, :S].copy_(src["c_kv"])
                if self.dr > 0:
                    st["k_rope_buf"].zero_()
                    st["k_rope_buf"][:, :S].copy_(src["k_rope"])
                st["pos"].fill_(S)
        self._rebuild_work_cache()

    def _snapshot(self):
        out = []
        for st in self.static_layers:
            if "conv" in st:
                out.append({"conv": {nm: t.clone() for nm, t in st["conv"].items()},
                            "recurrent": st["recurrent"].clone()})
            else:
                snap = {"c_kv_buf": st["c_kv_buf"].clone(),
                        "pos": st["pos"].clone()}
                if self.dr > 0:
                    snap["k_rope_buf"] = st["k_rope_buf"].clone()
                out.append(snap)
        return out

    def _rebuild_work_cache(self):
        """Point the model's state reads at the STATIC tensors.

        ``KimiLinearModel.forward`` replaces ``cache.layers[i]`` with the
        freshly returned state dicts on every step, so after a warmup run the
        work cache points at stale (already freed) tensors instead of the
        static ones.  A graph captured in that state would *read freed memory*
        on every replay -- it can appear to work only if the allocator happens
        to reuse those blocks with the right values.  Rebuilding the work
        cache from the static dicts makes the captured kernels read and
        (through the copy-back) write the static tensors directly, so state
        chaining is correct by construction.
        """
        self.work_cache = _WorkCache([dict(st) if "conv" in st else st
                                      for st in self.static_layers])

    def _restore(self, snap):
        for st, sn in zip(self.static_layers, snap):
            if "conv" in st:
                for nm in st["conv"]:
                    st["conv"][nm].copy_(sn["conv"][nm])
                st["recurrent"].copy_(sn["recurrent"])
            else:
                st["c_kv_buf"].copy_(sn["c_kv_buf"])
                st["pos"].copy_(sn["pos"])
                if self.dr > 0:
                    st["k_rope_buf"].copy_(sn["k_rope_buf"])
        self._rebuild_work_cache()

    # ------------------------------------------------------------------ #
    # the captured step
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _step(self):
        """One decode step, fully on static tensors (the captured region).

        ``model.model`` reassigns ``work_cache.layers[i]`` with the freshly
        returned state dicts during capture; the copy-back loop below then
        writes them into the static KDA tensors *inside the captured region*,
        which is what makes one replay a complete, correctly chained step.
        MLA layers mutate their state in place and need no copy-back.
        """
        if self.input_mode == "embeds":
            hidden, _ = self.backbone(inputs_embeds=self.input_static,
                                      cache=self.work_cache, use_cache=True)
        else:
            hidden, _ = self.backbone(self.ids_static, cache=self.work_cache,
                                      use_cache=True)
        layers = self.work_cache.layers
        for i, st in enumerate(self.static_layers):
            if "conv" in st:                                   # KDA layer
                new = layers[i]
                for nm in st["conv"]:
                    st["conv"][nm].copy_(new["conv"][nm])
                st["recurrent"].copy_(new["recurrent"])
        out = hidden[:, -1, :]
        if self.lm_head is not None:
            out = self.lm_head(out)
        self.logits_static.copy_(out)

    @torch.no_grad()
    def _capture(self):
        torch.cuda.synchronize()
        # snapshot the post-prefill state FIRST: warmup below (and the capture
        # itself) *execute* the step, mutating the static state
        snap = self._snapshot()
        # warm up on a side stream (PyTorch's recipe for cuda graph capture)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.warmup_iters):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        # warmup executed the step `warmup_iters` times -> restore, capture
        # (which executes once more), restore again
        self._restore(snap)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()
        self._restore(snap)
        torch.cuda.synchronize()

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor = None,
                inputs_embeds: torch.Tensor = None) -> torch.Tensor:
        """Eager prefill + (re)capture. Returns last-token fp32 logits.

        Exactly one of ``input_ids`` (``[B, S]``) / ``inputs_embeds``
        (``[B, S, d]``) is required -- the latter is what wrapper models like
        the music adapter feed.
        """
        # re-assert: autotuning a new conv shape between capture and replay
        # corrupts the captured graph (see __init__)
        torch.backends.cudnn.benchmark = False
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("prefill needs input_ids or inputs_embeds")
            src = input_ids
            B, S = input_ids.shape
        else:
            src = inputs_embeds
            B, S = inputs_embeds.shape[0], inputs_embeds.shape[1]
        if not src.is_cuda:
            raise ValueError("GraphedDecoder requires CUDA tensors")
        need_rebuild = (self.graph is None or B != self._B
                        or S + self.max_new_tokens > self._L_max)
        if need_rebuild and self.graph is not None:
            # capacity grew: drop the old graph so its pool is released
            self.graph = None
        if need_rebuild:
            dtype = next(self.model.parameters()).dtype
            self._alloc_static(B, S, src.device, dtype)
        if inputs_embeds is None:
            hidden, cache = self.backbone(input_ids, use_cache=True)
        else:
            hidden, cache = self.backbone(inputs_embeds=inputs_embeds,
                                          use_cache=True)
        if self.graph is None:
            self._build_static_from(cache, S)
            self._capture()
        else:
            self._fill_static(cache, S)
        self._avail = self._L_max - S
        out = hidden[:, -1, :]
        if self.lm_head is not None:
            out = self.lm_head(out)
        return out.float()

    def step(self, token_ids: torch.Tensor = None,
             embeds: torch.Tensor = None) -> torch.Tensor:
        """Advance one token.

        ``token_ids``: [B, 1] int64 (``input_mode="ids"``) or ``embeds``:
        [B, 1, d] (``input_mode="embeds"``) on the model device.
        Returns ``[B, V]`` logits in the storage dtype (a view of the static
        output buffer -- consume before the next call).  Exactly one kernel
        launch from the host side (plus the input copy).
        """
        if self.graph is None:
            raise RuntimeError("call prefill() before step()")
        if self._avail <= 0:
            raise RuntimeError(
                "MLA static buffer exhausted: re-prefill with a larger "
                "max_new_tokens (capacity is fixed at capture time)")
        if self.input_mode == "embeds":
            if embeds is None:
                raise ValueError("step() needs embeds in input_mode='embeds'")
            self.input_static.copy_(embeds)
        else:
            if token_ids is None:
                raise ValueError("step() needs token_ids in input_mode='ids'")
            self.ids_static.copy_(token_ids)
        self.graph.replay()
        self._avail -= 1
        return self.logits_static

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = None,
                 do_sample: bool = False, temperature: float = 1.0,
                 top_k: int = 0) -> torch.Tensor:
        """Greedy / top-k generation on the graphed decode path."""
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        if max_new_tokens > self.max_new_tokens:
            self.max_new_tokens = max_new_tokens
            self.graph = None                    # force re-capture with capacity
        logits = self.prefill(input_ids)
        generated = [input_ids]
        cur = None
        for _ in range(max_new_tokens):
            if do_sample:
                probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
                if top_k > 0:
                    topv, _ = torch.topk(probs, top_k, dim=-1)
                    probs = torch.where(probs >= topv[..., -1:], probs,
                                        torch.zeros_like(probs))
                    probs = probs / probs.sum(-1, keepdim=True)
                cur = torch.multinomial(probs, 1)
            else:
                cur = logits.argmax(-1, keepdim=True)           # [B, 1]
            generated.append(cur)
            logits = self.step(cur).float()
        return torch.cat(generated, dim=1)
