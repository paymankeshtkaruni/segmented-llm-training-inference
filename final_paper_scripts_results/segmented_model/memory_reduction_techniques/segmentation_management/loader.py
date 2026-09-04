"""
StrictSegmentLoader — the one-active-segment invariant.

This is the heart of the memory mechanism: at most ONE segment module is ever
materialized on the compute device. The loader builds a segment (fresh-initialized,
or with weights loaded from a store), hands it out, and on release moves its weights
back to the store (if asked) and frees the device copy (`empty_cache` on CUDA). The
`active_segment_count <= 1` invariant is asserted, so any accidental "hold two
segments" is caught immediately.

`build_segment(key, model, seg)` constructs the correct module type for a key with
the right sliced dimensions (segments.py). The loader never builds the full model.
"""

from __future__ import annotations

import gc
import random
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn

from config import ModelConfig, SegmentationConfig
from segments import (AttentionHeadSegment, MLPHiddenChunk, EmbeddingSlice,
                      OutputHeadSlice, vocab_range, dmodel_range)
from stores import SegmentStore, SegmentKey
from memory import free_device  # gc + empty_cache(CUDA) + malloc_trim(CPU)


def build_segment(key: SegmentKey, m: ModelConfig, s: SegmentationConfig) -> nn.Module:
    """Construct the segment module for `key` with correctly-sliced dimensions."""
    if key.kind == "attention":
        return AttentionHeadSegment(m.d_model, m.n_heads, s.attention_segments, m.dropout)
    if key.kind == "mlp":
        return MLPHiddenChunk(m.d_model, m.d_ff, s.mlp_chunks, m.dropout)
    if key.kind == "embedding":
        d0, d1 = dmodel_range(m.d_model, s.embedding_segments, key.seg)
        return EmbeddingSlice(m.vocab_size, d1 - d0, m.max_seq_len, m.dropout)
    if key.kind == "output_head":
        v0, v1 = vocab_range(m.vocab_size, s.output_head_segments, key.seg)
        return OutputHeadSlice(m.d_model, v1 - v0)
    if key.kind == "attn_out_proj":
        # the per-layer attention output projection W_o (one segment per layer), streamed
        # instead of resident — keeps its ~226 MB of weights off the constrained device.
        return nn.Linear(m.d_model, m.d_model)
    raise ValueError(f"unknown segment kind {key.kind!r}")


def _flag(tech, name: str) -> bool:
    """Read a technique flag; tech=None (default) means EVERY technique ON (unchanged
    behavior). The ablation drivers pass a techniques.Tech to switch some OFF."""
    return True if tech is None else bool(getattr(tech, name))


class StrictSegmentLoader:
    def __init__(self, model: ModelConfig, seg: SegmentationConfig,
                 store: SegmentStore, device: str, tech=None):
        self.model, self.seg, self.store, self.device = model, seg, store, device
        self.tech = tech
        # stream_segments ON  -> one segment resident, rest parked in the store (default).
        # stream_segments OFF -> lazy-loading disabled: every touched segment is KEPT
        #   resident on the device (cached below), never parked/freed -> peak holds ALL
        #   segment weights at once (the naive baseline).
        self._stream = _flag(tech, "stream_segments")
        # segment_wo ON (default): W_o (attn_out_proj) streams like any segment. OFF: W_o is
        # kept RESIDENT on the device even while everything else streams — isolating its
        # ~226 MB resident-weight floor (the #7e reduction, as its own ladder rung).
        self._segment_wo = _flag(tech, "segment_wo")
        self._resident: dict = {}                      # device-resident cache (see _is_resident)
        self._active: Optional[nn.Module] = None
        self._active_key: Optional[SegmentKey] = None
        # train/eval mode applied to every loaded segment — MUST match the rest of
        # the model, else dropout differs and segmented != reference (caught by the
        # forward-engine identity test).
        self.training: bool = True
        # OPTIONAL cost profiler (default None = no overhead, verified behavior intact).
        # If set, its on_load(key)/on_release(key) bracket each segment's residency so a
        # cost run can measure per-segment time + peak memory. Never used by the
        # correctness path.
        self.profiler = None

    def set_training(self, mode: bool) -> "StrictSegmentLoader":
        self.training = mode
        return self

    @property
    def active_segment_count(self) -> int:
        return 0 if self._active is None else 1

    def _is_resident(self, key: SegmentKey) -> bool:
        """A segment stays resident on-device (cached, never freed) if streaming is OFF
        (all-resident baseline) OR it is W_o with segment_wo OFF (W_o kept resident while
        everything else streams — isolates its resident-weight floor)."""
        if not self._stream:
            return True
        return key.kind == "attn_out_proj" and not self._segment_wo

    def load_segment(self, key: SegmentKey, *, init_if_missing: bool = True) -> nn.Module:
        if self._active is not None:
            raise RuntimeError(
                f"one-active-segment violated: {self._active_key} still active "
                f"while loading {key}")
        # reuse the device-resident copy if we already built it (all-resident baseline, or
        # W_o with segment_wo OFF) — the cached module is the source of truth, store bypassed.
        if self._is_resident(key) and key in self._resident:
            self._active, self._active_key = self._resident[key], key
            # re-apply the CURRENT train/eval flag: the cached module keeps the flag it
            # was built with, so an eval->train transition would otherwise leave its
            # internal dropouts silently off (caught by X1: T3-vs-T1 grad delta 3.3e-3).
            self._active.train(self.training)
            if self.profiler is not None:
                self.profiler.on_load(key)
            return self._active
        # RNG-neutral build (mirrors execution/rng.py capture/restore): module
        # constructors CONSUME the global CPU RNG for their immediately-overwritten
        # init draws. On CPU, dropout masks draw from that same generator, so an
        # unpreserved build makes a streaming run draw different masks than a
        # resident run (caught by X1: CPU T3-vs-T1 grad delta 3.8e-3). Constructors
        # never touch CUDA RNG, so only CPU/python state needs preserving.
        _py_state, _cpu_state = random.getstate(), torch.get_rng_state()
        module = build_segment(key, self.model, self.seg)
        random.setstate(_py_state); torch.set_rng_state(_cpu_state)
        sd = self.store.get(key)
        if sd is not None:
            module.load_state_dict(sd, strict=True)
        elif not init_if_missing:
            raise KeyError(f"segment {key} not in store and init_if_missing=False")
        else:
            # first touch: persist the fresh-initialized weights so future loads match
            self.store.put(key, module.state_dict())
        module.to(self.device)
        module.train(self.training)   # match the model's mode (dropout on/off)
        if isinstance(module, AttentionHeadSegment):
            module.use_sdpa = _flag(self.tech, "sdpa")   # SDPA vs explicit score matrix
        if self._is_resident(key):
            self._resident[key] = module              # keep on device (never freed)
        self._active, self._active_key = module, key
        if self.profiler is not None:
            self.profiler.on_load(key)
        return module

    def release_segment(self, *, save: bool = False) -> None:
        if self._active is None:
            return
        if save:
            self.store.put(self._active_key, self._active.state_dict())
        if self.profiler is not None:
            self.profiler.on_release(self._active_key)   # read this segment's peak before freeing
        resident = self._is_resident(self._active_key)
        self._active = None
        self._active_key = None
        # resident keys (all-resident baseline, or W_o w/ segment_wo OFF): keep in
        # self._resident — do NOT free.
        # free_device OFF: keep torch's cached VRAM / glibc heap (skip empty_cache/trim).
        if not resident and _flag(self.tech, "free_device"):
            free_device(self.device)   # gc + empty_cache (CUDA) / malloc_trim (CPU)

    @contextmanager
    def acquire_segment(self, key: SegmentKey, *, save_on_exit: bool = False,
                        init_if_missing: bool = True):
        module = self.load_segment(key, init_if_missing=init_if_missing)
        try:
            yield module
        finally:
            self.release_segment(save=save_on_exit)


if __name__ == "__main__":
    from config import SMALL_MODEL, SEG_8x2x2x8
    from stores import make_store
    torch.manual_seed(0)
    m, s = SMALL_MODEL, SEG_8x2x2x8
    loader = StrictSegmentLoader(m, s, make_store("cpu_ram"), "cpu")
    seen_counts = []
    # iterate several segments one at a time; the invariant must hold throughout
    keys = ([SegmentKey(-1, "embedding", i) for i in range(s.embedding_segments)] +
            [SegmentKey(0, "attention", i) for i in range(s.attention_segments)] +
            [SegmentKey(0, "mlp", i) for i in range(s.mlp_chunks)] +
            [SegmentKey(-1, "output_head", i) for i in range(s.output_head_segments)])
    for k in keys:
        with loader.acquire_segment(k, save_on_exit=True) as seg:
            seen_counts.append(loader.active_segment_count)   # must be 1 inside
            assert sum(p.numel() for p in seg.parameters()) > 0
        assert loader.active_segment_count == 0               # must be 0 after release
    # second pass: weights must reload identically from the store (not re-init)
    k0 = keys[len(keys)//2]
    with loader.acquire_segment(k0) as seg:
        w_reload = next(iter(seg.state_dict().values())).clone()
    w_store = next(iter(loader.store.get(k0).values()))
    print(f"iterated {len(keys)} segments | active inside always 1: {set(seen_counts)=={1}} | "
          f"active after release always 0 | reload==store: {torch.allclose(w_reload, w_store)}")
    print("LOADER ONE-ACTIVE INVARIANT OK")
