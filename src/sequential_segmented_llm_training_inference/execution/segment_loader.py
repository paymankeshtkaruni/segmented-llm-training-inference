"""Strict single-segment loader.

Phase 8 scope:
- Load one segment module from a SegmentStore into the compute device.
- Enforce the strict invariant: at most one active segment at any time.
- Release the active segment, optionally saving updated weights back to storage.
- Provide a context-manager interface for safe acquire/release usage.

The loader does not create training logic, forward composition, or backward
recomputation. Those are later phases. It only manages live segment lifecycle.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
from torch import nn

from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId
from sequential_segmented_llm_training_inference.storage.segment_store import SegmentStore
from sequential_segmented_llm_training_inference.training.profiler import (
    SegmentedTrainingProfiler,
)


SegmentModuleFactory = Callable[[SegmentId], nn.Module]


def _module_state_nbytes(module: nn.Module) -> int:
    """Return the total byte size of a module's state dict tensors."""

    total = 0
    for tensor in module.state_dict().values():
        if isinstance(tensor, torch.Tensor):
            total += tensor.numel() * tensor.element_size()
    return total


@dataclass(slots=True)
class ActiveSegment:
    """Runtime record for the currently loaded segment module."""

    segment_id: SegmentId
    module: nn.Module
    device: torch.device


class StrictSegmentLoader:
    """Load and release segments under a strict one-active-segment invariant.

    Args:
        segment_store: Storage backend containing inactive segment state dicts.
        module_factory: Callable that creates a fresh module for a SegmentId.
        device: Compute device to move the loaded module to.

    Invariant:
        ``active_segment_count <= 1`` must always hold.
    """

    def __init__(
        self,
        *,
        segment_store: SegmentStore,
        module_factory: SegmentModuleFactory,
        device: str | torch.device = "cpu",
        profiler: SegmentedTrainingProfiler | None = None,
    ) -> None:
        if not callable(module_factory):
            raise TypeError("module_factory must be callable.")

        self.segment_store = segment_store
        self.module_factory = module_factory
        self.device = torch.device(device)
        self.profiler = profiler
        self._active: ActiveSegment | None = None
        self._t0: float = time.perf_counter()

    def _read_memory(self) -> int | None:
        """Read current memory for the active device.

        - GPU: returns ``torch.cuda.memory_allocated`` in bytes (not reserved).
        - CPU: returns process RSS in bytes via psutil, or None if unavailable.
        """
        if self.device.type == "cuda":
            try:
                return int(torch.cuda.memory_allocated(self.device))
            except Exception:
                return None
        else:
            try:
                import psutil
                return int(psutil.Process().memory_info().rss)
            except Exception:
                return None

    @property
    def active_segment(self) -> ActiveSegment | None:
        """Return the active segment record, if one exists."""

        return self._active

    @property
    def active_segment_id(self) -> SegmentId | None:
        """Return the active SegmentId, if one exists."""

        if self._active is None:
            return None
        return self._active.segment_id

    @property
    def active_module(self) -> nn.Module | None:
        """Return the active live module, if one exists."""

        if self._active is None:
            return None
        return self._active.module

    @property
    def active_segment_count(self) -> int:
        """Return 1 if a segment is active, otherwise 0."""

        return 0 if self._active is None else 1

    def has_active_segment(self) -> bool:
        """Return True when a segment is currently active."""

        return self._active is not None

    def require_no_active_segment(self) -> None:
        """Raise RuntimeError if a segment is already active."""

        if self._active is not None:
            raise RuntimeError(
                "Strict single-segment invariant violation: "
                f"segment {self._active.segment_id.to_key()} is already active. "
                "Release it before loading another segment."
            )

    def load_segment(self, segment_id: SegmentId) -> nn.Module:
        """Load one segment module to the compute device and mark it active."""

        self.require_no_active_segment()
        self.segment_store.require_segment(segment_id)

        # Record memory BEFORE loading so per-segment plots have a before/after/release triplet.
        mem_before = self._read_memory()
        elapsed_before = time.perf_counter() - self._t0
        if self.profiler is not None and mem_before is not None:
            self.profiler.record_segment_memory_event(
                segment_key=segment_id.to_key(),
                segment_type=segment_id.segment_type,
                memory_bytes=0,
                absolute_memory_bytes=mem_before,
                event="before_load",
                elapsed_s=elapsed_before,
            )

        start = time.perf_counter()
        module = self.module_factory(segment_id)
        if not isinstance(module, nn.Module):
            raise TypeError(
                "module_factory must return torch.nn.Module, got "
                f"{type(module).__name__}."
            )

        state_dict = self.segment_store.load_segment(segment_id)
        missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=True)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "Failed to load segment state strictly. "
                f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
            )

        module.to(self.device)
        self._active = ActiveSegment(
            segment_id=segment_id,
            module=module,
            device=self.device,
        )
        if self.profiler is not None:
            self.profiler.record_segment_load(seconds=time.perf_counter() - start)
            self.profiler.increment("segment_bytes_loaded", _module_state_nbytes(module))

        mem_after = self._read_memory()
        elapsed = time.perf_counter() - self._t0
        if self.profiler is not None:
            # Use exact parameter bytes for the per-segment/type peak on both CPU and GPU.
            # RSS delta (CPU) is unreliable because freed pages are reused immediately.
            # memory_allocated delta (GPU) picks up always-resident tensors and allocator
            # noise. Parameter bytes are exact and device-independent.
            footprint = _module_state_nbytes(module)
            self.profiler.record_segment_memory_event(
                segment_key=segment_id.to_key(),
                segment_type=segment_id.segment_type,
                memory_bytes=footprint,
                absolute_memory_bytes=mem_after,
                event="load",
                elapsed_s=elapsed,
            )

        return module

    def save_active_segment(self) -> None:
        """Save the active module state back to the segment store."""

        if self._active is None:
            raise RuntimeError("No active segment to save.")
        start = time.perf_counter()
        self.segment_store.save_segment(
            self._active.segment_id,
            self._active.module.state_dict(),
        )
        if self.profiler is not None:
            self.profiler.record_segment_save(seconds=time.perf_counter() - start)
            self.profiler.increment(
                "segment_bytes_saved", _module_state_nbytes(self._active.module)
            )

    def release_segment(self, *, save: bool = False) -> None:
        """Release the active segment.

        Args:
            save: If True, save the active module state before releasing it.
        """

        if self._active is None:
            raise RuntimeError("No active segment to release.")

        # Capture segment identity before clearing _active.
        prev_seg_key = self._active.segment_id.to_key()
        prev_seg_type = self._active.segment_id.segment_type

        if save:
            self.save_active_segment()
        elif self.profiler is not None:
            self.profiler.record_segment_unload()

        # Drop the only live module reference managed by this loader.
        # On CUDA, call empty_cache() so the freed segment memory is returned to
        # the driver immediately — otherwise PyTorch's allocator keeps it in its
        # pool and nvidia-smi still counts it, masking the actual VRAM savings.
        device = self._active.device
        self._active = None
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Record memory after release — only used for the timeline, not for peaks.
        mem = self._read_memory()
        elapsed = time.perf_counter() - self._t0
        if self.profiler is not None and mem is not None:
            self.profiler.record_segment_memory_event(
                segment_key=prev_seg_key,
                segment_type=prev_seg_type,
                memory_bytes=mem,
                absolute_memory_bytes=mem,
                event="release",
                elapsed_s=elapsed,
            )

    def discard_active_segment(self) -> None:
        """Release the active segment without saving updated state."""

        self.release_segment(save=False)

    @contextmanager
    def acquire_segment(
        self,
        segment_id: SegmentId,
        *,
        save_on_exit: bool = False,
    ) -> Iterator[nn.Module]:
        """Context manager that loads one segment and releases it on exit."""

        module = self.load_segment(segment_id)
        try:
            yield module
        finally:
            if self.has_active_segment():
                self.release_segment(save=save_on_exit)
