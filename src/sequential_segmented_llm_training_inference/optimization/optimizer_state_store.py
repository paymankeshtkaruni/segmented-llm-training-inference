"""Pluggable optimizer-state store for segment-wise optimizers.

In-memory (default) holds all segment state in RAM (current behaviour).
Disk backend keeps one segment's state file at a time, streaming the rest to
disk -- for memory-minimal CPU training. Keyed by SegmentId: one file holds all
of that segment's parameter states ({param_name: {step, exp_avg, exp_avg_sq}}).

The optimizer state is *persistent* across steps/epochs (unlike runtime records
and accumulated gradients). It is removed only on an explicit ``clear()`` (new
run / reset) and is included in checkpoints via the optimizer ``state_dict``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import shutil

import torch

from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId


# A segment's optimizer state: {parameter_name: {state_field: value}}
SegmentOptState = dict[str, dict[str, object]]


class OptimizerStateStore(ABC):
    """Abstract per-segment optimizer-state store."""

    @abstractmethod
    def get(self, segment_id: SegmentId) -> SegmentOptState:
        """Return the segment's optimizer state, or an empty dict if absent."""

    @abstractmethod
    def put(self, segment_id: SegmentId, state: SegmentOptState) -> None:
        """Store the segment's optimizer state."""

    @abstractmethod
    def has(self, segment_id: SegmentId) -> bool:
        """Return whether any state exists for the segment."""

    @abstractmethod
    def segment_ids(self) -> list[SegmentId]:
        """Return all segment ids with stored state, sorted."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all stored state (explicit reset / new run)."""


class InMemoryOptimizerStateStore(OptimizerStateStore):
    """Holds all segment optimizer state in RAM (default, current behaviour)."""

    def __init__(self) -> None:
        self._store: dict[SegmentId, SegmentOptState] = {}

    def get(self, segment_id: SegmentId) -> SegmentOptState:
        return self._store.get(segment_id, {})

    def put(self, segment_id: SegmentId, state: SegmentOptState) -> None:
        self._store[segment_id] = state

    def has(self, segment_id: SegmentId) -> bool:
        return segment_id in self._store

    def segment_ids(self) -> list[SegmentId]:
        return sorted(self._store.keys())

    def clear(self) -> None:
        self._store.clear()


class DiskOptimizerStateStore(OptimizerStateStore):
    """One ``.pt`` file per segment. Persistent across steps; cleared only on
    ``clear()``.

    Only one segment's optimizer state is resident in RAM at a time during the
    apply pass, instead of every segment's state for the whole run.
    """

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._keys: set[SegmentId] = set()

    def _path(self, segment_id: SegmentId) -> Path:
        return self.root_dir / f"{segment_id.to_path_name()}.pt"

    def get(self, segment_id: SegmentId) -> SegmentOptState:
        path = self._path(segment_id)
        if not path.exists():
            return {}
        return torch.load(path, map_location="cpu", weights_only=False)

    def put(self, segment_id: SegmentId, state: SegmentOptState) -> None:
        # Tensors are written to disk as CPU tensors by the optimizers.
        torch.save(state, self._path(segment_id))
        self._keys.add(segment_id)

    def has(self, segment_id: SegmentId) -> bool:
        return segment_id in self._keys or self._path(segment_id).exists()

    def segment_ids(self) -> list[SegmentId]:
        # SegmentId has no from_path_name(); the store is persistent within a
        # run so the in-memory key set is authoritative. load_state_dict()
        # repopulates _keys after a checkpoint restore.
        return sorted(self._keys)

    def clear(self) -> None:
        if self.root_dir.exists():
            shutil.rmtree(self.root_dir, ignore_errors=True)
            self.root_dir.mkdir(parents=True, exist_ok=True)
        self._keys.clear()
