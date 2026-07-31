"""Disk streaming segment storage backend.

This backend stores one ``.pt`` file per segment. It is the required backend
for CPU execution and an optional backend for GPU execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId
from sequential_segmented_llm_training_inference.storage.segment_store import (
    SegmentStore,
    StateDict,
    clone_state_dict_to_cpu,
)


class DiskSegmentStore(SegmentStore):
    """Store inactive segment state dictionaries as deterministic disk files."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def segment_path(self, segment_id: SegmentId) -> Path:
        """Return the deterministic path for a segment state file."""

        return self.root_dir / f"{segment_id.to_path_name()}.pt"

    def save_segment(self, segment_id: SegmentId, state_dict: StateDict) -> None:
        path = self.segment_path(segment_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(clone_state_dict_to_cpu(state_dict), path)

    def load_segment(self, segment_id: SegmentId) -> dict[str, Any]:
        path = self.segment_path(segment_id)
        if not path.exists():
            raise FileNotFoundError(f"Segment file not found: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    def has_segment(self, segment_id: SegmentId) -> bool:
        return self.segment_path(segment_id).exists()

    def list_segments(self) -> list[SegmentId]:
        segment_ids: list[SegmentId] = []
        for path in sorted(self.root_dir.glob("layer_*__*__segment_*.pt")):
            segment_ids.append(self._segment_id_from_path(path))
        return sorted(segment_ids)

    @staticmethod
    def _segment_id_from_path(path: Path) -> SegmentId:
        stem = path.stem
        parts = stem.split("__")
        if len(parts) != 3:
            raise ValueError(f"Invalid segment file name: {path.name}")

        layer_part, segment_type, segment_part = parts
        if not layer_part.startswith("layer_"):
            raise ValueError(f"Invalid layer part in segment file: {path.name}")
        if not segment_part.startswith("segment_"):
            raise ValueError(f"Invalid segment part in segment file: {path.name}")

        return SegmentId(
            layer_id=int(layer_part.removeprefix("layer_")),
            segment_type=segment_type,
            segment_id=int(segment_part.removeprefix("segment_")),
        )
