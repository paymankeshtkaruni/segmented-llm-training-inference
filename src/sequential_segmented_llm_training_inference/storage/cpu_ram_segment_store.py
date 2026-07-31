"""CPU RAM offload segment storage backend.

This backend stores inactive segment state dictionaries as CPU tensors in
process memory. It is intended for GPU execution where inactive segments live
in CPU RAM and the active segment is copied to the GPU by the segment loader.
"""

from __future__ import annotations

from typing import Any

from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId
from sequential_segmented_llm_training_inference.storage.segment_store import (
    SegmentStore,
    StateDict,
    clone_state_dict_to_cpu,
)


class CpuRamSegmentStore(SegmentStore):
    """Store inactive segment state dictionaries in CPU RAM."""

    def __init__(self) -> None:
        self._segments: dict[SegmentId, dict[str, Any]] = {}

    def save_segment(self, segment_id: SegmentId, state_dict: StateDict) -> None:
        self._segments[segment_id] = clone_state_dict_to_cpu(state_dict)

    def load_segment(self, segment_id: SegmentId) -> dict[str, Any]:
        if segment_id not in self._segments:
            raise FileNotFoundError(f"Segment not found: {segment_id.to_key()}")
        return clone_state_dict_to_cpu(self._segments[segment_id])

    def has_segment(self, segment_id: SegmentId) -> bool:
        return segment_id in self._segments

    def list_segments(self) -> list[SegmentId]:
        return sorted(self._segments)

    def clear(self) -> None:
        """Remove all stored segment states."""

        self._segments.clear()
