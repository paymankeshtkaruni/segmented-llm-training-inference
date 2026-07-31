"""Abstract segment storage interface.

Segment stores keep inactive attention/MLP segment state dictionaries outside
the active compute path. A SegmentStore does not keep live modules; it stores
and retrieves serializable PyTorch state dictionaries keyed by SegmentId.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId


StateDict = Mapping[str, Any]


class SegmentStore(ABC):
    """Abstract interface for inactive segment storage backends."""

    @abstractmethod
    def save_segment(self, segment_id: SegmentId, state_dict: StateDict) -> None:
        """Persist one segment state dictionary."""

    @abstractmethod
    def load_segment(self, segment_id: SegmentId) -> dict[str, Any]:
        """Load and return one segment state dictionary."""

    @abstractmethod
    def has_segment(self, segment_id: SegmentId) -> bool:
        """Return True if the segment exists in this store."""

    @abstractmethod
    def list_segments(self) -> list[SegmentId]:
        """Return all segment ids in deterministic sorted order."""

    def require_segment(self, segment_id: SegmentId) -> None:
        """Raise FileNotFoundError if a segment is missing."""

        if not self.has_segment(segment_id):
            raise FileNotFoundError(f"Segment not found: {segment_id.to_key()}")


def clone_state_dict_to_cpu(state_dict: StateDict) -> dict[str, Any]:
    """Clone tensors to CPU and copy non-tensor values safely.

    This prevents later in-place modifications of a live module from mutating
    the stored segment state.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a project dependency
        raise RuntimeError("torch is required for segment storage.") from exc

    copied: dict[str, Any] = {}
    for name, value in state_dict.items():
        if torch.is_tensor(value):
            copied[name] = value.detach().cpu().clone()
        else:
            copied[name] = value
    return copied
