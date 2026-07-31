"""Segment gradient accumulation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch import Tensor

from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId
from sequential_segmented_llm_training_inference.storage.runtime_records import (
    AttentionSegmentGradientRecord,
    MLPSegmentGradientRecord,
)
from sequential_segmented_llm_training_inference.optimization.segment_optimizer import (
    clone_gradient_to_cpu,
)


GradientRecord = AttentionSegmentGradientRecord | MLPSegmentGradientRecord


@dataclass(slots=True)
class SegmentGradientAccumulator:
    """Accumulate parameter gradients by logical SegmentId.

    The accumulator stores CPU tensors so it can be used with strict
    single-segment execution and later reloaded segments.
    """

    _gradients: dict[SegmentId, dict[str, Tensor]] = field(default_factory=dict)

    def accumulate(
        self,
        segment_id: SegmentId,
        gradients: dict[str, Tensor],
        *,
        scale: float = 1.0,
    ) -> None:
        if not isinstance(segment_id, SegmentId):
            raise TypeError("segment_id must be a SegmentId.")
        if scale == 0:
            raise ValueError("scale must be non-zero.")
        if not gradients:
            raise ValueError("gradients must not be empty.")

        bucket = self._gradients.setdefault(segment_id, {})
        for name, gradient in gradients.items():
            if not isinstance(name, str) or not name:
                raise ValueError("gradient names must be non-empty strings.")
            if not isinstance(gradient, Tensor):
                raise TypeError(f"Gradient for {name!r} must be a torch.Tensor.")
            value = clone_gradient_to_cpu(gradient)
            if scale != 1.0:
                value = value * float(scale)
            if name in bucket:
                if tuple(bucket[name].shape) != tuple(value.shape):
                    raise ValueError(
                        f"Gradient shape mismatch while accumulating {segment_id.to_key()}.{name}: "
                        f"existing {tuple(bucket[name].shape)}, new {tuple(value.shape)}."
                    )
                bucket[name] = bucket[name] + value
            else:
                bucket[name] = value

    def accumulate_record(self, record: GradientRecord, *, scale: float = 1.0) -> None:
        self.accumulate(record.segment_id, record.parameter_gradients, scale=scale)

    def accumulate_records(
        self, records: Iterable[GradientRecord], *, scale: float = 1.0
    ) -> None:
        for record in records:
            self.accumulate_record(record, scale=scale)

    def has_segment(self, segment_id: SegmentId) -> bool:
        return segment_id in self._gradients

    def get(self, segment_id: SegmentId) -> dict[str, Tensor]:
        if segment_id not in self._gradients:
            raise KeyError(f"No accumulated gradients for {segment_id.to_key()}.")
        return {name: grad.clone() for name, grad in self._gradients[segment_id].items()}

    def pop(self, segment_id: SegmentId) -> dict[str, Tensor]:
        if segment_id not in self._gradients:
            raise KeyError(f"No accumulated gradients for {segment_id.to_key()}.")
        gradients = self._gradients.pop(segment_id)
        return {name: grad.clone() for name, grad in gradients.items()}

    def segment_ids(self) -> list[SegmentId]:
        return sorted(self._gradients)

    def clear(self) -> None:
        self._gradients.clear()

    @property
    def num_segments(self) -> int:
        return len(self._gradients)

    @property
    def num_parameter_gradients(self) -> int:
        return sum(len(bucket) for bucket in self._gradients.values())

    def is_empty(self) -> bool:
        return not self._gradients
