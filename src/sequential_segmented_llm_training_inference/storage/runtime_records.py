"""Segment-level runtime execution, composition, and gradient records.

Phase 9 implements the runtime record layer used by segmented forward and
recomputation-based segmented backward.

Design constraints:
- Runtime execution records are segment-level, not block-level.
- The transformer layer id is metadata used for ordering/scope.
- Attention records must be keyed by attention SegmentId values.
- MLP records must be keyed by MLP SegmentId values.
- Shared input references are supported so the same normalized tensor can be
  stored once and referenced by many segment records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional

import torch
from torch import Tensor

from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId
from sequential_segmented_llm_training_inference.storage.runtime_record_store_backends import (
    RuntimeRecordBackend,
)


CompositionType = Literal["concat", "sum"]
ResidualBranchType = Literal["attention", "mlp"]


def _validate_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")


def _validate_segment_type(segment_id: SegmentId, expected: str) -> None:
    if segment_id.segment_type != expected:
        raise ValueError(
            f"Expected a {expected!r} SegmentId, got {segment_id.segment_type!r}."
        )


def _shape_tuple(tensor: Tensor | None) -> tuple[int, ...] | None:
    if tensor is None:
        return None
    return tuple(int(dim) for dim in tensor.shape)


def _device_string(tensor: Tensor | None) -> str | None:
    if tensor is None:
        return None
    return str(tensor.device)


def _dtype_string(tensor: Tensor | None) -> str | None:
    if tensor is None:
        return None
    return str(tensor.dtype)


def _maybe_detach_clone(tensor: Tensor | None, detach: bool, clone: bool) -> Tensor | None:
    if tensor is None:
        return None
    result = tensor.detach() if detach else tensor
    return result.clone() if clone else result


@dataclass(slots=True)
class AttentionSegmentExecutionRecord:
    """Forward execution record for one attention segment.

    The record stores exactly the information needed to recompute this segment
    during backward. The input may be held directly as ``input_tensor`` or by a
    shared ``input_reference`` key managed by a higher-level runtime store.
    """

    segment_id: SegmentId
    step_id: int
    microbatch_id: int
    output_tensor: Tensor
    input_reference: str | None = None
    input_tensor: Tensor | None = None
    attention_mask_reference: str | None = None
    attention_mask_tensor: Tensor | None = None
    rng_state: dict[str, Any] | None = None
    dtype: str | None = None
    device: str | None = None
    shape: tuple[int, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_segment_type(self.segment_id, "attention")
        _validate_non_negative_int("step_id", self.step_id)
        _validate_non_negative_int("microbatch_id", self.microbatch_id)
        if not isinstance(self.output_tensor, Tensor):
            raise TypeError("output_tensor must be a torch.Tensor.")
        if self.input_reference is None and self.input_tensor is None:
            raise ValueError("Either input_reference or input_tensor must be provided.")
        if self.shape is None:
            self.shape = _shape_tuple(self.output_tensor)
        if self.dtype is None:
            self.dtype = _dtype_string(self.output_tensor)
        if self.device is None:
            self.device = _device_string(self.output_tensor)

    @property
    def layer_id(self) -> int:
        return self.segment_id.layer_id

    def detached_copy(self, *, clone: bool = True) -> "AttentionSegmentExecutionRecord":
        """Return a copy with tensor fields detached from autograd."""

        return AttentionSegmentExecutionRecord(
            segment_id=self.segment_id,
            step_id=self.step_id,
            microbatch_id=self.microbatch_id,
            output_tensor=_maybe_detach_clone(self.output_tensor, True, clone),
            input_reference=self.input_reference,
            input_tensor=_maybe_detach_clone(self.input_tensor, True, clone),
            attention_mask_reference=self.attention_mask_reference,
            attention_mask_tensor=_maybe_detach_clone(self.attention_mask_tensor, True, clone),
            rng_state=self.rng_state,
            dtype=self.dtype,
            device=self.device,
            shape=self.shape,
            metadata=dict(self.metadata),
        )


@dataclass(slots=True)
class MLPSegmentExecutionRecord:
    """Forward execution record for one MLP segment."""

    segment_id: SegmentId
    step_id: int
    microbatch_id: int
    output_tensor: Tensor
    input_reference: str | None = None
    input_tensor: Tensor | None = None
    rng_state: dict[str, Any] | None = None
    dtype: str | None = None
    device: str | None = None
    shape: tuple[int, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_segment_type(self.segment_id, "mlp")
        _validate_non_negative_int("step_id", self.step_id)
        _validate_non_negative_int("microbatch_id", self.microbatch_id)
        if not isinstance(self.output_tensor, Tensor):
            raise TypeError("output_tensor must be a torch.Tensor.")
        if self.input_reference is None and self.input_tensor is None:
            raise ValueError("Either input_reference or input_tensor must be provided.")
        if self.shape is None:
            self.shape = _shape_tuple(self.output_tensor)
        if self.dtype is None:
            self.dtype = _dtype_string(self.output_tensor)
        if self.device is None:
            self.device = _device_string(self.output_tensor)

    @property
    def layer_id(self) -> int:
        return self.segment_id.layer_id

    def detached_copy(self, *, clone: bool = True) -> "MLPSegmentExecutionRecord":
        """Return a copy with tensor fields detached from autograd."""

        return MLPSegmentExecutionRecord(
            segment_id=self.segment_id,
            step_id=self.step_id,
            microbatch_id=self.microbatch_id,
            output_tensor=_maybe_detach_clone(self.output_tensor, True, clone),
            input_reference=self.input_reference,
            input_tensor=_maybe_detach_clone(self.input_tensor, True, clone),
            rng_state=self.rng_state,
            dtype=self.dtype,
            device=self.device,
            shape=self.shape,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class AttentionCompositionRecord:
    """Record describing attention segment-output composition for one layer."""

    layer_id: int
    segment_order: tuple[int, ...]
    composition_type: CompositionType = "concat"
    output_projection_applied: bool = True
    dropout_applied: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_negative_int("layer_id", self.layer_id)
        if self.composition_type != "concat":
            raise ValueError("Attention composition_type must be 'concat'.")
        if not self.segment_order:
            raise ValueError("segment_order must not be empty.")
        if tuple(sorted(self.segment_order)) != self.segment_order:
            raise ValueError("segment_order must be deterministic and sorted ascending.")
        if len(set(self.segment_order)) != len(self.segment_order):
            raise ValueError("segment_order must not contain duplicates.")


@dataclass(frozen=True, slots=True)
class MLPCompositionRecord:
    """Record describing MLP segment-output composition for one layer."""

    layer_id: int
    segment_order: tuple[int, ...]
    composition_type: CompositionType = "sum"
    shared_output_bias_added_once: bool = True
    dropout_applied: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_negative_int("layer_id", self.layer_id)
        if self.composition_type != "sum":
            raise ValueError("MLP composition_type must be 'sum'.")
        if not self.segment_order:
            raise ValueError("segment_order must not be empty.")
        if tuple(sorted(self.segment_order)) != self.segment_order:
            raise ValueError("segment_order must be deterministic and sorted ascending.")
        if len(set(self.segment_order)) != len(self.segment_order):
            raise ValueError("segment_order must not contain duplicates.")


@dataclass(frozen=True, slots=True)
class ResidualCompositionRecord:
    """Record describing a residual addition branch inside a layer."""

    layer_id: int
    branch_type: ResidualBranchType
    operation: str = "residual_addition"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_negative_int("layer_id", self.layer_id)
        if self.branch_type not in {"attention", "mlp"}:
            raise ValueError("branch_type must be either 'attention' or 'mlp'.")
        if self.operation != "residual_addition":
            raise ValueError("operation must be 'residual_addition'.")


@dataclass(slots=True)
class AttentionSegmentGradientRecord:
    """Gradient record for one attention segment during segmented backward."""

    segment_id: SegmentId
    parameter_gradients: dict[str, Tensor]
    input_gradient: Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_segment_type(self.segment_id, "attention")
        if not isinstance(self.input_gradient, Tensor):
            raise TypeError("input_gradient must be a torch.Tensor.")
        for name, grad in self.parameter_gradients.items():
            if not isinstance(name, str) or not name:
                raise ValueError("parameter gradient names must be non-empty strings.")
            if not isinstance(grad, Tensor):
                raise TypeError(f"Gradient for {name!r} must be a torch.Tensor.")


@dataclass(slots=True)
class MLPSegmentGradientRecord:
    """Gradient record for one MLP segment during segmented backward."""

    segment_id: SegmentId
    parameter_gradients: dict[str, Tensor]
    input_gradient: Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_segment_type(self.segment_id, "mlp")
        if not isinstance(self.input_gradient, Tensor):
            raise TypeError("input_gradient must be a torch.Tensor.")
        for name, grad in self.parameter_gradients.items():
            if not isinstance(name, str) or not name:
                raise ValueError("parameter gradient names must be non-empty strings.")
            if not isinstance(grad, Tensor):
                raise TypeError(f"Gradient for {name!r} must be a torch.Tensor.")


class RuntimeRecordStore:
    """Container for segment-level runtime records for one step/microbatch scope."""

    def __init__(self, backend: RuntimeRecordBackend | None = None) -> None:
        self.attention_records: dict[SegmentId, AttentionSegmentExecutionRecord] = {}
        self.mlp_records: dict[SegmentId, MLPSegmentExecutionRecord] = {}
        self.attention_gradient_records: dict[SegmentId, AttentionSegmentGradientRecord] = {}
        self.mlp_gradient_records: dict[SegmentId, MLPSegmentGradientRecord] = {}
        self.attention_composition_records: dict[int, AttentionCompositionRecord] = {}
        self.mlp_composition_records: dict[int, MLPCompositionRecord] = {}
        self.residual_composition_records: list[ResidualCompositionRecord] = []
        self.shared_tensors: dict[str, Tensor] = {}
        # Metadata for non-tensor execution state needed by true recomputation
        # backward, e.g. dropout RNG states. Values are intentionally generic
        # because RngState lives in execution.rng and should not create a hard
        # storage-layer dependency.
        self.metadata: dict[str, Any] = {}
        self._backend: RuntimeRecordBackend | None = backend
        # In-memory tensor overrides checked before the disk backend. Used by
        # lazy-records backward to provide recomputed intermediates without disk I/O.
        self._local_tensors: dict[str, Tensor] = {}

    def add_local_tensor(self, key: str, tensor: Tensor) -> None:
        """Store a tensor in the in-memory override dict (bypasses disk backend)."""
        self._local_tensors[key] = tensor

    def pop_local_tensor(self, key: str) -> None:
        """Remove a local tensor to release its memory as soon as it is no longer needed."""
        self._local_tensors.pop(key, None)

    def clear_local_tensors(self) -> None:
        """Release all in-memory override tensors at the end of a lazy-records backward layer."""
        self._local_tensors.clear()

    def add_shared_tensor(
        self,
        key: str,
        tensor: Tensor,
        *,
        overwrite: bool = False,
        detach: bool = True,
        clone: bool = False,
    ) -> None:
        if not key:
            raise ValueError("shared tensor key must not be empty.")
        if not isinstance(tensor, Tensor):
            raise TypeError("shared tensor must be a torch.Tensor.")
        processed = _maybe_detach_clone(tensor, detach, clone)
        if self._backend is not None:
            if self._backend.has_tensor(key) and not overwrite:
                raise KeyError(f"Shared tensor {key!r} already exists.")
            self._backend.put_tensor(key, processed)
        else:
            if key in self.shared_tensors and not overwrite:
                raise KeyError(f"Shared tensor {key!r} already exists.")
            self.shared_tensors[key] = processed

    def get_shared_tensor(self, key: str) -> Tensor:
        # In-memory overrides take precedence over the disk backend so that
        # lazy-records backward can inject recomputed intermediates without disk I/O.
        if key in self._local_tensors:
            return self._local_tensors[key]
        if self._backend is not None:
            return self._backend.get_tensor(key)
        try:
            return self.shared_tensors[key]
        except KeyError as exc:
            raise KeyError(f"Shared tensor {key!r} was not found.") from exc

    def add_attention_record(
        self,
        record: AttentionSegmentExecutionRecord,
        *,
        overwrite: bool = False,
    ) -> None:
        _validate_segment_type(record.segment_id, "attention")
        if record.segment_id in self.attention_records and not overwrite:
            raise KeyError(f"Attention record already exists for {record.segment_id}.")
        self.attention_records[record.segment_id] = record

    def get_attention_record(self, segment_id: SegmentId) -> AttentionSegmentExecutionRecord:
        _validate_segment_type(segment_id, "attention")
        try:
            return self.attention_records[segment_id]
        except KeyError as exc:
            raise KeyError(f"No attention record found for {segment_id}.") from exc

    def add_mlp_record(
        self,
        record: MLPSegmentExecutionRecord,
        *,
        overwrite: bool = False,
    ) -> None:
        _validate_segment_type(record.segment_id, "mlp")
        if record.segment_id in self.mlp_records and not overwrite:
            raise KeyError(f"MLP record already exists for {record.segment_id}.")
        self.mlp_records[record.segment_id] = record

    def get_mlp_record(self, segment_id: SegmentId) -> MLPSegmentExecutionRecord:
        _validate_segment_type(segment_id, "mlp")
        try:
            return self.mlp_records[segment_id]
        except KeyError as exc:
            raise KeyError(f"No MLP record found for {segment_id}.") from exc

    def add_attention_gradient_record(
        self,
        record: AttentionSegmentGradientRecord,
        *,
        overwrite: bool = False,
    ) -> None:
        if record.segment_id in self.attention_gradient_records and not overwrite:
            raise KeyError(
                f"Attention gradient record already exists for {record.segment_id}."
            )
        self.attention_gradient_records[record.segment_id] = record

    def get_attention_gradient_record(
        self, segment_id: SegmentId
    ) -> AttentionSegmentGradientRecord:
        _validate_segment_type(segment_id, "attention")
        try:
            return self.attention_gradient_records[segment_id]
        except KeyError as exc:
            raise KeyError(
                f"No attention gradient record found for {segment_id}."
            ) from exc

    def add_mlp_gradient_record(
        self,
        record: MLPSegmentGradientRecord,
        *,
        overwrite: bool = False,
    ) -> None:
        if record.segment_id in self.mlp_gradient_records and not overwrite:
            raise KeyError(f"MLP gradient record already exists for {record.segment_id}.")
        self.mlp_gradient_records[record.segment_id] = record

    def get_mlp_gradient_record(self, segment_id: SegmentId) -> MLPSegmentGradientRecord:
        _validate_segment_type(segment_id, "mlp")
        try:
            return self.mlp_gradient_records[segment_id]
        except KeyError as exc:
            raise KeyError(f"No MLP gradient record found for {segment_id}.") from exc

    def add_attention_composition_record(
        self,
        record: AttentionCompositionRecord,
        *,
        overwrite: bool = False,
    ) -> None:
        if record.layer_id in self.attention_composition_records and not overwrite:
            raise KeyError(
                f"Attention composition record already exists for layer {record.layer_id}."
            )
        self.attention_composition_records[record.layer_id] = record

    def get_attention_composition_record(self, layer_id: int) -> AttentionCompositionRecord:
        try:
            return self.attention_composition_records[layer_id]
        except KeyError as exc:
            raise KeyError(
                f"No attention composition record found for layer {layer_id}."
            ) from exc

    def add_mlp_composition_record(
        self,
        record: MLPCompositionRecord,
        *,
        overwrite: bool = False,
    ) -> None:
        if record.layer_id in self.mlp_composition_records and not overwrite:
            raise KeyError(
                f"MLP composition record already exists for layer {record.layer_id}."
            )
        self.mlp_composition_records[record.layer_id] = record

    def get_mlp_composition_record(self, layer_id: int) -> MLPCompositionRecord:
        try:
            return self.mlp_composition_records[layer_id]
        except KeyError as exc:
            raise KeyError(f"No MLP composition record found for layer {layer_id}.") from exc

    def add_residual_composition_record(self, record: ResidualCompositionRecord) -> None:
        self.residual_composition_records.append(record)

    def list_attention_segment_ids(self) -> list[SegmentId]:
        return sorted(self.attention_records)

    def list_mlp_segment_ids(self) -> list[SegmentId]:
        return sorted(self.mlp_records)

    def list_attention_records(self) -> list[AttentionSegmentExecutionRecord]:
        return [self.attention_records[key] for key in self.list_attention_segment_ids()]

    def list_mlp_records(self) -> list[MLPSegmentExecutionRecord]:
        return [self.mlp_records[key] for key in self.list_mlp_segment_ids()]

    def evict_layer(self, layer_id: int) -> None:
        """Free memory/disk for one layer's shared tensors.

        Execution records (attention_records, mlp_records) are intentionally
        kept — the backward engine needs them for input_reference and rng_state.
        Only the large shared tensors are evicted.
        """
        if self._backend is not None:
            self._backend.evict_layer(layer_id)
        else:
            prefix_dot = f"layer_{layer_id}."
            to_del = [k for k in self.shared_tensors if k.startswith(prefix_dot)]
            for k in to_del:
                del self.shared_tensors[k]

    def clear_execution_records(self) -> None:
        self.attention_records.clear()
        self.mlp_records.clear()
        self.attention_composition_records.clear()
        self.mlp_composition_records.clear()
        self.residual_composition_records.clear()
        self.shared_tensors.clear()
        self.metadata.clear()
        if self._backend is not None:
            self._backend.clear()

    def clear_gradient_records(self) -> None:
        self.attention_gradient_records.clear()
        self.mlp_gradient_records.clear()

    def clear(self) -> None:
        self.clear_execution_records()
        self.clear_gradient_records()

    @property
    def num_execution_records(self) -> int:
        return len(self.attention_records) + len(self.mlp_records)

    @property
    def num_gradient_records(self) -> int:
        return len(self.attention_gradient_records) + len(self.mlp_gradient_records)

    def is_empty(self) -> bool:
        return (
            not self.attention_records
            and not self.mlp_records
            and not self.attention_gradient_records
            and not self.mlp_gradient_records
            and not self.attention_composition_records
            and not self.mlp_composition_records
            and not self.residual_composition_records
            and not self.shared_tensors
            and not self.metadata
            and self._backend is None
        )
