"""Factory for creating all attention and MLP segments.

Phase 6 scope:
- Create deterministic attention/MLP segment IDs for every layer.
- Create all attention-head segments and MLP hidden-dimension segments.
- Verify complete non-overlapping coverage through each segment metadata object.
- Do not implement storage, loading, forward engine, or backward engine here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from torch import nn

from sequential_segmented_llm_training_inference.config.model_config import ModelConfig
from sequential_segmented_llm_training_inference.config.segmentation_config import (
    SegmentationConfig,
)
from sequential_segmented_llm_training_inference.segments.attention_segment import (
    AttentionHeadSegment,
)
from sequential_segmented_llm_training_inference.segments.mlp_segment import (
    ActivationName,
    MLPHiddenSegment,
)
from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId


SegmentModule = AttentionHeadSegment | MLPHiddenSegment


def build_attention_segment_ids(
    *,
    n_layers: int,
    attention_segments: int,
) -> tuple[SegmentId, ...]:
    """Create deterministic attention segment IDs for all layers."""

    _validate_positive_int("n_layers", n_layers)
    _validate_segment_count("attention_segments", attention_segments)

    return tuple(
        SegmentId(layer_id=layer_id, segment_type="attention", segment_id=segment_id)
        for layer_id in range(n_layers)
        for segment_id in range(attention_segments)
    )


def build_mlp_segment_ids(
    *,
    n_layers: int,
    mlp_chunks: int,
) -> tuple[SegmentId, ...]:
    """Create deterministic MLP segment IDs for all layers."""

    _validate_positive_int("n_layers", n_layers)
    _validate_segment_count("mlp_chunks", mlp_chunks)

    return tuple(
        SegmentId(layer_id=layer_id, segment_type="mlp", segment_id=segment_id)
        for layer_id in range(n_layers)
        for segment_id in range(mlp_chunks)
    )


def build_all_segment_ids(
    *,
    n_layers: int,
    attention_segments: int,
    mlp_chunks: int,
) -> tuple[SegmentId, ...]:
    """Create deterministic attention IDs followed by MLP IDs for all layers."""

    return (
        build_attention_segment_ids(
            n_layers=n_layers,
            attention_segments=attention_segments,
        )
        + build_mlp_segment_ids(n_layers=n_layers, mlp_chunks=mlp_chunks)
    )


@dataclass(slots=True)
class SegmentCollection:
    """Container for all created true segment modules.

    The collection stores modules by stable logical ``SegmentId``. It is a
    construction-time object only. Later phases will move segment state into
    storage backends and enforce strict one-segment loading at runtime.
    """

    attention_segments: dict[SegmentId, AttentionHeadSegment]
    mlp_segments: dict[SegmentId, MLPHiddenSegment]

    def __post_init__(self) -> None:
        for segment_id in self.attention_segments:
            if segment_id.segment_type != "attention":
                raise ValueError(
                    "attention_segments keys must have segment_type='attention'."
                )
        for segment_id in self.mlp_segments:
            if segment_id.segment_type != "mlp":
                raise ValueError("mlp_segments keys must have segment_type='mlp'.")

    @property
    def num_attention_segments(self) -> int:
        return len(self.attention_segments)

    @property
    def num_mlp_segments(self) -> int:
        return len(self.mlp_segments)

    @property
    def num_segments(self) -> int:
        return self.num_attention_segments + self.num_mlp_segments

    def attention_segment_ids(self, layer_id: int | None = None) -> tuple[SegmentId, ...]:
        ids = tuple(sorted(self.attention_segments))
        if layer_id is None:
            return ids
        return tuple(segment_id for segment_id in ids if segment_id.layer_id == layer_id)

    def mlp_segment_ids(self, layer_id: int | None = None) -> tuple[SegmentId, ...]:
        ids = tuple(sorted(self.mlp_segments))
        if layer_id is None:
            return ids
        return tuple(segment_id for segment_id in ids if segment_id.layer_id == layer_id)

    def all_segment_ids(self) -> tuple[SegmentId, ...]:
        """Return all segment IDs in deterministic order."""

        return self.attention_segment_ids() + self.mlp_segment_ids()

    def get_attention_segment(self, segment_id: SegmentId) -> AttentionHeadSegment:
        return self.attention_segments[segment_id]

    def get_mlp_segment(self, segment_id: SegmentId) -> MLPHiddenSegment:
        return self.mlp_segments[segment_id]

    def get_segment(self, segment_id: SegmentId) -> SegmentModule:
        if segment_id.segment_type == "attention":
            return self.get_attention_segment(segment_id)
        return self.get_mlp_segment(segment_id)

    def iter_segments(self) -> Iterator[tuple[SegmentId, SegmentModule]]:
        for segment_id in self.all_segment_ids():
            yield segment_id, self.get_segment(segment_id)

    def as_module_dict(self) -> nn.ModuleDict:
        """Return a ModuleDict with deterministic string keys.

        This is useful for debug/test registration. It is not the runtime
        storage backend and does not alter the strict single-segment design.
        """

        return nn.ModuleDict(
            (segment_id.to_path_name(), module)
            for segment_id, module in self.iter_segments()
        )

    def metadata_dict(self) -> dict[str, list[dict[str, object]]]:
        """Return YAML/JSON-safe metadata for every created segment."""

        return {
            "attention_segments": [
                segment.metadata.to_dict()
                for _, segment in sorted(self.attention_segments.items())
            ],
            "mlp_segments": [
                segment.metadata.to_dict()
                for _, segment in sorted(self.mlp_segments.items())
            ],
        }

    def validate_complete_coverage(
        self,
        *,
        model_config: ModelConfig,
        segmentation_config: SegmentationConfig,
    ) -> None:
        """Validate that created segments fully cover heads and MLP units."""

        expected_attention_ids = build_attention_segment_ids(
            n_layers=model_config.n_layers,
            attention_segments=segmentation_config.attention_segments,
        )
        expected_mlp_ids = build_mlp_segment_ids(
            n_layers=model_config.n_layers,
            mlp_chunks=segmentation_config.mlp_chunks,
        )
        if self.attention_segment_ids() != expected_attention_ids:
            raise ValueError("Attention segment IDs do not match expected coverage.")
        if self.mlp_segment_ids() != expected_mlp_ids:
            raise ValueError("MLP segment IDs do not match expected coverage.")

        for layer_id in range(model_config.n_layers):
            owned_heads: list[int] = []
            for segment_id in self.attention_segment_ids(layer_id=layer_id):
                owned_heads.extend(self.attention_segments[segment_id].metadata.owned_heads)
            if owned_heads != list(range(model_config.n_heads)):
                raise ValueError(
                    f"Attention heads are not fully covered in layer {layer_id}."
                )

            owned_hidden_units: list[int] = []
            for segment_id in self.mlp_segment_ids(layer_id=layer_id):
                owned_hidden_units.extend(self.mlp_segments[segment_id].metadata.owned_hidden_units)
            if owned_hidden_units != list(range(model_config.d_ff)):
                raise ValueError(
                    f"MLP hidden units are not fully covered in layer {layer_id}."
                )


@dataclass(frozen=True, slots=True)
class SegmentFactory:
    """Factory for creating all true attention and MLP segments."""

    model_config: ModelConfig
    segmentation_config: SegmentationConfig
    attention_bias: bool = True
    attention_dropout: float | None = None
    mlp_activation: ActivationName = "gelu"
    mlp_input_bias: bool = True
    mlp_output_bias: bool = False

    def __post_init__(self) -> None:
        self.model_config.validate()
        self.segmentation_config.validate_against_model(self.model_config)
        if self.attention_dropout is not None:
            if not 0.0 <= self.attention_dropout < 1.0:
                raise ValueError(
                    "attention_dropout must be in [0, 1), got "
                    f"{self.attention_dropout}."
                )

    @property
    def effective_attention_dropout(self) -> float:
        if self.attention_dropout is None:
            return self.model_config.dropout
        return self.attention_dropout

    def create_attention_segment(
        self,
        *,
        layer_id: int,
        segment_index: int,
    ) -> AttentionHeadSegment:
        self._validate_layer_id(layer_id)
        self._validate_attention_segment_index(segment_index)
        return AttentionHeadSegment(
            segment_id=SegmentId(
                layer_id=layer_id,
                segment_type="attention",
                segment_id=segment_index,
            ),
            d_model=self.model_config.d_model,
            n_heads=self.model_config.n_heads,
            attention_segments=self.segmentation_config.attention_segments,
            dropout=self.effective_attention_dropout,
            bias=self.attention_bias,
        )

    def create_mlp_segment(
        self,
        *,
        layer_id: int,
        segment_index: int,
    ) -> MLPHiddenSegment:
        self._validate_layer_id(layer_id)
        self._validate_mlp_segment_index(segment_index)
        return MLPHiddenSegment(
            segment_id=SegmentId(
                layer_id=layer_id,
                segment_type="mlp",
                segment_id=segment_index,
            ),
            d_model=self.model_config.d_model,
            d_ff=self.model_config.d_ff,
            mlp_chunks=self.segmentation_config.mlp_chunks,
            activation=self.mlp_activation,
            input_bias=self.mlp_input_bias,
            output_bias=self.mlp_output_bias,
        )

    def create_all_segments(self) -> SegmentCollection:
        attention_segments: dict[SegmentId, AttentionHeadSegment] = {}
        mlp_segments: dict[SegmentId, MLPHiddenSegment] = {}

        for layer_id in range(self.model_config.n_layers):
            for segment_index in range(self.segmentation_config.attention_segments):
                segment = self.create_attention_segment(
                    layer_id=layer_id,
                    segment_index=segment_index,
                )
                attention_segments[segment.segment_id] = segment

            for segment_index in range(self.segmentation_config.mlp_chunks):
                segment = self.create_mlp_segment(
                    layer_id=layer_id,
                    segment_index=segment_index,
                )
                mlp_segments[segment.segment_id] = segment

        collection = SegmentCollection(
            attention_segments=attention_segments,
            mlp_segments=mlp_segments,
        )
        collection.validate_complete_coverage(
            model_config=self.model_config,
            segmentation_config=self.segmentation_config,
        )
        return collection

    def expected_segment_ids(self) -> tuple[SegmentId, ...]:
        return build_all_segment_ids(
            n_layers=self.model_config.n_layers,
            attention_segments=self.segmentation_config.attention_segments,
            mlp_chunks=self.segmentation_config.mlp_chunks,
        )

    def _validate_layer_id(self, layer_id: int) -> None:
        if not isinstance(layer_id, int):
            raise TypeError(f"layer_id must be an int, got {type(layer_id).__name__}.")
        if layer_id < 0 or layer_id >= self.model_config.n_layers:
            raise ValueError(
                f"layer_id must be in [0, {self.model_config.n_layers - 1}], "
                f"got {layer_id}."
            )

    def _validate_attention_segment_index(self, segment_index: int) -> None:
        if not isinstance(segment_index, int):
            raise TypeError(
                f"segment_index must be an int, got {type(segment_index).__name__}."
            )
        if segment_index < 0 or segment_index >= self.segmentation_config.attention_segments:
            raise ValueError(
                "attention segment_index must be in "
                f"[0, {self.segmentation_config.attention_segments - 1}], "
                f"got {segment_index}."
            )

    def _validate_mlp_segment_index(self, segment_index: int) -> None:
        if not isinstance(segment_index, int):
            raise TypeError(
                f"segment_index must be an int, got {type(segment_index).__name__}."
            )
        if segment_index < 0 or segment_index >= self.segmentation_config.mlp_chunks:
            raise ValueError(
                "MLP segment_index must be in "
                f"[0, {self.segmentation_config.mlp_chunks - 1}], "
                f"got {segment_index}."
            )


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")


def _validate_segment_count(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
    if value <= 1:
        raise ValueError(f"{name} must be > 1, got {value}.")
