"""Segment modules and stable logical segment identities."""

from sequential_segmented_llm_training_inference.segments.attention_segment import (
    AttentionHeadSegment,
    AttentionSegmentMetadata,
    attention_segment_head_range,
    attention_segment_output_dim,
    validate_attention_segmentation,
)
from sequential_segmented_llm_training_inference.segments.mlp_segment import (
    MLPHiddenSegment,
    MLPSegmentMetadata,
    build_activation,
    mlp_chunk_hidden_size,
    mlp_chunk_range,
    validate_mlp_segmentation,
)
from sequential_segmented_llm_training_inference.segments.segment_factory import (
    SegmentCollection,
    SegmentFactory,
    build_all_segment_ids,
    build_attention_segment_ids,
    build_mlp_segment_ids,
)
from sequential_segmented_llm_training_inference.segments.segment_ids import (
    GLOBAL_SEGMENT_TYPES,
    SegmentId,
    SegmentParameterKey,
    SegmentType,
    VALID_SEGMENT_TYPES,
)
from sequential_segmented_llm_training_inference.segments.global_segments import (
    AttentionOutputProjSegmentModule,
    EmbeddingSegmentModule,
    OutputHeadSegmentModule,
)

__all__ = [
    "AttentionHeadSegment",
    "AttentionOutputProjSegmentModule",
    "AttentionSegmentMetadata",
    "EmbeddingSegmentModule",
    "GLOBAL_SEGMENT_TYPES",
    "MLPHiddenSegment",
    "MLPSegmentMetadata",
    "OutputHeadSegmentModule",
    "SegmentCollection",
    "SegmentFactory",
    "SegmentId",
    "SegmentParameterKey",
    "SegmentType",
    "VALID_SEGMENT_TYPES",
    "attention_segment_head_range",
    "attention_segment_output_dim",
    "build_activation",
    "build_all_segment_ids",
    "build_attention_segment_ids",
    "build_mlp_segment_ids",
    "mlp_chunk_hidden_size",
    "mlp_chunk_range",
    "validate_attention_segmentation",
    "validate_mlp_segmentation",
]
