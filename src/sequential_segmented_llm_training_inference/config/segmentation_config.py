"""Segmentation configuration for attention and MLP segments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .model_config import ModelConfig


@dataclass(frozen=True)
class SegmentationConfig:
    """Defines strict segmented attention and segmented MLP configuration."""

    attention_segments: int = 4
    mlp_chunks: int = 4
    attention_segmentation_axis: str = "head_groups"
    mlp_segmentation_axis: str = "feedforward_hidden_dimension"
    strict_segmented_attention: bool = True
    strict_segmented_mlp: bool = True
    embedding_segments: int = 1
    output_head_segments: int = 1

    def __post_init__(self) -> None:
        self.validate_basic()

    def validate_basic(self) -> None:
        if self.attention_segments <= 1:
            raise ValueError("attention_segments must be > 1. Full-attention mode is not supported.")
        if self.mlp_chunks <= 1:
            raise ValueError("mlp_chunks must be > 1. Full-MLP mode is not supported.")
        if self.attention_segmentation_axis != "head_groups":
            raise ValueError("attention_segmentation_axis must be 'head_groups'.")
        if self.mlp_segmentation_axis != "feedforward_hidden_dimension":
            raise ValueError("mlp_segmentation_axis must be 'feedforward_hidden_dimension'.")
        if not self.strict_segmented_attention:
            raise ValueError("strict_segmented_attention must be true.")
        if not self.strict_segmented_mlp:
            raise ValueError("strict_segmented_mlp must be true.")
        if self.embedding_segments < 1:
            raise ValueError("embedding_segments must be >= 1.")
        if self.output_head_segments < 1:
            raise ValueError("output_head_segments must be >= 1.")

    def validate_against_model(self, model: ModelConfig) -> None:
        self.validate_basic()
        if model.n_heads % self.attention_segments != 0:
            raise ValueError("n_heads must be divisible by attention_segments.")
        if model.d_ff % self.mlp_chunks != 0:
            raise ValueError("d_ff must be divisible by mlp_chunks.")
        if self.embedding_segments > 1 and model.d_model % self.embedding_segments != 0:
            raise ValueError("d_model must be divisible by embedding_segments.")
        if self.output_head_segments > 1 and model.d_model % self.output_head_segments != 0:
            raise ValueError("d_model must be divisible by output_head_segments.")

    def heads_per_segment(self, model: ModelConfig) -> int:
        self.validate_against_model(model)
        return model.n_heads // self.attention_segments

    def mlp_chunk_size(self, model: ModelConfig) -> int:
        self.validate_against_model(model)
        return model.d_ff // self.mlp_chunks

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SegmentationConfig":
        return cls(**data)
