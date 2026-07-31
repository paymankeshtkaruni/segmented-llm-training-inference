"""MLP segment implemented by feed-forward hidden-dimension chunks.

Phase 4 scope:
- Implement one MLP segment that owns a deterministic chunk of the feed-forward
  hidden dimension.
- All MLP segments in a layer receive the same normalized MLP input.
- Each segment returns an output in d_model space.
- Segment outputs are summed outside the segment.
- The shared MLP output bias is intentionally outside the segments by default
  so it is added once after summation, not duplicated per chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch
from torch import Tensor, nn

from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId


ActivationName = Literal["gelu", "relu", "silu", "tanh"]
VALID_ACTIVATIONS: tuple[str, ...] = ("gelu", "relu", "silu", "tanh")


def validate_mlp_segmentation(
    *,
    d_model: int,
    d_ff: int,
    mlp_chunks: int,
) -> None:
    """Validate MLP hidden-dimension chunking constraints."""

    if not isinstance(d_model, int):
        raise TypeError(f"d_model must be an int, got {type(d_model).__name__}.")
    if not isinstance(d_ff, int):
        raise TypeError(f"d_ff must be an int, got {type(d_ff).__name__}.")
    if not isinstance(mlp_chunks, int):
        raise TypeError(f"mlp_chunks must be an int, got {type(mlp_chunks).__name__}.")
    if d_model <= 0:
        raise ValueError(f"d_model must be > 0, got {d_model}.")
    if d_ff <= 0:
        raise ValueError(f"d_ff must be > 0, got {d_ff}.")
    if mlp_chunks <= 1:
        raise ValueError("mlp_chunks must be > 1 for the segmented MLP mode.")
    if d_ff % mlp_chunks != 0:
        raise ValueError(
            f"d_ff must be divisible by mlp_chunks, got d_ff={d_ff}, "
            f"mlp_chunks={mlp_chunks}."
        )


def mlp_chunk_range(
    *,
    d_ff: int,
    mlp_chunks: int,
    segment_id: int,
) -> tuple[int, int]:
    """Return the inclusive/exclusive hidden-unit range owned by one MLP chunk."""

    if not isinstance(segment_id, int):
        raise TypeError(f"segment_id must be an int, got {type(segment_id).__name__}.")
    if d_ff <= 0:
        raise ValueError(f"d_ff must be > 0, got {d_ff}.")
    if mlp_chunks <= 1:
        raise ValueError("mlp_chunks must be > 1 for the segmented MLP mode.")
    if d_ff % mlp_chunks != 0:
        raise ValueError(
            f"d_ff must be divisible by mlp_chunks, got d_ff={d_ff}, "
            f"mlp_chunks={mlp_chunks}."
        )
    if segment_id < 0 or segment_id >= mlp_chunks:
        raise ValueError(
            f"segment_id must be in [0, {mlp_chunks - 1}], got {segment_id}."
        )

    chunk_size = d_ff // mlp_chunks
    start_hidden = segment_id * chunk_size
    end_hidden = start_hidden + chunk_size
    return start_hidden, end_hidden


def mlp_chunk_hidden_size(*, d_ff: int, mlp_chunks: int) -> int:
    """Return the hidden dimension owned by each MLP segment."""

    validate_mlp_segmentation(d_model=1, d_ff=d_ff, mlp_chunks=mlp_chunks)
    return d_ff // mlp_chunks


def build_activation(name: str) -> nn.Module:
    """Create the activation module used inside each MLP segment."""

    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()

    allowed = ", ".join(VALID_ACTIVATIONS)
    raise ValueError(f"activation must be one of {{{allowed}}}, got {name!r}.")


@dataclass(frozen=True, slots=True)
class MLPSegmentMetadata:
    """Metadata describing deterministic hidden-dimension ownership."""

    layer_id: int
    segment_id: int
    d_ff: int
    mlp_chunks: int
    start_hidden: int
    end_hidden: int
    chunk_hidden_size: int
    d_model: int
    activation: str
    output_bias_inside_segment: bool

    @property
    def owned_hidden_units(self) -> tuple[int, ...]:
        """Exact hidden-unit indices owned by this segment."""

        return tuple(range(self.start_hidden, self.end_hidden))

    def to_dict(self) -> dict[str, Any]:
        """Return YAML/JSON-safe metadata."""

        return {
            "layer_id": self.layer_id,
            "segment_id": self.segment_id,
            "d_ff": self.d_ff,
            "mlp_chunks": self.mlp_chunks,
            "start_hidden": self.start_hidden,
            "end_hidden": self.end_hidden,
            "chunk_hidden_size": self.chunk_hidden_size,
            "d_model": self.d_model,
            "activation": self.activation,
            "output_bias_inside_segment": self.output_bias_inside_segment,
        }


class MLPHiddenSegment(nn.Module):
    """One MLP feed-forward hidden-dimension chunk.

    This module owns one deterministic slice of the MLP hidden dimension. It
    maps the shared normalized MLP input into its own hidden chunk and then back
    into d_model space. Outputs from all chunks must be summed outside this
    module. By default, the second projection has no bias because a shared MLP
    output bias must be added once after summing all chunk outputs.

    Args:
        segment_id: Logical segment id. ``segment_type`` must be ``"mlp"``.
        d_model: Transformer hidden size.
        d_ff: Full feed-forward hidden dimension.
        mlp_chunks: Number of MLP hidden-dimension chunks.
        activation: Activation name: ``gelu``, ``relu``, ``silu``, or ``tanh``.
        input_bias: Whether the first projection uses bias.
        output_bias: Whether the second projection has a per-segment bias.
            Defaults to ``False`` to avoid duplicating the shared MLP output bias.
    """

    def __init__(
        self,
        *,
        segment_id: SegmentId,
        d_model: int,
        d_ff: int,
        mlp_chunks: int,
        activation: ActivationName = "gelu",
        input_bias: bool = True,
        output_bias: bool = False,
    ) -> None:
        super().__init__()

        if segment_id.segment_type != "mlp":
            raise ValueError("MLPHiddenSegment requires segment_id.segment_type == 'mlp'.")
        validate_mlp_segmentation(d_model=d_model, d_ff=d_ff, mlp_chunks=mlp_chunks)
        if segment_id.segment_id >= mlp_chunks:
            raise ValueError(
                f"segment_id.segment_id must be < mlp_chunks, got "
                f"{segment_id.segment_id} and {mlp_chunks}."
            )
        if activation not in VALID_ACTIVATIONS:
            allowed = ", ".join(VALID_ACTIVATIONS)
            raise ValueError(f"activation must be one of {{{allowed}}}, got {activation!r}.")

        self.segment_id = segment_id
        self.d_model = d_model
        self.d_ff = d_ff
        self.mlp_chunks = mlp_chunks
        self.chunk_hidden_size = d_ff // mlp_chunks
        self.start_hidden, self.end_hidden = mlp_chunk_range(
            d_ff=d_ff,
            mlp_chunks=mlp_chunks,
            segment_id=segment_id.segment_id,
        )
        self.activation_name = activation
        self.output_bias_inside_segment = output_bias

        self.fc1 = nn.Linear(d_model, self.chunk_hidden_size, bias=input_bias)
        self.activation = build_activation(activation)
        self.fc2 = nn.Linear(self.chunk_hidden_size, d_model, bias=output_bias)

    @property
    def metadata(self) -> MLPSegmentMetadata:
        """Return deterministic metadata for this MLP segment."""

        return MLPSegmentMetadata(
            layer_id=self.segment_id.layer_id,
            segment_id=self.segment_id.segment_id,
            d_ff=self.d_ff,
            mlp_chunks=self.mlp_chunks,
            start_hidden=self.start_hidden,
            end_hidden=self.end_hidden,
            chunk_hidden_size=self.chunk_hidden_size,
            d_model=self.d_model,
            activation=self.activation_name,
            output_bias_inside_segment=self.output_bias_inside_segment,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Run this MLP segment.

        Args:
            hidden_states: Shared normalized MLP input with shape
                ``[batch, seq_len, d_model]``.

        Returns:
            Tensor with shape ``[batch, seq_len, d_model]``.
        """

        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, seq_len, d_model], got "
                f"{tuple(hidden_states.shape)}."
            )
        if hidden_states.shape[-1] != self.d_model:
            raise ValueError(
                f"hidden_states last dimension must be d_model={self.d_model}, "
                f"got {hidden_states.shape[-1]}."
            )

        hidden = self.fc1(hidden_states)
        hidden = self.activation(hidden)
        return self.fc2(hidden)
