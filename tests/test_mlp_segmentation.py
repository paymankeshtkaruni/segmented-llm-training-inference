"""Tests for Phase 4 MLP segmentation."""

from __future__ import annotations

import pytest
import torch

from sequential_segmented_llm_training_inference.segments.mlp_segment import (
    MLPHiddenSegment,
    build_activation,
    mlp_chunk_hidden_size,
    mlp_chunk_range,
    validate_mlp_segmentation,
)
from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId


def test_validate_mlp_segmentation_accepts_valid_config() -> None:
    validate_mlp_segmentation(d_model=32, d_ff=128, mlp_chunks=4)


@pytest.mark.parametrize(
    "d_model,d_ff,mlp_chunks",
    [
        (0, 128, 4),
        (32, 0, 4),
        (32, 128, 1),
        (32, 130, 4),
    ],
)
def test_validate_mlp_segmentation_rejects_invalid_config(
    d_model: int,
    d_ff: int,
    mlp_chunks: int,
) -> None:
    with pytest.raises(ValueError):
        validate_mlp_segmentation(d_model=d_model, d_ff=d_ff, mlp_chunks=mlp_chunks)


def test_mlp_chunk_range_is_deterministic() -> None:
    assert mlp_chunk_range(d_ff=128, mlp_chunks=4, segment_id=0) == (0, 32)
    assert mlp_chunk_range(d_ff=128, mlp_chunks=4, segment_id=1) == (32, 64)
    assert mlp_chunk_range(d_ff=128, mlp_chunks=4, segment_id=2) == (64, 96)
    assert mlp_chunk_range(d_ff=128, mlp_chunks=4, segment_id=3) == (96, 128)


def test_mlp_chunks_cover_d_ff_once() -> None:
    owned_units: list[int] = []
    for segment_id in range(4):
        start, end = mlp_chunk_range(d_ff=128, mlp_chunks=4, segment_id=segment_id)
        owned_units.extend(range(start, end))

    assert owned_units == list(range(128))
    assert len(set(owned_units)) == 128


def test_mlp_chunk_hidden_size() -> None:
    assert mlp_chunk_hidden_size(d_ff=128, mlp_chunks=4) == 32


def test_mlp_segment_metadata() -> None:
    segment = MLPHiddenSegment(
        segment_id=SegmentId(layer_id=2, segment_type="mlp", segment_id=1),
        d_model=32,
        d_ff=128,
        mlp_chunks=4,
        activation="gelu",
    )

    metadata = segment.metadata
    assert metadata.layer_id == 2
    assert metadata.segment_id == 1
    assert metadata.start_hidden == 32
    assert metadata.end_hidden == 64
    assert metadata.chunk_hidden_size == 32
    assert metadata.owned_hidden_units == tuple(range(32, 64))
    assert metadata.output_bias_inside_segment is False
    assert metadata.to_dict()["chunk_hidden_size"] == 32


def test_mlp_segment_requires_mlp_segment_id() -> None:
    with pytest.raises(ValueError):
        MLPHiddenSegment(
            segment_id=SegmentId(layer_id=0, segment_type="attention", segment_id=0),
            d_model=32,
            d_ff=128,
            mlp_chunks=4,
        )


def test_mlp_segment_rejects_out_of_range_segment_id() -> None:
    with pytest.raises(ValueError):
        MLPHiddenSegment(
            segment_id=SegmentId(layer_id=0, segment_type="mlp", segment_id=4),
            d_model=32,
            d_ff=128,
            mlp_chunks=4,
        )


def test_mlp_segment_forward_shape() -> None:
    torch.manual_seed(123)
    segment = MLPHiddenSegment(
        segment_id=SegmentId(layer_id=0, segment_type="mlp", segment_id=0),
        d_model=32,
        d_ff=128,
        mlp_chunks=4,
        activation="gelu",
    )
    hidden_states = torch.randn(2, 5, 32)
    output = segment(hidden_states)
    assert output.shape == (2, 5, 32)


def test_summed_mlp_outputs_restore_d_model_dimension() -> None:
    torch.manual_seed(123)
    hidden_states = torch.randn(2, 5, 32)
    outputs = []

    for segment_index in range(4):
        segment = MLPHiddenSegment(
            segment_id=SegmentId(layer_id=0, segment_type="mlp", segment_id=segment_index),
            d_model=32,
            d_ff=128,
            mlp_chunks=4,
            activation="gelu",
        )
        outputs.append(segment(hidden_states))

    summed = torch.stack(outputs, dim=0).sum(dim=0)
    assert summed.shape == (2, 5, 32)


def test_shared_output_bias_is_not_duplicated_by_default() -> None:
    segment = MLPHiddenSegment(
        segment_id=SegmentId(layer_id=0, segment_type="mlp", segment_id=0),
        d_model=32,
        d_ff=128,
        mlp_chunks=4,
    )
    assert segment.fc2.bias is None
    assert segment.metadata.output_bias_inside_segment is False


def test_output_bias_can_be_enabled_explicitly() -> None:
    segment = MLPHiddenSegment(
        segment_id=SegmentId(layer_id=0, segment_type="mlp", segment_id=0),
        d_model=32,
        d_ff=128,
        mlp_chunks=4,
        output_bias=True,
    )
    assert segment.fc2.bias is not None
    assert segment.metadata.output_bias_inside_segment is True


@pytest.mark.parametrize("activation", ["gelu", "relu", "silu", "tanh"])
def test_supported_activations_forward(activation: str) -> None:
    segment = MLPHiddenSegment(
        segment_id=SegmentId(layer_id=0, segment_type="mlp", segment_id=0),
        d_model=16,
        d_ff=64,
        mlp_chunks=4,
        activation=activation,  # type: ignore[arg-type]
    )
    output = segment(torch.randn(1, 3, 16))
    assert output.shape == (1, 3, 16)


def test_invalid_activation_raises() -> None:
    with pytest.raises(ValueError):
        build_activation("invalid")
    with pytest.raises(ValueError):
        MLPHiddenSegment(
            segment_id=SegmentId(layer_id=0, segment_type="mlp", segment_id=0),
            d_model=16,
            d_ff=64,
            mlp_chunks=4,
            activation="invalid",  # type: ignore[arg-type]
        )


def test_invalid_hidden_state_shape_raises() -> None:
    segment = MLPHiddenSegment(
        segment_id=SegmentId(layer_id=0, segment_type="mlp", segment_id=0),
        d_model=16,
        d_ff=64,
        mlp_chunks=4,
    )
    with pytest.raises(ValueError):
        segment(torch.randn(1, 16))


def test_invalid_hidden_state_last_dimension_raises() -> None:
    segment = MLPHiddenSegment(
        segment_id=SegmentId(layer_id=0, segment_type="mlp", segment_id=0),
        d_model=16,
        d_ff=64,
        mlp_chunks=4,
    )
    with pytest.raises(ValueError):
        segment(torch.randn(1, 3, 15))


def test_mlp_segment_backward_produces_gradients() -> None:
    torch.manual_seed(123)
    segment = MLPHiddenSegment(
        segment_id=SegmentId(layer_id=0, segment_type="mlp", segment_id=0),
        d_model=16,
        d_ff=64,
        mlp_chunks=4,
    )
    hidden_states = torch.randn(2, 4, 16, requires_grad=True)
    loss = segment(hidden_states).pow(2).mean()
    loss.backward()

    assert hidden_states.grad is not None
    assert segment.fc1.weight.grad is not None
    assert segment.fc2.weight.grad is not None
