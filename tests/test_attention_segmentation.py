"""Tests for Phase 3 attention segmentation."""

from __future__ import annotations

import pytest
import torch

from sequential_segmented_llm_training_inference.segments.attention_segment import (
    AttentionHeadSegment,
    attention_segment_head_range,
    attention_segment_output_dim,
    validate_attention_segmentation,
)
from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId


def test_validate_attention_segmentation_accepts_valid_config() -> None:
    validate_attention_segmentation(d_model=32, n_heads=8, attention_segments=4)


@pytest.mark.parametrize(
    "d_model,n_heads,attention_segments",
    [
        (30, 8, 4),
        (32, 7, 1),
        (32, 8, 1),
        (32, 8, 3),
    ],
)
def test_validate_attention_segmentation_rejects_invalid_config(
    d_model: int,
    n_heads: int,
    attention_segments: int,
) -> None:
    with pytest.raises(ValueError):
        validate_attention_segmentation(
            d_model=d_model,
            n_heads=n_heads,
            attention_segments=attention_segments,
        )


def test_attention_segment_head_range_is_deterministic() -> None:
    assert attention_segment_head_range(n_heads=8, attention_segments=4, segment_id=0) == (0, 2)
    assert attention_segment_head_range(n_heads=8, attention_segments=4, segment_id=1) == (2, 4)
    assert attention_segment_head_range(n_heads=8, attention_segments=4, segment_id=2) == (4, 6)
    assert attention_segment_head_range(n_heads=8, attention_segments=4, segment_id=3) == (6, 8)


def test_attention_segments_cover_all_heads_once() -> None:
    owned_heads: list[int] = []
    for segment_id in range(4):
        start, end = attention_segment_head_range(
            n_heads=8,
            attention_segments=4,
            segment_id=segment_id,
        )
        owned_heads.extend(range(start, end))

    assert owned_heads == list(range(8))
    assert len(set(owned_heads)) == 8


def test_attention_segment_output_dim() -> None:
    assert attention_segment_output_dim(d_model=32, n_heads=8, attention_segments=4) == 8


def test_attention_segment_metadata() -> None:
    segment = AttentionHeadSegment(
        segment_id=SegmentId(layer_id=2, segment_type="attention", segment_id=1),
        d_model=32,
        n_heads=8,
        attention_segments=4,
        dropout=0.0,
    )

    metadata = segment.metadata
    assert metadata.layer_id == 2
    assert metadata.segment_id == 1
    assert metadata.start_head == 2
    assert metadata.end_head == 4
    assert metadata.owned_heads == (2, 3)
    assert metadata.output_dim == 8
    assert metadata.to_dict()["output_dim"] == 8


def test_attention_segment_requires_attention_segment_id() -> None:
    with pytest.raises(ValueError):
        AttentionHeadSegment(
            segment_id=SegmentId(layer_id=0, segment_type="mlp", segment_id=0),
            d_model=32,
            n_heads=8,
            attention_segments=4,
        )


def test_attention_segment_forward_shape() -> None:
    torch.manual_seed(123)
    segment = AttentionHeadSegment(
        segment_id=SegmentId(layer_id=0, segment_type="attention", segment_id=0),
        d_model=32,
        n_heads=8,
        attention_segments=4,
        dropout=0.0,
    )
    hidden_states = torch.randn(2, 5, 32)
    output = segment(hidden_states)
    assert output.shape == (2, 5, 8)


def test_concatenated_attention_outputs_restore_d_model_dimension() -> None:
    torch.manual_seed(123)
    hidden_states = torch.randn(2, 5, 32)
    outputs = []

    for segment_index in range(4):
        segment = AttentionHeadSegment(
            segment_id=SegmentId(layer_id=0, segment_type="attention", segment_id=segment_index),
            d_model=32,
            n_heads=8,
            attention_segments=4,
            dropout=0.0,
        )
        outputs.append(segment(hidden_states))

    concat = torch.cat(outputs, dim=-1)
    assert concat.shape == (2, 5, 32)


def test_padding_mask_changes_output_for_masked_token() -> None:
    torch.manual_seed(123)
    segment = AttentionHeadSegment(
        segment_id=SegmentId(layer_id=0, segment_type="attention", segment_id=0),
        d_model=16,
        n_heads=4,
        attention_segments=2,
        dropout=0.0,
    )
    segment.eval()
    hidden_states = torch.randn(1, 4, 16)

    unmasked = segment(hidden_states, attention_mask=torch.tensor([[1, 1, 1, 1]]), causal=False)
    masked = segment(hidden_states, attention_mask=torch.tensor([[1, 1, 1, 0]]), causal=False)

    assert not torch.allclose(unmasked, masked)


def test_causal_mask_prevents_future_token_attention() -> None:
    torch.manual_seed(123)
    segment = AttentionHeadSegment(
        segment_id=SegmentId(layer_id=0, segment_type="attention", segment_id=0),
        d_model=16,
        n_heads=4,
        attention_segments=2,
        dropout=0.0,
    )
    segment.eval()

    base = torch.randn(1, 4, 16)
    changed_future = base.clone()
    changed_future[:, 3, :] += 1000.0

    out_base = segment(base, causal=True)
    out_changed = segment(changed_future, causal=True)

    # The first position cannot attend to future positions, so it is unchanged.
    assert torch.allclose(out_base[:, 0, :], out_changed[:, 0, :], atol=1e-5, rtol=1e-5)
    # The final position can attend to itself, so it should change.
    assert not torch.allclose(out_base[:, 3, :], out_changed[:, 3, :])


def test_invalid_hidden_state_shape_raises() -> None:
    segment = AttentionHeadSegment(
        segment_id=SegmentId(layer_id=0, segment_type="attention", segment_id=0),
        d_model=16,
        n_heads=4,
        attention_segments=2,
        dropout=0.0,
    )
    with pytest.raises(ValueError):
        segment(torch.randn(2, 16))


def test_invalid_hidden_dimension_raises() -> None:
    segment = AttentionHeadSegment(
        segment_id=SegmentId(layer_id=0, segment_type="attention", segment_id=0),
        d_model=16,
        n_heads=4,
        attention_segments=2,
        dropout=0.0,
    )
    with pytest.raises(ValueError):
        segment(torch.randn(2, 5, 15))
