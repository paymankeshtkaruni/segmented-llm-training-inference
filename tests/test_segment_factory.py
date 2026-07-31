"""Tests for Phase 6 segment factory."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from sequential_segmented_llm_training_inference.config import (
    ModelConfig,
    SegmentationConfig,
)
from sequential_segmented_llm_training_inference.segments import (
    AttentionHeadSegment,
    MLPHiddenSegment,
    SegmentCollection,
    SegmentFactory,
    SegmentId,
    build_all_segment_ids,
    build_attention_segment_ids,
    build_mlp_segment_ids,
)


def model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=100,
        max_seq_len=16,
        n_layers=3,
        d_model=32,
        n_heads=8,
        d_ff=128,
        dropout=0.0,
    )


def segmentation_config() -> SegmentationConfig:
    return SegmentationConfig(attention_segments=4, mlp_chunks=4)


def test_build_attention_segment_ids_are_deterministic() -> None:
    ids = build_attention_segment_ids(n_layers=2, attention_segments=3)
    assert ids == (
        SegmentId(layer_id=0, segment_type="attention", segment_id=0),
        SegmentId(layer_id=0, segment_type="attention", segment_id=1),
        SegmentId(layer_id=0, segment_type="attention", segment_id=2),
        SegmentId(layer_id=1, segment_type="attention", segment_id=0),
        SegmentId(layer_id=1, segment_type="attention", segment_id=1),
        SegmentId(layer_id=1, segment_type="attention", segment_id=2),
    )


def test_build_mlp_segment_ids_are_deterministic() -> None:
    ids = build_mlp_segment_ids(n_layers=2, mlp_chunks=2)
    assert ids == (
        SegmentId(layer_id=0, segment_type="mlp", segment_id=0),
        SegmentId(layer_id=0, segment_type="mlp", segment_id=1),
        SegmentId(layer_id=1, segment_type="mlp", segment_id=0),
        SegmentId(layer_id=1, segment_type="mlp", segment_id=1),
    )


def test_build_all_segment_ids_attention_then_mlp() -> None:
    ids = build_all_segment_ids(n_layers=1, attention_segments=2, mlp_chunks=3)
    assert ids == (
        SegmentId(layer_id=0, segment_type="attention", segment_id=0),
        SegmentId(layer_id=0, segment_type="attention", segment_id=1),
        SegmentId(layer_id=0, segment_type="mlp", segment_id=0),
        SegmentId(layer_id=0, segment_type="mlp", segment_id=1),
        SegmentId(layer_id=0, segment_type="mlp", segment_id=2),
    )


def test_segment_factory_creates_expected_number_of_segments() -> None:
    factory = SegmentFactory(model_config(), segmentation_config())
    collection = factory.create_all_segments()

    assert collection.num_attention_segments == 3 * 4
    assert collection.num_mlp_segments == 3 * 4
    assert collection.num_segments == 24


def test_segment_factory_creates_expected_segment_types() -> None:
    collection = SegmentFactory(model_config(), segmentation_config()).create_all_segments()

    for segment_id in collection.attention_segment_ids():
        assert isinstance(collection.get_segment(segment_id), AttentionHeadSegment)
    for segment_id in collection.mlp_segment_ids():
        assert isinstance(collection.get_segment(segment_id), MLPHiddenSegment)


def test_expected_segment_ids_match_collection_ids() -> None:
    factory = SegmentFactory(model_config(), segmentation_config())
    collection = factory.create_all_segments()
    assert collection.all_segment_ids() == factory.expected_segment_ids()


def test_attention_segments_cover_all_heads_per_layer_once() -> None:
    cfg = model_config()
    collection = SegmentFactory(cfg, segmentation_config()).create_all_segments()

    for layer_id in range(cfg.n_layers):
        owned_heads: list[int] = []
        for segment_id in collection.attention_segment_ids(layer_id=layer_id):
            owned_heads.extend(collection.get_attention_segment(segment_id).metadata.owned_heads)
        assert owned_heads == list(range(cfg.n_heads))
        assert len(set(owned_heads)) == cfg.n_heads


def test_mlp_segments_cover_all_hidden_units_per_layer_once() -> None:
    cfg = model_config()
    collection = SegmentFactory(cfg, segmentation_config()).create_all_segments()

    for layer_id in range(cfg.n_layers):
        owned_units: list[int] = []
        for segment_id in collection.mlp_segment_ids(layer_id=layer_id):
            owned_units.extend(collection.get_mlp_segment(segment_id).metadata.owned_hidden_units)
        assert owned_units == list(range(cfg.d_ff))
        assert len(set(owned_units)) == cfg.d_ff


def test_segment_factory_validates_segmentation_against_model() -> None:
    cfg = ModelConfig(
        vocab_size=100,
        max_seq_len=16,
        n_layers=2,
        d_model=30,
        n_heads=6,
        d_ff=128,
        dropout=0.0,
    )
    bad_segmentation = SegmentationConfig(attention_segments=4, mlp_chunks=4)
    with pytest.raises(ValueError):
        SegmentFactory(cfg, bad_segmentation)


def test_segment_factory_rejects_invalid_layer_id() -> None:
    factory = SegmentFactory(model_config(), segmentation_config())
    with pytest.raises(ValueError):
        factory.create_attention_segment(layer_id=3, segment_index=0)
    with pytest.raises(ValueError):
        factory.create_mlp_segment(layer_id=-1, segment_index=0)


def test_segment_factory_rejects_invalid_segment_index() -> None:
    factory = SegmentFactory(model_config(), segmentation_config())
    with pytest.raises(ValueError):
        factory.create_attention_segment(layer_id=0, segment_index=4)
    with pytest.raises(ValueError):
        factory.create_mlp_segment(layer_id=0, segment_index=4)


def test_created_attention_segment_forward_shape() -> None:
    factory = SegmentFactory(model_config(), segmentation_config())
    segment = factory.create_attention_segment(layer_id=1, segment_index=2)
    output = segment(torch.randn(2, 5, 32), causal=True)
    assert output.shape == (2, 5, 8)
    assert segment.segment_id == SegmentId(
        layer_id=1,
        segment_type="attention",
        segment_id=2,
    )


def test_created_mlp_segment_forward_shape() -> None:
    factory = SegmentFactory(model_config(), segmentation_config())
    segment = factory.create_mlp_segment(layer_id=1, segment_index=2)
    output = segment(torch.randn(2, 5, 32))
    assert output.shape == (2, 5, 32)
    assert segment.segment_id == SegmentId(layer_id=1, segment_type="mlp", segment_id=2)


def test_collection_get_segment_raises_key_error_for_missing_segment() -> None:
    collection = SegmentFactory(model_config(), segmentation_config()).create_all_segments()
    with pytest.raises(KeyError):
        collection.get_segment(SegmentId(layer_id=100, segment_type="attention", segment_id=0))


def test_collection_metadata_dict_is_yaml_safe() -> None:
    collection = SegmentFactory(model_config(), segmentation_config()).create_all_segments()
    metadata = collection.metadata_dict()

    assert len(metadata["attention_segments"]) == 12
    assert len(metadata["mlp_segments"]) == 12
    assert metadata["attention_segments"][0]["layer_id"] == 0
    assert metadata["mlp_segments"][0]["layer_id"] == 0


def test_collection_as_module_dict_uses_deterministic_keys() -> None:
    collection = SegmentFactory(model_config(), segmentation_config()).create_all_segments()
    module_dict = collection.as_module_dict()

    assert isinstance(module_dict, nn.ModuleDict)
    assert "layer_0__attention__segment_0" in module_dict
    assert "layer_0__mlp__segment_0" in module_dict


def test_collection_rejects_wrong_attention_key_type() -> None:
    mlp_key = SegmentId(layer_id=0, segment_type="mlp", segment_id=0)
    with pytest.raises(ValueError):
        SegmentCollection(attention_segments={mlp_key: object()}, mlp_segments={})  # type: ignore[dict-item]


def test_factory_can_disable_biases_and_set_dropout() -> None:
    factory = SegmentFactory(
        model_config(),
        segmentation_config(),
        attention_bias=False,
        attention_dropout=0.0,
        mlp_input_bias=False,
        mlp_output_bias=True,
    )
    attention_segment = factory.create_attention_segment(layer_id=0, segment_index=0)
    mlp_segment = factory.create_mlp_segment(layer_id=0, segment_index=0)

    assert attention_segment.q_proj.bias is None
    assert mlp_segment.fc1.bias is None
    assert mlp_segment.fc2.bias is not None


def test_validate_complete_coverage_detects_missing_segment() -> None:
    factory = SegmentFactory(model_config(), segmentation_config())
    collection = factory.create_all_segments()
    first_id = collection.attention_segment_ids()[0]
    del collection.attention_segments[first_id]

    with pytest.raises(ValueError):
        collection.validate_complete_coverage(
            model_config=model_config(),
            segmentation_config=segmentation_config(),
        )
