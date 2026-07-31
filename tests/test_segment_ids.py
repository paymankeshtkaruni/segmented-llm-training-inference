"""Tests for Phase 2 logical segment identifiers."""

import pytest
import yaml

from sequential_segmented_llm_training_inference.segments import (
    SegmentId,
    SegmentParameterKey,
)


def test_segment_id_is_hashable_and_comparable() -> None:
    first = SegmentId(layer_id=1, segment_type="attention", segment_id=2)
    second = SegmentId(layer_id=1, segment_type="attention", segment_id=2)
    different = SegmentId(layer_id=1, segment_type="mlp", segment_id=2)

    assert first == second
    assert first != different
    assert {first, second, different} == {first, different}


def test_segment_id_serialization_is_deterministic() -> None:
    segment_id = SegmentId(layer_id=3, segment_type="mlp", segment_id=4)

    assert segment_id.to_key() == "layer_3.mlp.4"
    assert segment_id.to_path_name() == "layer_3__mlp__segment_4"
    assert segment_id.to_dict() == {
        "layer_id": 3,
        "segment_type": "mlp",
        "segment_id": 4,
    }
    assert SegmentId.from_key(segment_id.to_key()) == segment_id
    assert SegmentId.from_dict(segment_id.to_dict()) == segment_id


def test_segment_id_yaml_roundtrip() -> None:
    segment_id = SegmentId(layer_id=0, segment_type="attention", segment_id=1)
    dumped = yaml.safe_dump(segment_id.to_dict(), sort_keys=True)
    loaded = yaml.safe_load(dumped)

    assert SegmentId.from_dict(loaded) == segment_id


@pytest.mark.parametrize(
    "kwargs",
    [
        {"layer_id": -1, "segment_type": "attention", "segment_id": 0},
        {"layer_id": 0, "segment_type": "invalid", "segment_id": 0},
        {"layer_id": 0, "segment_type": "attention", "segment_id": -1},
        {"layer_id": 0.5, "segment_type": "attention", "segment_id": 0},
    ],
)
def test_segment_id_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        SegmentId(**kwargs)  # type: ignore[arg-type]


def test_segment_id_rejects_bad_dicts_and_keys() -> None:
    with pytest.raises(ValueError):
        SegmentId.from_dict({"layer_id": 0, "segment_type": "attention"})
    with pytest.raises(ValueError):
        SegmentId.from_key("bad")
    with pytest.raises(ValueError):
        SegmentId.from_key("layer_x.attention.0")
    with pytest.raises(ValueError):
        SegmentId.from_key("layer_0.bad.0")


def test_segment_parameter_key_is_hashable_and_deterministic() -> None:
    key = SegmentParameterKey(
        layer_id=3,
        segment_type="attention",
        segment_id=2,
        parameter_name="q_proj.weight",
    )

    assert key.to_key() == "layer_3.attention.2.q_proj.weight"
    assert SegmentParameterKey.from_key(key.to_key()) == key
    assert SegmentParameterKey.from_dict(key.to_dict()) == key
    assert {key, SegmentParameterKey.from_key(key.to_key())} == {key}


def test_segment_parameter_key_from_segment_id() -> None:
    segment_id = SegmentId(layer_id=5, segment_type="mlp", segment_id=1)
    key = SegmentParameterKey.from_segment_id(segment_id, "fc1.weight")

    assert key.layer_id == 5
    assert key.segment_type == "mlp"
    assert key.segment_id == 1
    assert key.parameter_name == "fc1.weight"
    assert key.segment_id_object == segment_id


def test_segment_parameter_key_yaml_roundtrip() -> None:
    key = SegmentParameterKey(
        layer_id=4,
        segment_type="mlp",
        segment_id=3,
        parameter_name="fc2.bias",
    )
    dumped = yaml.safe_dump(key.to_dict(), sort_keys=True)
    loaded = yaml.safe_load(dumped)

    assert SegmentParameterKey.from_dict(loaded) == key


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "layer_id": -1,
            "segment_type": "attention",
            "segment_id": 0,
            "parameter_name": "q_proj.weight",
        },
        {
            "layer_id": 0,
            "segment_type": "bad",
            "segment_id": 0,
            "parameter_name": "q_proj.weight",
        },
        {
            "layer_id": 0,
            "segment_type": "attention",
            "segment_id": -1,
            "parameter_name": "q_proj.weight",
        },
        {
            "layer_id": 0,
            "segment_type": "attention",
            "segment_id": 0,
            "parameter_name": "",
        },
        {
            "layer_id": 0,
            "segment_type": "attention",
            "segment_id": 0,
            "parameter_name": 123,
        },
    ],
)
def test_segment_parameter_key_rejects_invalid_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        SegmentParameterKey(**kwargs)  # type: ignore[arg-type]


def test_segment_parameter_key_rejects_bad_dicts_and_keys() -> None:
    with pytest.raises(ValueError):
        SegmentParameterKey.from_dict(
            {"layer_id": 0, "segment_type": "attention", "segment_id": 0}
        )
    with pytest.raises(ValueError):
        SegmentParameterKey.from_key("bad")
    with pytest.raises(ValueError):
        SegmentParameterKey.from_key("layer_x.attention.0.q_proj.weight")
    with pytest.raises(ValueError):
        SegmentParameterKey.from_key("layer_0.bad.0.q_proj.weight")
