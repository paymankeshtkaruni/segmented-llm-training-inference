"""Tests for Phase 7 segment storage backends."""

from __future__ import annotations

import pytest
import torch

from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId
from sequential_segmented_llm_training_inference.storage import (
    CpuRamSegmentStore,
    DiskSegmentStore,
    clone_state_dict_to_cpu,
)


def _state_dict(offset: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3) + offset,
        "bias": torch.tensor([1.0, 2.0], dtype=torch.float32) + offset,
    }


def test_clone_state_dict_to_cpu_clones_tensors() -> None:
    state = _state_dict()
    cloned = clone_state_dict_to_cpu(state)

    assert cloned is not state
    assert torch.equal(cloned["weight"], state["weight"])
    assert cloned["weight"].device.type == "cpu"

    state["weight"].add_(100)
    assert not torch.equal(cloned["weight"], state["weight"])


def test_disk_store_save_load_roundtrip(tmp_path) -> None:
    store = DiskSegmentStore(tmp_path)
    segment_id = SegmentId(layer_id=0, segment_type="attention", segment_id=1)
    state = _state_dict()

    assert not store.has_segment(segment_id)
    store.save_segment(segment_id, state)

    assert store.has_segment(segment_id)
    loaded = store.load_segment(segment_id)

    assert torch.equal(loaded["weight"], state["weight"])
    assert torch.equal(loaded["bias"], state["bias"])


def test_disk_store_uses_deterministic_path(tmp_path) -> None:
    store = DiskSegmentStore(tmp_path)
    segment_id = SegmentId(layer_id=3, segment_type="mlp", segment_id=2)

    path = store.segment_path(segment_id)

    assert path.name == "layer_3__mlp__segment_2.pt"
    assert path.parent == tmp_path


def test_disk_store_list_segments_sorted(tmp_path) -> None:
    store = DiskSegmentStore(tmp_path)
    ids = [
        SegmentId(layer_id=1, segment_type="mlp", segment_id=0),
        SegmentId(layer_id=0, segment_type="attention", segment_id=1),
        SegmentId(layer_id=0, segment_type="attention", segment_id=0),
    ]

    for idx, segment_id in enumerate(ids):
        store.save_segment(segment_id, _state_dict(float(idx)))

    assert store.list_segments() == sorted(ids)


def test_disk_store_load_missing_raises(tmp_path) -> None:
    store = DiskSegmentStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load_segment(SegmentId(layer_id=9, segment_type="attention", segment_id=0))


def test_cpu_ram_store_save_load_roundtrip() -> None:
    store = CpuRamSegmentStore()
    segment_id = SegmentId(layer_id=0, segment_type="mlp", segment_id=0)
    state = _state_dict()

    assert not store.has_segment(segment_id)
    store.save_segment(segment_id, state)

    assert store.has_segment(segment_id)
    loaded = store.load_segment(segment_id)

    assert torch.equal(loaded["weight"], state["weight"])
    assert torch.equal(loaded["bias"], state["bias"])


def test_cpu_ram_store_load_returns_clone() -> None:
    store = CpuRamSegmentStore()
    segment_id = SegmentId(layer_id=0, segment_type="attention", segment_id=0)
    store.save_segment(segment_id, _state_dict())

    loaded = store.load_segment(segment_id)
    loaded["weight"].add_(100)
    loaded_again = store.load_segment(segment_id)

    assert not torch.equal(loaded["weight"], loaded_again["weight"])


def test_cpu_ram_store_list_segments_sorted() -> None:
    store = CpuRamSegmentStore()
    ids = [
        SegmentId(layer_id=2, segment_type="mlp", segment_id=1),
        SegmentId(layer_id=0, segment_type="attention", segment_id=0),
        SegmentId(layer_id=1, segment_type="attention", segment_id=0),
    ]

    for idx, segment_id in enumerate(ids):
        store.save_segment(segment_id, _state_dict(float(idx)))

    assert store.list_segments() == sorted(ids)


def test_cpu_ram_store_clear_removes_segments() -> None:
    store = CpuRamSegmentStore()
    segment_id = SegmentId(layer_id=0, segment_type="attention", segment_id=0)
    store.save_segment(segment_id, _state_dict())

    store.clear()

    assert not store.has_segment(segment_id)
    assert store.list_segments() == []


def test_require_segment_raises_for_missing_segment(tmp_path) -> None:
    store = DiskSegmentStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.require_segment(SegmentId(layer_id=0, segment_type="mlp", segment_id=0))


def test_require_segment_passes_for_existing_segment(tmp_path) -> None:
    store = DiskSegmentStore(tmp_path)
    segment_id = SegmentId(layer_id=0, segment_type="mlp", segment_id=0)
    store.save_segment(segment_id, _state_dict())

    store.require_segment(segment_id)
