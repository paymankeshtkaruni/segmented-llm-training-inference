"""Tests for Phase 8 strict single-segment loader."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from sequential_segmented_llm_training_inference.execution.segment_loader import (
    StrictSegmentLoader,
)
from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId
from sequential_segmented_llm_training_inference.storage.cpu_ram_segment_store import (
    CpuRamSegmentStore,
)


class TinySegment(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def make_segment_id(index: int = 0) -> SegmentId:
    return SegmentId(layer_id=0, segment_type="mlp", segment_id=index)


def make_store_with_segment(segment_id: SegmentId, weight_value: float = 1.0) -> CpuRamSegmentStore:
    store = CpuRamSegmentStore()
    module = TinySegment()
    with torch.no_grad():
        module.linear.weight.fill_(weight_value)
    store.save_segment(segment_id, module.state_dict())
    return store


def module_factory(_: SegmentId) -> nn.Module:
    return TinySegment()


def test_loader_starts_with_no_active_segment() -> None:
    segment_id = make_segment_id()
    store = make_store_with_segment(segment_id)
    loader = StrictSegmentLoader(
        segment_store=store,
        module_factory=module_factory,
        device="cpu",
    )

    assert loader.active_segment is None
    assert loader.active_segment_id is None
    assert loader.active_module is None
    assert loader.active_segment_count == 0
    assert not loader.has_active_segment()


def test_load_segment_sets_active_segment_and_device() -> None:
    segment_id = make_segment_id()
    store = make_store_with_segment(segment_id, weight_value=2.0)
    loader = StrictSegmentLoader(
        segment_store=store,
        module_factory=module_factory,
        device="cpu",
    )

    module = loader.load_segment(segment_id)

    assert loader.active_segment_count == 1
    assert loader.active_segment_id == segment_id
    assert loader.active_module is module
    assert next(module.parameters()).device.type == "cpu"
    assert torch.equal(module.linear.weight, torch.full_like(module.linear.weight, 2.0))


def test_loading_second_segment_before_release_raises() -> None:
    first_id = make_segment_id(0)
    second_id = make_segment_id(1)
    store = make_store_with_segment(first_id)
    second_module = TinySegment()
    store.save_segment(second_id, second_module.state_dict())

    loader = StrictSegmentLoader(
        segment_store=store,
        module_factory=module_factory,
        device="cpu",
    )
    loader.load_segment(first_id)

    with pytest.raises(RuntimeError, match="already active"):
        loader.load_segment(second_id)

    assert loader.active_segment_count == 1
    assert loader.active_segment_id == first_id


def test_release_without_save_discards_weight_changes() -> None:
    segment_id = make_segment_id()
    store = make_store_with_segment(segment_id, weight_value=1.0)
    loader = StrictSegmentLoader(
        segment_store=store,
        module_factory=module_factory,
        device="cpu",
    )

    module = loader.load_segment(segment_id)
    with torch.no_grad():
        module.linear.weight.fill_(9.0)

    loader.release_segment(save=False)
    reloaded = store.load_segment(segment_id)

    assert loader.active_segment_count == 0
    assert torch.equal(reloaded["linear.weight"], torch.full_like(reloaded["linear.weight"], 1.0))


def test_release_with_save_persists_weight_changes() -> None:
    segment_id = make_segment_id()
    store = make_store_with_segment(segment_id, weight_value=1.0)
    loader = StrictSegmentLoader(
        segment_store=store,
        module_factory=module_factory,
        device="cpu",
    )

    module = loader.load_segment(segment_id)
    with torch.no_grad():
        module.linear.weight.fill_(7.0)

    loader.release_segment(save=True)
    reloaded = store.load_segment(segment_id)

    assert loader.active_segment_count == 0
    assert torch.equal(reloaded["linear.weight"], torch.full_like(reloaded["linear.weight"], 7.0))


def test_save_active_segment_without_releasing() -> None:
    segment_id = make_segment_id()
    store = make_store_with_segment(segment_id, weight_value=1.0)
    loader = StrictSegmentLoader(
        segment_store=store,
        module_factory=module_factory,
        device="cpu",
    )

    module = loader.load_segment(segment_id)
    with torch.no_grad():
        module.linear.weight.fill_(5.0)

    loader.save_active_segment()
    reloaded = store.load_segment(segment_id)

    assert loader.active_segment_count == 1
    assert torch.equal(reloaded["linear.weight"], torch.full_like(reloaded["linear.weight"], 5.0))


def test_release_without_active_segment_raises() -> None:
    segment_id = make_segment_id()
    store = make_store_with_segment(segment_id)
    loader = StrictSegmentLoader(
        segment_store=store,
        module_factory=module_factory,
        device="cpu",
    )

    with pytest.raises(RuntimeError, match="No active segment"):
        loader.release_segment()


def test_save_without_active_segment_raises() -> None:
    segment_id = make_segment_id()
    store = make_store_with_segment(segment_id)
    loader = StrictSegmentLoader(
        segment_store=store,
        module_factory=module_factory,
        device="cpu",
    )

    with pytest.raises(RuntimeError, match="No active segment"):
        loader.save_active_segment()


def test_missing_segment_raises_file_not_found() -> None:
    store = CpuRamSegmentStore()
    loader = StrictSegmentLoader(
        segment_store=store,
        module_factory=module_factory,
        device="cpu",
    )

    with pytest.raises(FileNotFoundError):
        loader.load_segment(make_segment_id())


def test_module_factory_must_be_callable() -> None:
    segment_id = make_segment_id()
    store = make_store_with_segment(segment_id)

    with pytest.raises(TypeError, match="module_factory must be callable"):
        StrictSegmentLoader(
            segment_store=store,
            module_factory=None,  # type: ignore[arg-type]
            device="cpu",
        )


def test_module_factory_must_return_module() -> None:
    segment_id = make_segment_id()
    store = make_store_with_segment(segment_id)

    def bad_factory(_: SegmentId) -> object:
        return object()

    loader = StrictSegmentLoader(
        segment_store=store,
        module_factory=bad_factory,  # type: ignore[arg-type]
        device="cpu",
    )

    with pytest.raises(TypeError, match="torch.nn.Module"):
        loader.load_segment(segment_id)


def test_acquire_segment_context_releases_without_save() -> None:
    segment_id = make_segment_id()
    store = make_store_with_segment(segment_id, weight_value=1.0)
    loader = StrictSegmentLoader(
        segment_store=store,
        module_factory=module_factory,
        device="cpu",
    )

    with loader.acquire_segment(segment_id, save_on_exit=False) as module:
        assert loader.active_segment_count == 1
        with torch.no_grad():
            module.linear.weight.fill_(6.0)

    reloaded = store.load_segment(segment_id)
    assert loader.active_segment_count == 0
    assert torch.equal(reloaded["linear.weight"], torch.full_like(reloaded["linear.weight"], 1.0))


def test_acquire_segment_context_saves_on_exit() -> None:
    segment_id = make_segment_id()
    store = make_store_with_segment(segment_id, weight_value=1.0)
    loader = StrictSegmentLoader(
        segment_store=store,
        module_factory=module_factory,
        device="cpu",
    )

    with loader.acquire_segment(segment_id, save_on_exit=True) as module:
        with torch.no_grad():
            module.linear.weight.fill_(8.0)

    reloaded = store.load_segment(segment_id)
    assert loader.active_segment_count == 0
    assert torch.equal(reloaded["linear.weight"], torch.full_like(reloaded["linear.weight"], 8.0))


def test_context_releases_on_exception() -> None:
    segment_id = make_segment_id()
    store = make_store_with_segment(segment_id)
    loader = StrictSegmentLoader(
        segment_store=store,
        module_factory=module_factory,
        device="cpu",
    )

    with pytest.raises(ValueError, match="boom"):
        with loader.acquire_segment(segment_id):
            assert loader.active_segment_count == 1
            raise ValueError("boom")

    assert loader.active_segment_count == 0
