"""Tests for Phase 13 RNG state capture and restoration."""

from __future__ import annotations

import random

import pytest
import torch

from sequential_segmented_llm_training_inference.execution.rng import (
    RngState,
    RngStateTracker,
    capture_rng_state,
    cuda_rng_capture_available,
    restore_rng_state,
    restored_rng_state,
)


def _dropout_like_operation(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    mask = (torch.rand_like(x) > p).to(x.dtype)
    return x * mask / (1.0 - p)


def test_capture_restore_reproduces_torch_cpu_random_values() -> None:
    torch.manual_seed(123)
    state = capture_rng_state(include_cuda=False)
    first = torch.rand(8)

    _ = torch.rand(17)
    restore_rng_state(state, restore_cuda=False)
    second = torch.rand(8)

    assert torch.equal(first, second)


def test_capture_restore_reproduces_python_random_values() -> None:
    random.seed(321)
    state = capture_rng_state(include_cuda=False)
    first = [random.random() for _ in range(5)]

    _ = [random.random() for _ in range(11)]
    restore_rng_state(state, restore_cuda=False)
    second = [random.random() for _ in range(5)]

    assert first == second


def test_dropout_mask_reproduced_with_rng_restore() -> None:
    torch.manual_seed(111)
    x = torch.ones(4, 6)
    state = capture_rng_state(include_cuda=False)
    first = _dropout_like_operation(x, p=0.5)

    _ = torch.rand(99)
    restore_rng_state(state, restore_cuda=False)
    second = _dropout_like_operation(x, p=0.5)

    assert torch.equal(first, second)


def test_dropout_mask_differs_without_rng_restore() -> None:
    torch.manual_seed(112)
    x = torch.ones(128, 128)
    first = _dropout_like_operation(x, p=0.5)
    second = _dropout_like_operation(x, p=0.5)

    assert not torch.equal(first, second)


def test_dropout_zero_is_deterministic_without_rng_restore() -> None:
    x = torch.randn(4, 5)
    first = torch.nn.functional.dropout(x, p=0.0, training=True)
    _ = torch.rand(10)
    second = torch.nn.functional.dropout(x, p=0.0, training=True)

    assert torch.equal(first, second)


def test_restored_rng_state_context_preserves_outer_rng_state() -> None:
    torch.manual_seed(777)
    segment_state = capture_rng_state(include_cuda=False)
    _ = torch.rand(5)
    state_before_context = capture_rng_state(include_cuda=False)

    with restored_rng_state(segment_state, restore_cuda=False, preserve_current=True):
        inside = torch.rand(5)

    after_context = torch.rand(5)

    restore_rng_state(segment_state, restore_cuda=False)
    expected_inside = torch.rand(5)
    assert torch.equal(inside, expected_inside)

    restore_rng_state(state_before_context, restore_cuda=False)
    expected_after_context = torch.rand(5)
    assert torch.equal(after_context, expected_after_context)


def test_context_can_leave_rng_advanced_when_not_preserving_current() -> None:
    torch.manual_seed(888)
    state = capture_rng_state(include_cuda=False)

    with restored_rng_state(state, restore_cuda=False, preserve_current=False):
        inside = torch.rand(3)

    restore_rng_state(state, restore_cuda=False)
    expected_inside = torch.rand(3)
    assert torch.equal(inside, expected_inside)


def test_rng_state_clone_is_independent_for_tensors() -> None:
    torch.manual_seed(999)
    state = capture_rng_state(include_cuda=False)
    clone = state.clone()

    clone.torch_cpu_state.zero_()

    assert not torch.equal(state.torch_cpu_state, clone.torch_cpu_state)


def test_checkpoint_roundtrip() -> None:
    torch.manual_seed(1234)
    state = capture_rng_state(include_cuda=False)
    data = state.to_checkpoint()
    restored = RngState.from_checkpoint(data)

    restore_rng_state(state, restore_cuda=False)
    first = torch.rand(4)
    restore_rng_state(restored, restore_cuda=False)
    second = torch.rand(4)

    assert torch.equal(first, second)


def test_checkpoint_missing_field_raises() -> None:
    with pytest.raises(ValueError, match="Missing RNG checkpoint"):
        RngState.from_checkpoint({})


def test_checkpoint_bad_cpu_state_raises() -> None:
    data = {
        "python_state": random.getstate(),
        "torch_cpu_state": "bad",
        "torch_cuda_states": None,
    }
    with pytest.raises(TypeError, match="torch_cpu_state"):
        RngState.from_checkpoint(data)


def test_tracker_capture_get_restore_and_clear() -> None:
    torch.manual_seed(2024)
    tracker = RngStateTracker()
    tracker.capture("layer_0.attention.0", include_cuda=False)
    expected = torch.rand(6)

    _ = torch.rand(13)
    tracker.restore("layer_0.attention.0", restore_cuda=False)
    actual = torch.rand(6)

    assert torch.equal(actual, expected)
    assert tracker.has("layer_0.attention.0")
    assert len(tracker) == 1
    assert tracker.keys() == ("layer_0.attention.0",)

    tracker.clear()
    assert len(tracker) == 0


def test_tracker_get_missing_key_raises() -> None:
    tracker = RngStateTracker()
    with pytest.raises(KeyError, match="missing"):
        tracker.get("missing")


def test_tracker_rejects_empty_key() -> None:
    tracker = RngStateTracker()
    with pytest.raises(ValueError, match="must not be empty"):
        tracker.capture("", include_cuda=False)


def test_restore_rejects_wrong_type() -> None:
    with pytest.raises(TypeError, match="RngState"):
        restore_rng_state("bad")  # type: ignore[arg-type]


@pytest.mark.skipif(not cuda_rng_capture_available(), reason="CUDA not available")
def test_cuda_rng_state_roundtrip_when_available() -> None:
    torch.cuda.manual_seed_all(42)
    state = capture_rng_state(include_cuda=True)
    first = torch.rand(8, device="cuda")

    _ = torch.rand(32, device="cuda")
    restore_rng_state(state, restore_cuda=True)
    second = torch.rand(8, device="cuda")

    assert torch.equal(first.cpu(), second.cpu())
