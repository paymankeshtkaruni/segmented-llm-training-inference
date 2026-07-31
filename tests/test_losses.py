"""Tests for Phase 11 causal language-modeling losses."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sequential_segmented_llm_training_inference.training.losses import (
    CausalCrossEntropyConfig,
    CausalCrossEntropyLoss,
    causal_cross_entropy_loss,
    shift_logits_and_labels,
    validate_causal_lm_shapes,
)


def test_shift_logits_and_labels_shapes() -> None:
    logits = torch.randn(2, 5, 11)
    labels = torch.randint(0, 11, (2, 5))

    shifted_logits, shifted_labels = shift_logits_and_labels(logits, labels)

    assert shifted_logits.shape == (2, 4, 11)
    assert shifted_labels.shape == (2, 4)


def test_shift_logits_and_labels_values() -> None:
    logits = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    labels = torch.tensor([[0, 1, 2, 0], [2, 1, 0, 2]])

    shifted_logits, shifted_labels = shift_logits_and_labels(logits, labels)

    assert torch.equal(shifted_logits, logits[:, :-1, :])
    assert torch.equal(shifted_labels, labels[:, 1:])


def test_causal_cross_entropy_matches_torch_reference() -> None:
    torch.manual_seed(7)
    logits = torch.randn(3, 6, 13)
    labels = torch.randint(0, 13, (3, 6))

    loss = causal_cross_entropy_loss(logits, labels)

    reference = F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, 13),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
        reduction="mean",
    )

    assert torch.allclose(loss, reference)


def test_causal_cross_entropy_respects_ignore_index() -> None:
    torch.manual_seed(11)
    logits = torch.randn(2, 5, 7)
    labels = torch.randint(0, 7, (2, 5))
    labels[0, 2] = -100
    labels[1, 4] = -100

    loss = causal_cross_entropy_loss(logits, labels, ignore_index=-100)

    reference = F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, 7),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
        reduction="mean",
    )

    assert torch.allclose(loss, reference)


def test_module_wrapper_matches_function() -> None:
    torch.manual_seed(13)
    logits = torch.randn(2, 4, 9)
    labels = torch.randint(0, 9, (2, 4))

    module = CausalCrossEntropyLoss(ignore_index=-100)
    module_loss = module(logits, labels)
    function_loss = causal_cross_entropy_loss(logits, labels)

    assert torch.allclose(module_loss, function_loss)


def test_loss_is_scalar_for_mean_reduction() -> None:
    logits = torch.randn(2, 5, 10)
    labels = torch.randint(0, 10, (2, 5))

    loss = causal_cross_entropy_loss(logits, labels, reduction="mean")

    assert loss.ndim == 0


def test_sum_reduction_is_scalar() -> None:
    logits = torch.randn(2, 5, 10)
    labels = torch.randint(0, 10, (2, 5))

    loss = causal_cross_entropy_loss(logits, labels, reduction="sum")

    assert loss.ndim == 0


def test_none_reduction_returns_flattened_shifted_losses() -> None:
    logits = torch.randn(2, 5, 10)
    labels = torch.randint(0, 10, (2, 5))

    loss = causal_cross_entropy_loss(logits, labels, reduction="none")

    assert loss.shape == (2 * 4,)


def test_invalid_reduction_raises() -> None:
    logits = torch.randn(2, 5, 10)
    labels = torch.randint(0, 10, (2, 5))

    with pytest.raises(ValueError, match="reduction"):
        causal_cross_entropy_loss(logits, labels, reduction="bad")  # type: ignore[arg-type]


def test_config_rejects_non_internal_shift() -> None:
    with pytest.raises(ValueError, match="label_shift"):
        CausalCrossEntropyConfig(label_shift="external")  # type: ignore[arg-type]


def test_validate_rejects_non_tensor_logits() -> None:
    labels = torch.randint(0, 10, (2, 5))
    with pytest.raises(TypeError, match="logits"):
        validate_causal_lm_shapes("not tensor", labels)  # type: ignore[arg-type]


def test_validate_rejects_non_tensor_labels() -> None:
    logits = torch.randn(2, 5, 10)
    with pytest.raises(TypeError, match="labels"):
        validate_causal_lm_shapes(logits, "not tensor")  # type: ignore[arg-type]


def test_validate_rejects_wrong_logits_rank() -> None:
    logits = torch.randn(2, 5)
    labels = torch.randint(0, 10, (2, 5))

    with pytest.raises(ValueError, match="logits"):
        validate_causal_lm_shapes(logits, labels)


def test_validate_rejects_wrong_labels_rank() -> None:
    logits = torch.randn(2, 5, 10)
    labels = torch.randint(0, 10, (2, 5, 1))

    with pytest.raises(ValueError, match="labels"):
        validate_causal_lm_shapes(logits, labels)


def test_validate_rejects_label_shape_mismatch() -> None:
    logits = torch.randn(2, 5, 10)
    labels = torch.randint(0, 10, (2, 4))

    with pytest.raises(ValueError, match="labels shape"):
        validate_causal_lm_shapes(logits, labels)


def test_validate_rejects_sequence_length_less_than_two() -> None:
    logits = torch.randn(2, 1, 10)
    labels = torch.randint(0, 10, (2, 1))

    with pytest.raises(ValueError, match="seq_len"):
        validate_causal_lm_shapes(logits, labels)
