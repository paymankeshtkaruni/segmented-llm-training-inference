"""Tests for Phase 17 segmented validation."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import Tensor, nn

from sequential_segmented_llm_training_inference.training.validator import (
    SegmentedValidator,
    ValidationResult,
)


class DummyForwardEngine(nn.Module):
    def __init__(self, vocab_size: int = 5) -> None:
        super().__init__()
        self.proj = nn.Linear(3, vocab_size)
        self.forward_calls = 0

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        self.forward_calls += 1
        features = torch.nn.functional.one_hot(input_ids % 3, num_classes=3).float()
        logits = self.proj(features)
        loss = logits.mean()
        return {"logits": logits, "loss": loss}


def make_batches(num_batches: int = 3, batch_size: int = 2) -> list[dict[str, Tensor]]:
    batches = []
    for i in range(num_batches):
        input_ids = torch.tensor(
            [[i, i + 1, i + 2], [i + 2, i + 3, i + 4]],
            dtype=torch.long,
        )[:batch_size]
        labels = input_ids.clone()
        attention_mask = torch.ones_like(input_ids)
        batches.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )
    return batches


def clone_state(module: nn.Module) -> dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def assert_state_equal(before: dict[str, Tensor], after: dict[str, Tensor]) -> None:
    assert before.keys() == after.keys()
    for key in before:
        assert torch.equal(before[key], after[key]), key


def test_validation_runs_segmented_forward_only_without_weight_changes() -> None:
    engine = DummyForwardEngine()
    before = clone_state(engine)

    validator = SegmentedValidator(engine)
    result = validator.validate(make_batches())

    assert isinstance(result, ValidationResult)
    assert result.loss is not None
    assert result.num_batches == 3
    assert result.num_examples == 6
    assert engine.forward_calls == 3
    assert_state_equal(before, engine.state_dict())


def test_validation_uses_no_grad() -> None:
    class GradCheckingEngine(DummyForwardEngine):
        def forward(self, **batch: Tensor) -> dict[str, Tensor]:
            assert not torch.is_grad_enabled()
            return super().forward(**batch)

    engine = GradCheckingEngine()
    result = SegmentedValidator(engine).validate(make_batches(num_batches=1))

    assert result.num_batches == 1


def test_validation_restores_training_mode() -> None:
    engine = DummyForwardEngine()
    engine.train(True)

    SegmentedValidator(engine).validate(make_batches(num_batches=1))

    assert engine.training is True

    engine.eval()
    SegmentedValidator(engine).validate(make_batches(num_batches=1))
    assert engine.training is False


def test_validation_can_compute_loss_with_external_loss_fn_when_output_has_no_loss() -> None:
    class LogitsOnlyEngine(nn.Module):
        def forward(self, input_ids: Tensor, labels: Tensor, **_: Tensor) -> dict[str, Tensor]:
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 7)
            return {"logits": logits}

    def loss_fn(logits: Tensor, labels: Tensor) -> Tensor:
        return logits.sum() * 0 + labels.float().mean()

    batches = make_batches(num_batches=2)
    expected = sum(float(b["labels"].float().mean()) * b["input_ids"].shape[0] for b in batches)
    expected /= sum(b["input_ids"].shape[0] for b in batches)

    result = SegmentedValidator(LogitsOnlyEngine(), loss_fn=loss_fn).validate(batches)

    assert result.loss == pytest.approx(expected)


def test_validation_metrics_are_averaged_by_examples() -> None:
    engine = DummyForwardEngine()

    def constant_metric(logits: Tensor, labels: Tensor) -> float:
        return float(labels.shape[0])

    batches = make_batches(num_batches=2, batch_size=2)
    result = SegmentedValidator(
        engine,
        metric_fns={"batch_size_metric": constant_metric},
    ).validate(batches)

    assert result.metrics["batch_size_metric"] == pytest.approx(2.0)


def test_validation_rejects_metric_without_logits() -> None:
    class LossOnlyEngine(nn.Module):
        def forward(self, **_: Tensor) -> dict[str, Tensor]:
            return {"loss": torch.tensor(1.0)}

    validator = SegmentedValidator(
        LossOnlyEngine(),
        metric_fns={"metric": lambda logits, labels: 1.0},
    )

    with pytest.raises(ValueError, match="logits"):
        validator.validate(make_batches(num_batches=1))


def test_validation_rejects_non_scalar_loss() -> None:
    class BadLossEngine(nn.Module):
        def forward(self, **_: Tensor) -> dict[str, Tensor]:
            return {"loss": torch.ones(2)}

    with pytest.raises(ValueError, match="scalar"):
        SegmentedValidator(BadLossEngine()).validate(make_batches(num_batches=1))


def test_result_to_dict() -> None:
    result = ValidationResult(
        loss=1.25,
        metrics={"accuracy": 0.5},
        num_batches=2,
        num_examples=8,
    )

    assert result.to_dict() == {
        "loss": 1.25,
        "metrics": {"accuracy": 0.5},
        "num_batches": 2,
        "num_examples": 8,
    }
