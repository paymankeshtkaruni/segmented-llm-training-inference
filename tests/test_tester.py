"""Tests for Phase 18 final segmented testing."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from sequential_segmented_llm_training_inference.training.tester import (
    SegmentedTester,
    TestResult,
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


def test_final_test_runs_segmented_forward_only_without_weight_changes() -> None:
    engine = DummyForwardEngine()
    before = clone_state(engine)

    tester = SegmentedTester(engine)
    result = tester.test(make_batches())

    assert isinstance(result, TestResult)
    assert result.loss is not None
    assert result.num_batches == 3
    assert result.num_examples == 6
    assert engine.forward_calls == 3
    assert_state_equal(before, engine.state_dict())


def test_test_uses_no_grad() -> None:
    class GradCheckingEngine(DummyForwardEngine):
        def forward(self, **batch: Tensor) -> dict[str, Tensor]:
            assert not torch.is_grad_enabled()
            return super().forward(**batch)

    result = SegmentedTester(GradCheckingEngine()).test(make_batches(num_batches=1))
    assert result.num_batches == 1


def test_test_restores_training_mode() -> None:
    engine = DummyForwardEngine()
    engine.train(True)

    SegmentedTester(engine).test(make_batches(num_batches=1))
    assert engine.training is True

    engine.eval()
    SegmentedTester(engine).test(make_batches(num_batches=1))
    assert engine.training is False


def test_test_can_compute_loss_with_external_loss_fn_when_output_has_no_loss() -> None:
    class LogitsOnlyEngine(nn.Module):
        def forward(self, input_ids: Tensor, labels: Tensor, **_: Tensor) -> dict[str, Tensor]:
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 7)
            return {"logits": logits}

    def loss_fn(logits: Tensor, labels: Tensor) -> Tensor:
        return logits.sum() * 0 + labels.float().mean()

    batches = make_batches(num_batches=2)
    expected = sum(float(b["labels"].float().mean()) * b["input_ids"].shape[0] for b in batches)
    expected /= sum(b["input_ids"].shape[0] for b in batches)

    result = SegmentedTester(LogitsOnlyEngine(), loss_fn=loss_fn).test(batches)
    assert result.loss == pytest.approx(expected)


def test_test_metrics_are_averaged_by_examples() -> None:
    engine = DummyForwardEngine()

    def constant_metric(logits: Tensor, labels: Tensor) -> float:
        return float(labels.shape[0])

    result = SegmentedTester(
        engine,
        metric_fns={"batch_size_metric": constant_metric},
    ).test(make_batches(num_batches=2, batch_size=2))

    assert result.metrics["batch_size_metric"] == pytest.approx(2.0)


def test_test_rejects_metric_without_logits() -> None:
    class LossOnlyEngine(nn.Module):
        def forward(self, **_: Tensor) -> dict[str, Tensor]:
            return {"loss": torch.tensor(1.0)}

    tester = SegmentedTester(
        LossOnlyEngine(),
        metric_fns={"metric": lambda logits, labels: 1.0},
    )

    with pytest.raises(ValueError, match="logits"):
        tester.test(make_batches(num_batches=1))


def test_test_rejects_non_scalar_loss() -> None:
    class BadLossEngine(nn.Module):
        def forward(self, **_: Tensor) -> dict[str, Tensor]:
            return {"loss": torch.ones(2)}

    with pytest.raises(ValueError, match="scalar"):
        SegmentedTester(BadLossEngine()).test(make_batches(num_batches=1))


def test_checkpoint_requires_loader() -> None:
    with pytest.raises(ValueError, match="checkpoint_loader"):
        SegmentedTester(DummyForwardEngine()).test(
            make_batches(num_batches=1),
            checkpoint="best_segmented_checkpoint",
        )


def test_checkpoint_loader_called_before_test() -> None:
    engine = DummyForwardEngine()
    calls: list[str] = []

    def loader(checkpoint: str | Path) -> dict[str, object]:
        calls.append(str(checkpoint))
        return {"metadata": {"checkpoint_name": str(checkpoint), "loaded": True}}

    result = SegmentedTester(engine, checkpoint_loader=loader).test(
        make_batches(num_batches=1),
        checkpoint="best_segmented_checkpoint",
    )

    assert calls == ["best_segmented_checkpoint"]
    assert result.checkpoint == "best_segmented_checkpoint"
    assert result.checkpoint_metadata == {
        "checkpoint_name": "best_segmented_checkpoint",
        "loaded": True,
    }


def test_empty_dataloader_returns_empty_result() -> None:
    result = SegmentedTester(DummyForwardEngine()).test([])
    assert result.loss is None
    assert result.metrics == {}
    assert result.num_batches == 0
    assert result.num_examples == 0


def test_result_to_dict() -> None:
    result = TestResult(
        loss=1.25,
        metrics={"accuracy": 0.5},
        num_batches=2,
        num_examples=8,
        checkpoint="best_segmented_checkpoint",
        checkpoint_metadata={"epoch": 3},
    )

    assert result.to_dict() == {
        "loss": 1.25,
        "metrics": {"accuracy": 0.5},
        "num_batches": 2,
        "num_examples": 8,
        "checkpoint": "best_segmented_checkpoint",
        "checkpoint_metadata": {"epoch": 3},
    }
