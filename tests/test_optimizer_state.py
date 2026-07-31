"""Tests for Phase 14 segment-wise optimizers."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from sequential_segmented_llm_training_inference.optimization import (
    SegmentGradientAccumulator,
    SegmentwiseAdamW,
    SegmentwiseSGD,
    segment_parameter_key,
    validate_update_style_and_accumulation,
)
from sequential_segmented_llm_training_inference.segments.segment_ids import (
    SegmentId,
    SegmentParameterKey,
)
from sequential_segmented_llm_training_inference.storage.runtime_records import (
    MLPSegmentGradientRecord,
)


def make_module() -> nn.Linear:
    module = nn.Linear(3, 2, bias=True)
    with torch.no_grad():
        module.weight.fill_(0.5)
        module.bias.fill_(0.1)
    return module


def make_gradients(module: nn.Module, value: float = 0.25) -> dict[str, torch.Tensor]:
    return {name: torch.full_like(param, value) for name, param in module.named_parameters()}


def test_segment_parameter_key_uses_logical_identity() -> None:
    segment_id = SegmentId(3, "attention", 2)
    key = segment_parameter_key(segment_id, "q_proj.weight")

    assert key == SegmentParameterKey(3, "attention", 2, "q_proj.weight")
    assert key.to_key() == "layer_3.attention.2.q_proj.weight"


def test_validate_update_style_accepts_after_full_backward_with_accumulation() -> None:
    validate_update_style_and_accumulation(
        update_style="after_full_backward",
        gradient_accumulation_steps=8,
    )


def test_validate_update_style_rejects_immediate_update_with_accumulation() -> None:
    with pytest.raises(ValueError, match="immediate_segment_update"):
        validate_update_style_and_accumulation(
            update_style="immediate_segment_update",
            gradient_accumulation_steps=2,
        )


def test_validate_update_style_allows_delayed_immediate_update() -> None:
    validate_update_style_and_accumulation(
        update_style="immediate_segment_update",
        gradient_accumulation_steps=2,
        delayed_immediate_updates=True,
    )


def test_sgd_updates_segment_weights() -> None:
    segment_id = SegmentId(0, "mlp", 1)
    module = make_module()
    before = module.weight.detach().clone()
    optimizer = SegmentwiseSGD(lr=0.1)

    result = optimizer.step_module(
        segment_id=segment_id,
        module=module,
        gradients=make_gradients(module, 0.5),
    )

    assert result.optimizer_type == "stateless_sgd"
    assert set(result.updated_parameter_names) == {"weight", "bias"}
    assert not torch.allclose(module.weight, before)


def test_sgd_momentum_state_persists_by_logical_key() -> None:
    segment_id = SegmentId(1, "mlp", 0)
    module_a = make_module()
    module_b = make_module()
    optimizer = SegmentwiseSGD(lr=0.1, momentum=0.9)

    optimizer.step_module(
        segment_id=segment_id,
        module=module_a,
        gradients=make_gradients(module_a, 0.2),
    )
    optimizer.step_module(
        segment_id=segment_id,
        module=module_b,
        gradients=make_gradients(module_b, 0.3),
    )

    key = SegmentParameterKey(1, "mlp", 0, "weight")
    assert key in optimizer.state
    assert "momentum_buffer" in optimizer.state[key]


def test_sgd_state_dict_roundtrip() -> None:
    segment_id = SegmentId(1, "mlp", 0)
    module = make_module()
    optimizer = SegmentwiseSGD(lr=0.1, momentum=0.9)
    optimizer.step_module(
        segment_id=segment_id,
        module=module,
        gradients=make_gradients(module),
    )

    restored = SegmentwiseSGD(lr=1.0)
    restored.load_state_dict(optimizer.state_dict())

    assert restored.lr == pytest.approx(0.1)
    assert SegmentParameterKey(1, "mlp", 0, "weight") in restored.state


def test_adamw_updates_segment_weights_and_stores_state() -> None:
    segment_id = SegmentId(2, "attention", 1)
    module = make_module()
    before = module.weight.detach().clone()
    optimizer = SegmentwiseAdamW(lr=0.01, weight_decay=0.0)

    result = optimizer.step_module(
        segment_id=segment_id,
        module=module,
        gradients=make_gradients(module, 0.1),
    )

    assert result.optimizer_type == "segmentwise_adamw"
    assert not torch.allclose(module.weight, before)
    key = SegmentParameterKey(2, "attention", 1, "weight")
    assert key in optimizer.state
    assert optimizer.state[key]["step"] == 1
    assert "exp_avg" in optimizer.state[key]
    assert "exp_avg_sq" in optimizer.state[key]


def test_adamw_state_persists_across_new_module_instance() -> None:
    segment_id = SegmentId(2, "attention", 1)
    module_a = make_module()
    module_b = make_module()
    optimizer = SegmentwiseAdamW(lr=0.01, weight_decay=0.0)

    optimizer.step_module(
        segment_id=segment_id,
        module=module_a,
        gradients=make_gradients(module_a, 0.1),
    )
    optimizer.step_module(
        segment_id=segment_id,
        module=module_b,
        gradients=make_gradients(module_b, 0.1),
    )

    key = SegmentParameterKey(2, "attention", 1, "weight")
    assert optimizer.state[key]["step"] == 2


def test_adamw_state_dict_roundtrip() -> None:
    segment_id = SegmentId(0, "attention", 0)
    module = make_module()
    optimizer = SegmentwiseAdamW(lr=0.01, weight_decay=0.0)
    optimizer.step_module(
        segment_id=segment_id,
        module=module,
        gradients=make_gradients(module),
    )

    restored = SegmentwiseAdamW(lr=0.1)
    restored.load_state_dict(optimizer.state_dict())

    key = SegmentParameterKey(0, "attention", 0, "weight")
    assert restored.lr == pytest.approx(0.01)
    assert key in restored.state
    assert restored.state[key]["step"] == 1


def test_strict_gradient_mapping_rejects_missing_gradient() -> None:
    segment_id = SegmentId(0, "mlp", 0)
    module = make_module()
    optimizer = SegmentwiseSGD(lr=0.1)

    with pytest.raises(KeyError, match="Missing gradients"):
        optimizer.step_module(
            segment_id=segment_id,
            module=module,
            gradients={"weight": torch.ones_like(module.weight)},
            strict=True,
        )


def test_strict_gradient_mapping_rejects_unexpected_gradient() -> None:
    segment_id = SegmentId(0, "mlp", 0)
    module = make_module()
    optimizer = SegmentwiseSGD(lr=0.1)
    gradients = make_gradients(module)
    gradients["extra"] = torch.tensor(1.0)

    with pytest.raises(KeyError, match="Unexpected gradients"):
        optimizer.step_module(
            segment_id=segment_id,
            module=module,
            gradients=gradients,
            strict=True,
        )


def test_gradient_shape_mismatch_raises() -> None:
    segment_id = SegmentId(0, "mlp", 0)
    module = make_module()
    optimizer = SegmentwiseSGD(lr=0.1)
    gradients = make_gradients(module)
    gradients["weight"] = torch.ones(1)

    with pytest.raises(ValueError, match="Gradient shape mismatch"):
        optimizer.step_module(
            segment_id=segment_id,
            module=module,
            gradients=gradients,
        )


def test_non_strict_update_allows_partial_gradients() -> None:
    segment_id = SegmentId(0, "mlp", 0)
    module = make_module()
    before_bias = module.bias.detach().clone()
    optimizer = SegmentwiseSGD(lr=0.1)

    result = optimizer.step_module(
        segment_id=segment_id,
        module=module,
        gradients={"weight": torch.ones_like(module.weight)},
        strict=False,
    )

    assert result.updated_parameter_names == ("weight",)
    assert torch.allclose(module.bias, before_bias)


def test_gradient_accumulator_sums_gradients() -> None:
    segment_id = SegmentId(0, "mlp", 0)
    accumulator = SegmentGradientAccumulator()
    accumulator.accumulate(segment_id, {"weight": torch.ones(2, 3)})
    accumulator.accumulate(segment_id, {"weight": torch.full((2, 3), 2.0)})

    gradients = accumulator.get(segment_id)

    assert torch.equal(gradients["weight"], torch.full((2, 3), 3.0))
    assert accumulator.num_segments == 1
    assert accumulator.num_parameter_gradients == 1


def test_gradient_accumulator_accepts_gradient_records() -> None:
    segment_id = SegmentId(0, "mlp", 0)
    record = MLPSegmentGradientRecord(
        segment_id=segment_id,
        parameter_gradients={"weight": torch.ones(2, 3)},
        input_gradient=torch.ones(1, 2, 3),
    )
    accumulator = SegmentGradientAccumulator()

    accumulator.accumulate_record(record, scale=0.5)

    assert torch.equal(accumulator.get(segment_id)["weight"], torch.full((2, 3), 0.5))


def test_gradient_accumulator_pop_and_clear() -> None:
    segment_id = SegmentId(0, "attention", 0)
    accumulator = SegmentGradientAccumulator()
    accumulator.accumulate(segment_id, {"q_proj.weight": torch.ones(2, 2)})

    popped = accumulator.pop(segment_id)
    assert torch.equal(popped["q_proj.weight"], torch.ones(2, 2))
    assert accumulator.is_empty()

    accumulator.accumulate(segment_id, {"q_proj.weight": torch.ones(2, 2)})
    accumulator.clear()
    assert accumulator.is_empty()
