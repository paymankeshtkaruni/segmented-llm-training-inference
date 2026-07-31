"""Tests for Phase 9 segment-level runtime records."""

from __future__ import annotations

import pytest
import torch

from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId
from sequential_segmented_llm_training_inference.storage.runtime_records import (
    AttentionCompositionRecord,
    AttentionSegmentExecutionRecord,
    AttentionSegmentGradientRecord,
    MLPCompositionRecord,
    MLPSegmentExecutionRecord,
    MLPSegmentGradientRecord,
    ResidualCompositionRecord,
    RuntimeRecordStore,
)


def attention_id(segment_id: int = 0, layer_id: int = 0) -> SegmentId:
    return SegmentId(layer_id=layer_id, segment_type="attention", segment_id=segment_id)


def mlp_id(segment_id: int = 0, layer_id: int = 0) -> SegmentId:
    return SegmentId(layer_id=layer_id, segment_type="mlp", segment_id=segment_id)


def test_attention_execution_record_is_segment_level_and_derives_metadata() -> None:
    output = torch.randn(2, 3, 4)
    record = AttentionSegmentExecutionRecord(
        segment_id=attention_id(2, layer_id=1),
        step_id=7,
        microbatch_id=0,
        input_reference="layer_1.attention_input",
        output_tensor=output,
        attention_mask_reference="batch.attention_mask",
    )

    assert record.segment_id == attention_id(2, layer_id=1)
    assert record.layer_id == 1
    assert record.shape == (2, 3, 4)
    assert record.dtype == str(output.dtype)
    assert record.device == str(output.device)
    assert record.input_reference == "layer_1.attention_input"


def test_mlp_execution_record_is_segment_level_and_derives_metadata() -> None:
    output = torch.randn(2, 3, 8)
    record = MLPSegmentExecutionRecord(
        segment_id=mlp_id(3, layer_id=2),
        step_id=9,
        microbatch_id=1,
        input_reference="layer_2.mlp_input",
        output_tensor=output,
    )

    assert record.segment_id == mlp_id(3, layer_id=2)
    assert record.layer_id == 2
    assert record.shape == (2, 3, 8)
    assert record.dtype == str(output.dtype)
    assert record.device == str(output.device)


def test_attention_record_rejects_mlp_segment_id() -> None:
    with pytest.raises(ValueError, match="attention"):
        AttentionSegmentExecutionRecord(
            segment_id=mlp_id(),
            step_id=0,
            microbatch_id=0,
            input_reference="x",
            output_tensor=torch.randn(1, 2, 3),
        )


def test_mlp_record_rejects_attention_segment_id() -> None:
    with pytest.raises(ValueError, match="mlp"):
        MLPSegmentExecutionRecord(
            segment_id=attention_id(),
            step_id=0,
            microbatch_id=0,
            input_reference="x",
            output_tensor=torch.randn(1, 2, 3),
        )


def test_execution_record_requires_input_reference_or_tensor() -> None:
    with pytest.raises(ValueError, match="input_reference or input_tensor"):
        MLPSegmentExecutionRecord(
            segment_id=mlp_id(),
            step_id=0,
            microbatch_id=0,
            output_tensor=torch.randn(1, 2, 3),
        )


def test_runtime_store_adds_and_retrieves_records_by_segment_id() -> None:
    store = RuntimeRecordStore()
    attn_record = AttentionSegmentExecutionRecord(
        segment_id=attention_id(0),
        step_id=0,
        microbatch_id=0,
        input_reference="shared_attention_input",
        output_tensor=torch.randn(1, 2, 4),
    )
    mlp_record = MLPSegmentExecutionRecord(
        segment_id=mlp_id(0),
        step_id=0,
        microbatch_id=0,
        input_reference="shared_mlp_input",
        output_tensor=torch.randn(1, 2, 8),
    )

    store.add_attention_record(attn_record)
    store.add_mlp_record(mlp_record)

    assert store.get_attention_record(attention_id(0)) is attn_record
    assert store.get_mlp_record(mlp_id(0)) is mlp_record
    assert store.num_execution_records == 2


def test_runtime_store_rejects_duplicate_records_unless_overwrite() -> None:
    store = RuntimeRecordStore()
    first = MLPSegmentExecutionRecord(
        segment_id=mlp_id(1),
        step_id=0,
        microbatch_id=0,
        input_reference="x",
        output_tensor=torch.zeros(1, 2, 8),
    )
    second = MLPSegmentExecutionRecord(
        segment_id=mlp_id(1),
        step_id=0,
        microbatch_id=0,
        input_reference="x",
        output_tensor=torch.ones(1, 2, 8),
    )

    store.add_mlp_record(first)
    with pytest.raises(KeyError):
        store.add_mlp_record(second)

    store.add_mlp_record(second, overwrite=True)
    assert torch.equal(store.get_mlp_record(mlp_id(1)).output_tensor, torch.ones(1, 2, 8))


def test_shared_input_reference_works() -> None:
    store = RuntimeRecordStore()
    shared = torch.randn(2, 3, 8, requires_grad=True)
    store.add_shared_tensor("layer_0.attention_input", shared)

    for idx in range(3):
        store.add_attention_record(
            AttentionSegmentExecutionRecord(
                segment_id=attention_id(idx),
                step_id=0,
                microbatch_id=0,
                input_reference="layer_0.attention_input",
                output_tensor=torch.randn(2, 3, 4),
            )
        )

    assert torch.equal(store.get_shared_tensor("layer_0.attention_input"), shared.detach())
    assert all(
        record.input_reference == "layer_0.attention_input"
        for record in store.list_attention_records()
    )


def test_list_records_returns_deterministic_order() -> None:
    store = RuntimeRecordStore()
    for idx in [2, 0, 1]:
        store.add_attention_record(
            AttentionSegmentExecutionRecord(
                segment_id=attention_id(idx),
                step_id=0,
                microbatch_id=0,
                input_reference="x",
                output_tensor=torch.randn(1, 2, 4),
            )
        )

    assert store.list_attention_segment_ids() == [
        attention_id(0),
        attention_id(1),
        attention_id(2),
    ]


def test_composition_records_validate_attention_concat_order() -> None:
    record = AttentionCompositionRecord(
        layer_id=0,
        segment_order=(0, 1, 2),
        dropout_applied=True,
    )
    assert record.composition_type == "concat"
    assert record.output_projection_applied is True

    with pytest.raises(ValueError, match="sorted"):
        AttentionCompositionRecord(layer_id=0, segment_order=(1, 0, 2))

    with pytest.raises(ValueError, match="concat"):
        AttentionCompositionRecord(
            layer_id=0,
            segment_order=(0, 1),
            composition_type="sum",  # type: ignore[arg-type]
        )


def test_composition_records_validate_mlp_sum_order() -> None:
    record = MLPCompositionRecord(
        layer_id=1,
        segment_order=(0, 1, 2, 3),
        shared_output_bias_added_once=True,
    )
    assert record.composition_type == "sum"

    with pytest.raises(ValueError, match="duplicates"):
        MLPCompositionRecord(layer_id=1, segment_order=(0, 1, 1))

    with pytest.raises(ValueError, match="sum"):
        MLPCompositionRecord(
            layer_id=1,
            segment_order=(0, 1),
            composition_type="concat",  # type: ignore[arg-type]
        )


def test_residual_composition_record_validates_branch() -> None:
    record = ResidualCompositionRecord(layer_id=0, branch_type="attention")
    assert record.operation == "residual_addition"

    with pytest.raises(ValueError, match="branch_type"):
        ResidualCompositionRecord(layer_id=0, branch_type="bad")  # type: ignore[arg-type]


def test_gradient_records_are_segment_typed() -> None:
    attn_grad = AttentionSegmentGradientRecord(
        segment_id=attention_id(0),
        parameter_gradients={"q_proj.weight": torch.ones(3, 3)},
        input_gradient=torch.ones(1, 2, 8),
    )
    mlp_grad = MLPSegmentGradientRecord(
        segment_id=mlp_id(0),
        parameter_gradients={"fc1.weight": torch.ones(3, 3)},
        input_gradient=torch.ones(1, 2, 8),
    )

    assert attn_grad.segment_id.segment_type == "attention"
    assert mlp_grad.segment_id.segment_type == "mlp"

    with pytest.raises(ValueError, match="attention"):
        AttentionSegmentGradientRecord(
            segment_id=mlp_id(0),
            parameter_gradients={},
            input_gradient=torch.ones(1, 2, 8),
        )


def test_runtime_store_adds_and_retrieves_gradient_records() -> None:
    store = RuntimeRecordStore()
    attn = AttentionSegmentGradientRecord(
        segment_id=attention_id(1),
        parameter_gradients={"q.weight": torch.ones(2, 2)},
        input_gradient=torch.ones(1, 2, 8),
    )
    mlp = MLPSegmentGradientRecord(
        segment_id=mlp_id(1),
        parameter_gradients={"fc.weight": torch.ones(2, 2)},
        input_gradient=torch.ones(1, 2, 8),
    )

    store.add_attention_gradient_record(attn)
    store.add_mlp_gradient_record(mlp)

    assert store.get_attention_gradient_record(attention_id(1)) is attn
    assert store.get_mlp_gradient_record(mlp_id(1)) is mlp
    assert store.num_gradient_records == 2


def test_runtime_store_composition_records_and_clear() -> None:
    store = RuntimeRecordStore()
    store.add_attention_composition_record(
        AttentionCompositionRecord(layer_id=0, segment_order=(0, 1))
    )
    store.add_mlp_composition_record(
        MLPCompositionRecord(layer_id=0, segment_order=(0, 1))
    )
    store.add_residual_composition_record(
        ResidualCompositionRecord(layer_id=0, branch_type="attention")
    )

    assert store.get_attention_composition_record(0).composition_type == "concat"
    assert store.get_mlp_composition_record(0).composition_type == "sum"
    assert len(store.residual_composition_records) == 1

    store.clear()
    assert store.is_empty()


def test_detached_copy_detaches_tensor_fields() -> None:
    input_tensor = torch.randn(1, 2, 8, requires_grad=True)
    output_tensor = input_tensor * 2
    record = MLPSegmentExecutionRecord(
        segment_id=mlp_id(0),
        step_id=0,
        microbatch_id=0,
        input_tensor=input_tensor,
        output_tensor=output_tensor,
    )

    copied = record.detached_copy()
    assert copied.input_tensor is not None
    assert copied.output_tensor is not None
    assert copied.input_tensor.requires_grad is False
    assert copied.output_tensor.requires_grad is False
    assert copied.input_tensor is not input_tensor
    assert copied.output_tensor is not output_tensor
