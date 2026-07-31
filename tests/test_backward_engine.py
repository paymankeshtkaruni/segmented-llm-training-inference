"""Tests for Phase 12 recomputation-based segmented backward engine."""

from __future__ import annotations

import pytest
import torch

from sequential_segmented_llm_training_inference.config.model_config import ModelConfig
from sequential_segmented_llm_training_inference.config.segmentation_config import (
    SegmentationConfig,
)
from sequential_segmented_llm_training_inference.execution.backward_engine import (
    RecomputationCheckConfig,
    RecomputationMismatchError,
    SegmentedBackwardEngine,
)
from sequential_segmented_llm_training_inference.execution.segment_loader import (
    StrictSegmentLoader,
)
from sequential_segmented_llm_training_inference.segments.attention_segment import (
    AttentionHeadSegment,
)
from sequential_segmented_llm_training_inference.segments.mlp_segment import MLPHiddenSegment
from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId
from sequential_segmented_llm_training_inference.storage.cpu_ram_segment_store import (
    CpuRamSegmentStore,
)
from sequential_segmented_llm_training_inference.storage.runtime_records import (
    AttentionSegmentExecutionRecord,
    MLPSegmentExecutionRecord,
    RuntimeRecordStore,
)


def _configs() -> tuple[ModelConfig, SegmentationConfig]:
    return (
        ModelConfig(
            vocab_size=31,
            max_seq_len=8,
            n_layers=2,
            d_model=8,
            n_heads=4,
            d_ff=16,
            dropout=0.0,
        ),
        SegmentationConfig(attention_segments=2, mlp_chunks=4),
    )


def _module_factory(model: ModelConfig, segmentation: SegmentationConfig):
    def factory(segment_id: SegmentId):
        if segment_id.segment_type == "attention":
            return AttentionHeadSegment(
                segment_id=segment_id,
                d_model=model.d_model,
                n_heads=model.n_heads,
                attention_segments=segmentation.attention_segments,
                dropout=0.0,
            )
        return MLPHiddenSegment(
            segment_id=segment_id,
            d_model=model.d_model,
            d_ff=model.d_ff,
            mlp_chunks=segmentation.mlp_chunks,
            activation="gelu",
        )

    return factory


def _store_segments(model: ModelConfig, segmentation: SegmentationConfig) -> CpuRamSegmentStore:
    store = CpuRamSegmentStore()
    factory = _module_factory(model, segmentation)
    torch.manual_seed(123)
    for layer_id in range(model.n_layers):
        for index in range(segmentation.attention_segments):
            sid = SegmentId(layer_id, "attention", index)
            store.save_segment(sid, factory(sid).state_dict())
        for index in range(segmentation.mlp_chunks):
            sid = SegmentId(layer_id, "mlp", index)
            store.save_segment(sid, factory(sid).state_dict())
    return store


def _engine() -> tuple[SegmentedBackwardEngine, CpuRamSegmentStore]:
    model, segmentation = _configs()
    store = _store_segments(model, segmentation)
    loader = StrictSegmentLoader(
        segment_store=store,
        module_factory=_module_factory(model, segmentation),
        device="cpu",
    )
    engine = SegmentedBackwardEngine(
        model_config=model,
        segmentation_config=segmentation,
        segment_loader=loader,
    )
    return engine, store


def _forward_mlp_records(
    engine: SegmentedBackwardEngine,
    layer_id: int = 0,
) -> tuple[RuntimeRecordStore, torch.Tensor]:
    records = RuntimeRecordStore()
    mlp_input = torch.randn(2, 3, engine.model_config.d_model)
    records.add_shared_tensor("layer_0.mlp_input", mlp_input, detach=True, clone=True)

    for index in range(engine.segmentation_config.mlp_chunks):
        sid = SegmentId(layer_id, "mlp", index)
        with engine.segment_loader.acquire_segment(sid) as segment:
            output = segment(mlp_input)
        records.add_mlp_record(
            MLPSegmentExecutionRecord(
                segment_id=sid,
                step_id=0,
                microbatch_id=0,
                input_reference="layer_0.mlp_input",
                output_tensor=output.detach().clone(),
            )
        )
    return records, mlp_input


def _forward_attention_records(
    engine: SegmentedBackwardEngine,
    layer_id: int = 0,
) -> tuple[RuntimeRecordStore, torch.Tensor]:
    records = RuntimeRecordStore()
    attention_input = torch.randn(2, 3, engine.model_config.d_model)
    records.add_shared_tensor("layer_0.attention_input", attention_input, detach=True, clone=True)

    for index in range(engine.segmentation_config.attention_segments):
        sid = SegmentId(layer_id, "attention", index)
        with engine.segment_loader.acquire_segment(sid) as segment:
            output = segment(attention_input, None)
        records.add_attention_record(
            AttentionSegmentExecutionRecord(
                segment_id=sid,
                step_id=0,
                microbatch_id=0,
                input_reference="layer_0.attention_input",
                output_tensor=output.detach().clone(),
            )
        )
    return records, attention_input


def test_backward_mlp_layer_produces_gradient_records() -> None:
    engine, _ = _engine()
    records, mlp_input = _forward_mlp_records(engine)
    grad_output = torch.randn_like(mlp_input)

    result = engine.backward_mlp_layer(
        layer_id=0,
        runtime_records=records,
        grad_mlp_output=grad_output,
    )

    assert result.branch_type == "mlp"
    assert result.input_gradient.shape == mlp_input.shape
    assert len(result.segment_results) == engine.segmentation_config.mlp_chunks
    assert len(records.mlp_gradient_records) == engine.segmentation_config.mlp_chunks
    assert engine.segment_loader.active_segment_count == 0

    for segment_result in result.segment_results:
        assert segment_result.recomputation_matched is True
        assert segment_result.input_gradient.shape == mlp_input.shape
        assert {"fc1.weight", "fc1.bias", "fc2.weight"}.issubset(
            set(segment_result.parameter_gradients)
        )


def test_backward_mlp_layer_sums_segment_input_gradients() -> None:
    engine, _ = _engine()
    records, mlp_input = _forward_mlp_records(engine)
    grad_output = torch.ones_like(mlp_input)

    result = engine.backward_mlp_layer(
        layer_id=0,
        runtime_records=records,
        grad_mlp_output=grad_output,
    )

    manual_sum = sum(
        segment_result.input_gradient for segment_result in result.segment_results
    )
    assert torch.allclose(result.input_gradient, manual_sum)


def test_backward_mlp_recomputation_mismatch_raises() -> None:
    engine, _ = _engine()
    records, _ = _forward_mlp_records(engine)
    sid = SegmentId(0, "mlp", 0)
    records.mlp_records[sid].output_tensor = records.mlp_records[sid].output_tensor + 100.0
    grad_output = torch.randn(2, 3, engine.model_config.d_model)

    with pytest.raises(RecomputationMismatchError):
        engine.backward_mlp_layer(
            layer_id=0,
            runtime_records=records,
            grad_mlp_output=grad_output,
        )


def test_backward_mlp_recomputation_mismatch_can_be_non_raising() -> None:
    model, segmentation = _configs()
    store = _store_segments(model, segmentation)
    loader = StrictSegmentLoader(
        segment_store=store,
        module_factory=_module_factory(model, segmentation),
        device="cpu",
    )
    engine = SegmentedBackwardEngine(
        model_config=model,
        segmentation_config=segmentation,
        segment_loader=loader,
        recomputation_check=RecomputationCheckConfig(raise_on_mismatch=False),
    )
    records, _ = _forward_mlp_records(engine)
    sid = SegmentId(0, "mlp", 0)
    records.mlp_records[sid].output_tensor = records.mlp_records[sid].output_tensor + 10.0
    grad_output = torch.randn(2, 3, model.d_model)

    result = engine.backward_mlp_layer(
        layer_id=0,
        runtime_records=records,
        grad_mlp_output=grad_output,
    )

    assert result.segment_results[0].recomputation_matched is False
    assert result.segment_results[0].max_abs_diff > 0.0


def test_backward_attention_layer_from_concat_gradient_produces_records() -> None:
    engine, _ = _engine()
    records, attention_input = _forward_attention_records(engine)
    grad_concat = torch.randn_like(attention_input)

    result = engine.backward_attention_layer_from_concat_gradient(
        layer_id=0,
        runtime_records=records,
        grad_attention_concat=grad_concat,
    )

    assert result.branch_type == "attention"
    assert result.input_gradient.shape == attention_input.shape
    assert len(result.segment_results) == engine.segmentation_config.attention_segments
    assert len(records.attention_gradient_records) == engine.segmentation_config.attention_segments
    assert engine.segment_loader.active_segment_count == 0

    for segment_result in result.segment_results:
        assert segment_result.recomputation_matched is True
        assert segment_result.input_gradient.shape == attention_input.shape
        assert {"q_proj.weight", "k_proj.weight", "v_proj.weight"}.issubset(
            set(segment_result.parameter_gradients)
        )


def test_backward_attention_input_gradient_is_sum_of_segments() -> None:
    engine, _ = _engine()
    records, attention_input = _forward_attention_records(engine)
    grad_concat = torch.ones_like(attention_input)

    result = engine.backward_attention_layer_from_concat_gradient(
        layer_id=0,
        runtime_records=records,
        grad_attention_concat=grad_concat,
    )

    manual_sum = sum(
        segment_result.input_gradient for segment_result in result.segment_results
    )
    assert torch.allclose(result.input_gradient, manual_sum)


def test_backward_attention_from_explicit_segment_gradients() -> None:
    engine, _ = _engine()
    records, attention_input = _forward_attention_records(engine)
    grad_segment_outputs = {
        record.segment_id: torch.randn_like(record.output_tensor)
        for record in records.list_attention_records()
    }

    result = engine.backward_attention_layer_from_segment_gradients(
        layer_id=0,
        runtime_records=records,
        grad_segment_outputs=grad_segment_outputs,
    )

    assert result.input_gradient.shape == attention_input.shape
    assert len(result.segment_results) == engine.segmentation_config.attention_segments


def test_backward_attention_rejects_bad_concat_gradient_shape() -> None:
    engine, _ = _engine()
    records, _ = _forward_attention_records(engine)
    bad_grad = torch.randn(2, 3, engine.model_config.d_model + 1)

    with pytest.raises(ValueError, match="d_model"):
        engine.backward_attention_layer_from_concat_gradient(
            layer_id=0,
            runtime_records=records,
            grad_attention_concat=bad_grad,
        )


def test_backward_mlp_rejects_bad_gradient_shape() -> None:
    engine, _ = _engine()
    records, _ = _forward_mlp_records(engine)
    bad_grad = torch.randn(2, 3, engine.model_config.d_model + 1)

    with pytest.raises(ValueError, match="d_model"):
        engine.backward_mlp_layer(
            layer_id=0,
            runtime_records=records,
            grad_mlp_output=bad_grad,
        )


def test_backward_rejects_invalid_layer_id() -> None:
    engine, _ = _engine()
    records, mlp_input = _forward_mlp_records(engine)

    with pytest.raises(ValueError, match="layer_id"):
        engine.backward_mlp_layer(
            layer_id=99,
            runtime_records=records,
            grad_mlp_output=torch.randn_like(mlp_input),
        )


def test_missing_attention_segment_gradient_raises_key_error() -> None:
    engine, _ = _engine()
    records, _ = _forward_attention_records(engine)
    only_first = {
        records.list_attention_records()[0].segment_id: torch.randn_like(
            records.list_attention_records()[0].output_tensor
        )
    }

    with pytest.raises(KeyError, match="Missing gradient"):
        engine.backward_attention_layer_from_segment_gradients(
            layer_id=0,
            runtime_records=records,
            grad_segment_outputs=only_first,
        )


def test_backward_engine_requires_segment_loader() -> None:
    model, segmentation = _configs()
    with pytest.raises(ValueError, match="segment_loader"):
        SegmentedBackwardEngine(
            model_config=model,
            segmentation_config=segmentation,
            segment_loader=None,  # type: ignore[arg-type]
        )
