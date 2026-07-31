"""Tests for Phase 10 segmented forward engine."""

from __future__ import annotations

import pytest
import torch

from sequential_segmented_llm_training_inference.config.model_config import ModelConfig
from sequential_segmented_llm_training_inference.config.segmentation_config import SegmentationConfig
from sequential_segmented_llm_training_inference.execution.composition import (
    concatenate_attention_outputs,
    residual_add,
    sum_mlp_outputs,
)
from sequential_segmented_llm_training_inference.execution.forward_engine import (
    SegmentedForwardEngine,
)
from sequential_segmented_llm_training_inference.execution.segment_loader import StrictSegmentLoader
from sequential_segmented_llm_training_inference.segments.segment_factory import SegmentFactory
from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId
from sequential_segmented_llm_training_inference.storage.cpu_ram_segment_store import (
    CpuRamSegmentStore,
)


def _build_engine(*, dropout: float = 0.0) -> tuple[SegmentedForwardEngine, CpuRamSegmentStore]:
    torch.manual_seed(123)
    model_config = ModelConfig(
        vocab_size=64,
        max_seq_len=16,
        n_layers=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        dropout=dropout,
    )
    segmentation_config = SegmentationConfig(attention_segments=2, mlp_chunks=4)
    factory = SegmentFactory(
        model_config=model_config,
        segmentation_config=segmentation_config,
        attention_dropout=0.0,
    )
    collection = factory.create_all_segments()
    store = CpuRamSegmentStore()
    for segment_id, module in collection.iter_segments():
        store.save_segment(segment_id, module.state_dict())

    def module_factory(segment_id: SegmentId):
        if segment_id.segment_type == "attention":
            return factory.create_attention_segment(
                layer_id=segment_id.layer_id,
                segment_index=segment_id.segment_id,
            )
        return factory.create_mlp_segment(
            layer_id=segment_id.layer_id,
            segment_index=segment_id.segment_id,
        )

    loader = StrictSegmentLoader(
        segment_store=store,
        module_factory=module_factory,
        device="cpu",
    )
    engine = SegmentedForwardEngine(
        model_config=model_config,
        segmentation_config=segmentation_config,
        segment_loader=loader,
        residual_dropout=dropout,
    )
    return engine, store


def test_forward_returns_expected_logits_shape() -> None:
    engine, _ = _build_engine()
    input_ids = torch.randint(0, engine.model_config.vocab_size, (3, 7))
    attention_mask = torch.ones_like(input_ids)

    output = engine(input_ids, attention_mask=attention_mask)

    assert output.logits.shape == (3, 7, engine.model_config.vocab_size)
    assert output.hidden_states.shape == (3, 7, engine.model_config.d_model)
    assert engine.segment_loader.active_segment_count == 0


def test_forward_creates_segment_level_runtime_records() -> None:
    engine, _ = _build_engine()
    input_ids = torch.randint(0, engine.model_config.vocab_size, (2, 5))

    output = engine(input_ids, store_records=True, step_id=9, microbatch_id=2)
    records = output.runtime_records

    assert records is not None
    expected_attention = engine.model_config.n_layers * engine.segmentation_config.attention_segments
    expected_mlp = engine.model_config.n_layers * engine.segmentation_config.mlp_chunks
    assert len(records.attention_records) == expected_attention
    assert len(records.mlp_records) == expected_mlp
    assert records.num_execution_records == expected_attention + expected_mlp

    first_attention_id = SegmentId(0, "attention", 0)
    first_mlp_id = SegmentId(0, "mlp", 0)
    assert records.get_attention_record(first_attention_id).step_id == 9
    assert records.get_attention_record(first_attention_id).microbatch_id == 2
    assert records.get_mlp_record(first_mlp_id).step_id == 9
    assert records.get_mlp_record(first_mlp_id).microbatch_id == 2


def test_forward_records_composition_metadata() -> None:
    engine, _ = _build_engine()
    input_ids = torch.randint(0, engine.model_config.vocab_size, (2, 5))

    output = engine(input_ids, store_records=True)
    records = output.runtime_records
    assert records is not None

    for layer_id in range(engine.model_config.n_layers):
        attention_record = records.get_attention_composition_record(layer_id)
        assert attention_record.composition_type == "concat"
        assert attention_record.segment_order == tuple(
            range(engine.segmentation_config.attention_segments)
        )
        assert attention_record.output_projection_applied is True

        mlp_record = records.get_mlp_composition_record(layer_id)
        assert mlp_record.composition_type == "sum"
        assert mlp_record.segment_order == tuple(range(engine.segmentation_config.mlp_chunks))
        assert mlp_record.shared_output_bias_added_once is True

    assert len(records.residual_composition_records) == engine.model_config.n_layers * 2


def test_forward_can_skip_runtime_records() -> None:
    engine, _ = _build_engine()
    input_ids = torch.randint(0, engine.model_config.vocab_size, (2, 5))

    output = engine(input_ids, store_records=False)

    assert output.runtime_records is None
    assert output.logits.shape == (2, 5, engine.model_config.vocab_size)


def test_forward_can_return_segment_outputs_for_debugging() -> None:
    engine, _ = _build_engine()
    input_ids = torch.randint(0, engine.model_config.vocab_size, (2, 6))

    output = engine(input_ids, return_segment_outputs=True)

    expected_attention = engine.model_config.n_layers * engine.segmentation_config.attention_segments
    expected_mlp = engine.model_config.n_layers * engine.segmentation_config.mlp_chunks
    assert len(output.attention_segment_outputs) == expected_attention
    assert len(output.mlp_segment_outputs) == expected_mlp

    attention_output_dim = engine.model_config.d_model // engine.segmentation_config.attention_segments
    first_attention_output = output.attention_segment_outputs[SegmentId(0, "attention", 0)]
    assert first_attention_output.shape == (2, 6, attention_output_dim)

    first_mlp_output = output.mlp_segment_outputs[SegmentId(0, "mlp", 0)]
    assert first_mlp_output.shape == (2, 6, engine.model_config.d_model)


def test_forward_uses_strict_loader_for_every_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = _build_engine()
    calls: list[SegmentId] = []
    original_acquire = engine.segment_loader.acquire_segment

    def wrapped_acquire(segment_id: SegmentId, *, save_on_exit: bool = False):
        assert engine.segment_loader.active_segment_count == 0
        calls.append(segment_id)
        return original_acquire(segment_id, save_on_exit=save_on_exit)

    monkeypatch.setattr(engine.segment_loader, "acquire_segment", wrapped_acquire)
    input_ids = torch.randint(0, engine.model_config.vocab_size, (1, 4))
    engine(input_ids)

    expected_calls = engine.model_config.n_layers * (
        engine.segmentation_config.attention_segments + engine.segmentation_config.mlp_chunks
    )
    assert len(calls) == expected_calls
    assert engine.segment_loader.active_segment_count == 0


def test_forward_rejects_invalid_input_shape() -> None:
    engine, _ = _build_engine()
    with pytest.raises(ValueError, match="input_ids must have shape"):
        engine(torch.randint(0, 10, (2, 3, 4)))


def test_forward_rejects_invalid_2d_attention_mask_shape() -> None:
    engine, _ = _build_engine()
    input_ids = torch.randint(0, engine.model_config.vocab_size, (2, 5))
    bad_mask = torch.ones(2, 4)
    with pytest.raises(ValueError, match="2-D attention_mask"):
        engine(input_ids, attention_mask=bad_mask)


def test_composition_concatenates_attention_outputs() -> None:
    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 3, 4)
    result = concatenate_attention_outputs([a, b])
    assert result.shape == (2, 3, 8)
    assert torch.equal(result[..., :4], a)
    assert torch.equal(result[..., 4:], b)


def test_composition_sums_mlp_outputs() -> None:
    a = torch.ones(2, 3, 4)
    b = torch.full((2, 3, 4), 2.0)
    result = sum_mlp_outputs([a, b])
    assert torch.equal(result, torch.full((2, 3, 4), 3.0))


def test_residual_add_validates_shape() -> None:
    with pytest.raises(ValueError, match="Residual addition requires"):
        residual_add(torch.randn(2, 3, 4), torch.randn(2, 3, 5))
