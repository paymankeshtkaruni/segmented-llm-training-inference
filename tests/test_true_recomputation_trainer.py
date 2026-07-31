"""Regression tests for true recomputation training.

These tests guard the architectural contract that training must not use the
original full-model autograd graph.
"""

from __future__ import annotations

import torch

from sequential_segmented_llm_training_inference.config.model_config import ModelConfig
from sequential_segmented_llm_training_inference.config.segmentation_config import SegmentationConfig
from sequential_segmented_llm_training_inference.execution.backward_engine import (
    RecomputationCheckConfig,
    SegmentedBackwardEngine,
)
from sequential_segmented_llm_training_inference.execution.forward_engine import SegmentedForwardEngine
from sequential_segmented_llm_training_inference.execution.segment_loader import StrictSegmentLoader
from sequential_segmented_llm_training_inference.optimization.segmentwise_sgd import SegmentwiseSGD
from sequential_segmented_llm_training_inference.segments.segment_factory import SegmentFactory
from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId
from sequential_segmented_llm_training_inference.storage.cpu_ram_segment_store import CpuRamSegmentStore
from sequential_segmented_llm_training_inference.training.trainer import SegmentTrainBatch, SegmentedTrainer


def _build_true_recompute_trainer() -> tuple[SegmentedTrainer, CpuRamSegmentStore]:
    torch.manual_seed(777)
    model_config = ModelConfig(
        vocab_size=31,
        max_seq_len=8,
        n_layers=1,
        d_model=8,
        n_heads=4,
        d_ff=16,
        dropout=0.0,
    )
    segmentation_config = SegmentationConfig(attention_segments=2, mlp_chunks=2)
    factory = SegmentFactory(
        model_config=model_config,
        segmentation_config=segmentation_config,
        attention_dropout=0.0,
    )
    store = CpuRamSegmentStore()
    for segment_id, module in factory.create_all_segments().iter_segments():
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

    loader = StrictSegmentLoader(segment_store=store, module_factory=module_factory, device="cpu")
    forward_engine = SegmentedForwardEngine(
        model_config=model_config,
        segmentation_config=segmentation_config,
        segment_loader=loader,
        residual_dropout=0.0,
    )
    backward_engine = SegmentedBackwardEngine(
        model_config=model_config,
        segmentation_config=segmentation_config,
        segment_loader=loader,
        recomputation_check=RecomputationCheckConfig(enabled=True),
    )
    trainer = SegmentedTrainer(
        forward_engine=forward_engine,
        backward_engine=backward_engine,
        segment_optimizer=SegmentwiseSGD(lr=0.01),
        non_segment_optimizer=torch.optim.SGD(forward_engine.parameters(), lr=0.01),
        update_style="after_full_backward",
    )
    return trainer, store


def test_training_does_not_call_tensor_backward(monkeypatch) -> None:
    trainer, _ = _build_true_recompute_trainer()
    input_ids = torch.randint(0, trainer.forward_engine.model_config.vocab_size, (2, 5))
    batch = SegmentTrainBatch(
        input_ids=input_ids,
        labels=input_ids.clone(),
        attention_mask=torch.ones_like(input_ids),
    )

    def forbidden_backward(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError("global Tensor.backward() must not be used")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden_backward, raising=False)
    result = trainer.train_step(batch)

    assert result.loss > 0
    assert result.metadata["backward_mode"] == "true_recomputation_local_autograd"
    assert result.metadata["full_forward_backward_used"] is False


def test_forward_records_for_recompute_are_detached_and_cpu_offloaded() -> None:
    trainer, _ = _build_true_recompute_trainer()
    input_ids = torch.randint(0, trainer.forward_engine.model_config.vocab_size, (2, 5))
    with torch.no_grad():
        output = trainer.forward_engine(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            store_records=True,
            detach_record_tensors=True,
            clone_record_tensors=True,
            record_tensors_to_cpu=True,
        )
    records = output.runtime_records
    assert records is not None
    for tensor in records.shared_tensors.values():
        assert tensor.grad_fn is None
        assert tensor.device.type == "cpu"
    for record in records.attention_records.values():
        assert record.output_tensor.grad_fn is None
        assert record.output_tensor.device.type == "cpu"
    for record in records.mlp_records.values():
        assert record.output_tensor.grad_fn is None
        assert record.output_tensor.device.type == "cpu"


def test_true_recomputation_with_dropout_enabled_matches_rng_records() -> None:
    torch.manual_seed(2026)
    model_config = ModelConfig(
        vocab_size=37,
        max_seq_len=8,
        n_layers=1,
        d_model=8,
        n_heads=4,
        d_ff=16,
        dropout=0.2,
    )
    segmentation_config = SegmentationConfig(attention_segments=2, mlp_chunks=2)
    factory = SegmentFactory(
        model_config=model_config,
        segmentation_config=segmentation_config,
        attention_dropout=model_config.dropout,
    )
    store = CpuRamSegmentStore()
    for segment_id, module in factory.create_all_segments().iter_segments():
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

    loader = StrictSegmentLoader(segment_store=store, module_factory=module_factory, device="cpu")
    forward_engine = SegmentedForwardEngine(
        model_config=model_config,
        segmentation_config=segmentation_config,
        segment_loader=loader,
        residual_dropout=model_config.dropout,
    )
    backward_engine = SegmentedBackwardEngine(
        model_config=model_config,
        segmentation_config=segmentation_config,
        segment_loader=loader,
        recomputation_check=RecomputationCheckConfig(enabled=True, raise_on_mismatch=True),
    )
    trainer = SegmentedTrainer(
        forward_engine=forward_engine,
        backward_engine=backward_engine,
        segment_optimizer=SegmentwiseSGD(lr=0.001),
        non_segment_optimizer=torch.optim.SGD(forward_engine.parameters(), lr=0.001),
        update_style="after_full_backward",
    )

    input_ids = torch.randint(0, model_config.vocab_size, (2, 6))
    result = trainer.train_step(
        SegmentTrainBatch(
            input_ids=input_ids,
            labels=input_ids.clone(),
            attention_mask=torch.ones_like(input_ids),
        )
    )

    assert result.loss > 0
    assert trainer.segment_loader.active_segment_count == 0


def test_true_global_gradient_clipping_rejects_immediate_update() -> None:
    trainer, _ = _build_true_recompute_trainer()
    import pytest

    with pytest.raises(ValueError, match="global gradient clipping"):
        SegmentedTrainer(
            forward_engine=trainer.forward_engine,
            backward_engine=trainer.backward_engine,
            segment_optimizer=SegmentwiseSGD(lr=0.01),
            update_style="immediate_segment_update",
            gradient_clip_norm=1.0,
        )
