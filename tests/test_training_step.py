"""Tests for Phase 16 segmented trainer."""

from __future__ import annotations

import pytest
import torch

from sequential_segmented_llm_training_inference.config.model_config import ModelConfig
from sequential_segmented_llm_training_inference.config.segmentation_config import SegmentationConfig
from sequential_segmented_llm_training_inference.execution.backward_engine import (
    RecomputationCheckConfig,
    SegmentedBackwardEngine,
)
from sequential_segmented_llm_training_inference.execution.forward_engine import SegmentedForwardEngine
from sequential_segmented_llm_training_inference.execution.segment_loader import StrictSegmentLoader
from sequential_segmented_llm_training_inference.optimization.segmentwise_adamw import SegmentwiseAdamW
from sequential_segmented_llm_training_inference.optimization.segmentwise_sgd import SegmentwiseSGD
from sequential_segmented_llm_training_inference.segments.segment_factory import SegmentFactory
from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId
from sequential_segmented_llm_training_inference.storage.cpu_ram_segment_store import CpuRamSegmentStore
from sequential_segmented_llm_training_inference.training.trainer import (
    SegmentedTrainer,
    SegmentTrainBatch,
)


def _build_trainer(*, update_style: str = "after_full_backward"):
    torch.manual_seed(1234)
    model_config = ModelConfig(
        vocab_size=41,
        max_seq_len=12,
        n_layers=1,
        d_model=8,
        n_heads=4,
        d_ff=16,
        dropout=0.0,
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
    if update_style == "after_full_backward":
        segment_optimizer = SegmentwiseSGD(lr=0.05)
    else:
        segment_optimizer = SegmentwiseAdamW(lr=0.01, weight_decay=0.0)
    non_segment_optimizer = torch.optim.SGD(forward_engine.parameters(), lr=0.05)
    trainer = SegmentedTrainer(
        forward_engine=forward_engine,
        backward_engine=backward_engine,
        segment_optimizer=segment_optimizer,
        non_segment_optimizer=non_segment_optimizer,
        update_style=update_style,  # type: ignore[arg-type]
    )
    return trainer, store


def _batch(vocab_size: int = 41) -> SegmentTrainBatch:
    input_ids = torch.randint(0, vocab_size, (3, 7))
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    return SegmentTrainBatch(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


def _first_segment_state(store: CpuRamSegmentStore) -> tuple[SegmentId, dict[str, torch.Tensor]]:
    segment_id = store.list_segments()[0]
    return segment_id, store.load_segment(segment_id)


def _state_changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> bool:
    return any(not torch.allclose(before[name], after[name]) for name in before)


def test_train_step_runs_and_updates_segment_and_non_segment_parameters() -> None:
    trainer, store = _build_trainer(update_style="after_full_backward")
    segment_id, before_segment = _first_segment_state(store)
    before_non_segment = next(trainer.forward_engine.parameters()).detach().clone()

    result = trainer.train_step(_batch())

    after_segment = store.load_segment(segment_id)
    after_non_segment = next(trainer.forward_engine.parameters()).detach().clone()

    assert result.loss > 0
    assert result.segment_update_count == len(store.list_segments())
    assert result.non_segment_update_applied is True
    assert result.optimizer_step_applied is True
    assert result.num_segment_records == len(store.list_segments())
    assert _state_changed(before_segment, after_segment)
    assert not torch.allclose(before_non_segment, after_non_segment)
    assert trainer.segment_loader.active_segment_count == 0


def test_immediate_segment_update_runs_and_updates_segments() -> None:
    trainer, store = _build_trainer(update_style="immediate_segment_update")
    segment_id, before_segment = _first_segment_state(store)

    result = trainer.train_step(_batch())

    after_segment = store.load_segment(segment_id)
    assert result.loss > 0
    assert result.segment_update_count == len(store.list_segments())
    assert _state_changed(before_segment, after_segment)
    assert trainer.segment_loader.active_segment_count == 0


def test_train_epoch_aggregates_losses() -> None:
    trainer, _ = _build_trainer()
    batches = [_batch(), _batch()]

    result = trainer.train_epoch(batches, epoch=2)

    assert result.epoch == 2
    assert result.steps == 2
    assert result.optimizer_steps == 2
    assert result.mean_loss > 0


def test_train_step_accepts_mapping_batch() -> None:
    trainer, _ = _build_trainer()
    batch = _batch()

    result = trainer.train_step(
        {
            "input_ids": batch.input_ids,
            "labels": batch.labels,
            "attention_mask": batch.attention_mask,
        }
    )

    assert result.loss > 0


def test_trainer_rejects_immediate_update_with_gradient_accumulation() -> None:
    trainer, _ = _build_trainer()
    with pytest.raises(ValueError, match="immediate_segment_update"):
        SegmentedTrainer(
            forward_engine=trainer.forward_engine,
            backward_engine=trainer.backward_engine,
            segment_optimizer=SegmentwiseSGD(lr=0.1),
            update_style="immediate_segment_update",
            gradient_accumulation_steps=2,
        )


def test_batch_validation_rejects_missing_label_shape() -> None:
    batch = SegmentTrainBatch(
        input_ids=torch.ones(2, 4, dtype=torch.long),
        labels=torch.ones(2, 3, dtype=torch.long),
    )
    with pytest.raises(ValueError, match="labels"):
        batch.validate()


def test_train_epoch_rejects_empty_iterable() -> None:
    trainer, _ = _build_trainer()
    with pytest.raises(ValueError, match="no batches"):
        trainer.train_epoch([])
