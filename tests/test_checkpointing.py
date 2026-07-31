"""Tests for Phase 15 segmented checkpointing."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId
from sequential_segmented_llm_training_inference.storage.checkpoint_store import (
    SegmentedCheckpoint,
    SegmentedCheckpointStore,
)
from sequential_segmented_llm_training_inference.storage.manifest import (
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    CheckpointIndex,
    CheckpointManifest,
)


def make_checkpoint(epoch: int = 1, step: int = 10) -> SegmentedCheckpoint:
    return SegmentedCheckpoint(
        epoch=epoch,
        global_step=step,
        segment_states={
            SegmentId(0, "attention", 0): {
                "q_proj.weight": torch.ones(2, 2),
                "k_proj.weight": torch.full((2, 2), 2.0),
            },
            SegmentId(0, "mlp", 1): {
                "fc1.weight": torch.arange(6, dtype=torch.float32).view(2, 3),
            },
        },
        non_segment_state={"lm_head.weight": torch.eye(3)},
        optimizer_state={"step": step, "state": {"x": torch.tensor([1.0])}},
        scheduler_state={"last_epoch": epoch},
        rng_state={"torch": torch.get_rng_state()},
        model_config={"d_model": 4, "n_heads": 2},
        segmentation_config={"attention_segments": 2, "mlp_chunks": 2},
        training_config={"update_style": "after_full_backward"},
        validation_metadata={"validation_loss": 1.25},
        metadata={"note": "unit test"},
    )


def test_checkpoint_requires_non_empty_segments() -> None:
    with pytest.raises(ValueError, match="segment_states"):
        SegmentedCheckpoint(epoch=0, global_step=0, segment_states={})


def test_save_checkpoint_writes_manifest_and_segment_files(tmp_path: Path) -> None:
    store = SegmentedCheckpointStore(tmp_path)
    manifest = store.save_checkpoint(
        checkpoint_name="custom_checkpoint",
        checkpoint=make_checkpoint(),
    )

    checkpoint_dir = tmp_path / "custom_checkpoint"
    assert (checkpoint_dir / MANIFEST_FILENAME).exists()
    assert len(manifest.segment_entries) == 2
    for entry in manifest.segment_entries:
        assert (checkpoint_dir / entry.relative_path).exists()
        assert entry.sha256
        assert entry.num_tensors > 0
        assert entry.num_parameters > 0


def test_load_checkpoint_restores_payload(tmp_path: Path) -> None:
    store = SegmentedCheckpointStore(tmp_path)
    original = make_checkpoint(epoch=2, step=25)
    store.save_checkpoint(checkpoint_name="checkpoint_a", checkpoint=original)

    loaded = store.load_checkpoint("checkpoint_a")
    restored = loaded.checkpoint

    assert restored.epoch == 2
    assert restored.global_step == 25
    assert restored.model_config == original.model_config
    assert restored.segmentation_config == original.segmentation_config
    assert restored.training_config == original.training_config
    assert restored.validation_metadata == original.validation_metadata
    assert restored.metadata == original.metadata
    assert set(restored.segment_states) == set(original.segment_states)
    assert torch.equal(
        restored.segment_states[SegmentId(0, "attention", 0)]["q_proj.weight"],
        original.segment_states[SegmentId(0, "attention", 0)]["q_proj.weight"],
    )
    assert torch.equal(
        restored.non_segment_state["lm_head.weight"],
        original.non_segment_state["lm_head.weight"],
    )
    assert restored.optimizer_state is not None
    assert restored.scheduler_state is not None
    assert restored.rng_state is not None


def test_hash_mismatch_is_detected(tmp_path: Path) -> None:
    store = SegmentedCheckpointStore(tmp_path)
    manifest = store.save_checkpoint(
        checkpoint_name="checkpoint_hash",
        checkpoint=make_checkpoint(),
    )
    first_segment = manifest.segment_entries[0]
    segment_path = tmp_path / "checkpoint_hash" / first_segment.relative_path
    segment_path.write_bytes(b"corrupted")

    with pytest.raises(ValueError, match="Hash mismatch"):
        store.load_checkpoint("checkpoint_hash")


def test_manifest_yaml_roundtrip(tmp_path: Path) -> None:
    store = SegmentedCheckpointStore(tmp_path)
    manifest = store.save_checkpoint(
        checkpoint_name="checkpoint_manifest",
        checkpoint=make_checkpoint(),
    )
    manifest_path = tmp_path / "checkpoint_manifest" / MANIFEST_FILENAME

    loaded_manifest = CheckpointManifest.load_yaml(manifest_path)

    assert loaded_manifest.checkpoint_name == manifest.checkpoint_name
    assert loaded_manifest.epoch == manifest.epoch
    assert loaded_manifest.global_step == manifest.global_step
    assert loaded_manifest.segment_entries[0].segment_id == manifest.segment_entries[0].segment_id


def test_save_last_checkpoint_updates_index(tmp_path: Path) -> None:
    store = SegmentedCheckpointStore(tmp_path)
    store.save_last_checkpoint(make_checkpoint(epoch=3, step=30))

    index = CheckpointIndex.load_yaml(tmp_path / INDEX_FILENAME)
    assert index.last_checkpoint == "last_segmented_checkpoint"

    loaded = store.load_last_checkpoint()
    assert loaded.checkpoint.epoch == 3
    assert loaded.checkpoint.global_step == 30


def test_save_best_checkpoint_uses_lower_is_better(tmp_path: Path) -> None:
    store = SegmentedCheckpointStore(tmp_path)

    first = store.save_best_checkpoint(
        make_checkpoint(epoch=1, step=10),
        metric_name="validation_loss",
        metric_value=1.0,
        lower_is_better=True,
    )
    worse = store.save_best_checkpoint(
        make_checkpoint(epoch=2, step=20),
        metric_name="validation_loss",
        metric_value=1.2,
        lower_is_better=True,
    )
    better = store.save_best_checkpoint(
        make_checkpoint(epoch=3, step=30),
        metric_name="validation_loss",
        metric_value=0.8,
        lower_is_better=True,
    )

    assert first is not None
    assert worse is None
    assert better is not None

    index = store.load_index()
    assert index.best_checkpoint == "best_segmented_checkpoint"
    assert index.best_metric_name == "validation_loss"
    assert index.best_metric_value == 0.8

    loaded = store.load_best_checkpoint()
    assert loaded.checkpoint.epoch == 3
    assert loaded.checkpoint.validation_metadata["validation_loss"] == 0.8


def test_save_best_checkpoint_uses_higher_is_better(tmp_path: Path) -> None:
    store = SegmentedCheckpointStore(tmp_path)

    first = store.save_best_checkpoint(
        make_checkpoint(epoch=1, step=10),
        metric_name="accuracy",
        metric_value=0.7,
        lower_is_better=False,
    )
    worse = store.save_best_checkpoint(
        make_checkpoint(epoch=2, step=20),
        metric_name="accuracy",
        metric_value=0.6,
        lower_is_better=False,
    )
    better = store.save_best_checkpoint(
        make_checkpoint(epoch=3, step=30),
        metric_name="accuracy",
        metric_value=0.9,
        lower_is_better=False,
    )

    assert first is not None
    assert worse is None
    assert better is not None
    assert store.load_index().best_metric_value == 0.9
    assert store.load_best_checkpoint().checkpoint.epoch == 3


def test_loading_missing_last_checkpoint_raises(tmp_path: Path) -> None:
    store = SegmentedCheckpointStore(tmp_path)
    with pytest.raises(FileNotFoundError, match="last checkpoint"):
        store.load_last_checkpoint()


def test_loading_missing_best_checkpoint_raises(tmp_path: Path) -> None:
    store = SegmentedCheckpointStore(tmp_path)
    with pytest.raises(FileNotFoundError, match="best checkpoint"):
        store.load_best_checkpoint()


def test_save_without_overwrite_rejects_existing_checkpoint(tmp_path: Path) -> None:
    store = SegmentedCheckpointStore(tmp_path)
    store.save_checkpoint(checkpoint_name="same", checkpoint=make_checkpoint())

    with pytest.raises(FileExistsError):
        store.save_checkpoint(
            checkpoint_name="same",
            checkpoint=make_checkpoint(),
            overwrite=False,
        )
