"""Segmented checkpoint save/load logic.

Phase 15 scope:
- Save and load segmented checkpoints.
- Store segment state dictionaries separately under deterministic paths.
- Store non-segment, optimizer, scheduler, and RNG state files when supplied.
- Maintain last and best segmented checkpoint entries.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId
from sequential_segmented_llm_training_inference.storage.manifest import (
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    CheckpointIndex,
    CheckpointManifest,
    CheckpointSegmentEntry,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_tensors_and_parameters(state: Mapping[str, Any]) -> tuple[int, int]:
    num_tensors = 0
    num_parameters = 0
    for value in state.values():
        if isinstance(value, Tensor):
            num_tensors += 1
            num_parameters += value.numel()
    return num_tensors, num_parameters


@dataclass(slots=True)
class SegmentedCheckpoint:
    """In-memory checkpoint payload for a segmented model."""

    epoch: int
    global_step: int
    segment_states: dict[SegmentId, dict[str, Any]]
    non_segment_state: dict[str, Any] = field(default_factory=dict)
    optimizer_state: dict[str, Any] | None = None
    scheduler_state: dict[str, Any] | None = None
    rng_state: dict[str, Any] | None = None
    model_config: dict[str, Any] = field(default_factory=dict)
    segmentation_config: dict[str, Any] = field(default_factory=dict)
    training_config: dict[str, Any] = field(default_factory=dict)
    validation_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {self.epoch}.")
        if self.global_step < 0:
            raise ValueError(
                f"global_step must be non-negative, got {self.global_step}."
            )
        if not self.segment_states:
            raise ValueError("segment_states must contain at least one segment.")


@dataclass(frozen=True, slots=True)
class LoadedSegmentedCheckpoint:
    """Loaded checkpoint with both payload and manifest."""

    checkpoint: SegmentedCheckpoint
    manifest: CheckpointManifest
    checkpoint_dir: Path


class SegmentedCheckpointStore:
    """Save/load segmented checkpoints under one root directory."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @property
    def index_path(self) -> Path:
        return self.root_dir / INDEX_FILENAME

    def load_index(self) -> CheckpointIndex:
        return CheckpointIndex.load_yaml(self.index_path)

    def save_index(self, index: CheckpointIndex) -> None:
        index.save_yaml(self.index_path)

    def checkpoint_dir(self, checkpoint_name: str) -> Path:
        if not checkpoint_name:
            raise ValueError("checkpoint_name must not be empty.")
        return self.root_dir / checkpoint_name

    def save_checkpoint(
        self,
        *,
        checkpoint_name: str,
        checkpoint: SegmentedCheckpoint,
        checkpoint_type: str = "segmented",
        overwrite: bool = True,
    ) -> CheckpointManifest:
        """Save a segmented checkpoint and return its manifest."""

        ckpt_dir = self.checkpoint_dir(checkpoint_name)
        if ckpt_dir.exists():
            if not overwrite:
                raise FileExistsError(f"Checkpoint already exists: {ckpt_dir}")
            shutil.rmtree(ckpt_dir)

        segments_dir = ckpt_dir / "segments"
        state_dir = ckpt_dir / "state"
        segments_dir.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)

        segment_entries: list[CheckpointSegmentEntry] = []
        for segment_id in sorted(checkpoint.segment_states):
            state = checkpoint.segment_states[segment_id]
            relative = Path("segments") / f"{segment_id.to_path_name()}.pt"
            path = ckpt_dir / relative
            torch.save(state, path)
            sha256 = _sha256_file(path)
            num_tensors, num_parameters = _count_tensors_and_parameters(state)
            segment_entries.append(
                CheckpointSegmentEntry(
                    segment_id=segment_id,
                    relative_path=relative.as_posix(),
                    sha256=sha256,
                    num_tensors=num_tensors,
                    num_parameters=num_parameters,
                )
            )

        non_segment_state_path: str | None = None
        if checkpoint.non_segment_state:
            relative = Path("state") / "non_segment_state.pt"
            torch.save(checkpoint.non_segment_state, ckpt_dir / relative)
            non_segment_state_path = relative.as_posix()

        optimizer_state_path: str | None = None
        if checkpoint.optimizer_state is not None:
            relative = Path("state") / "optimizer_state.pt"
            torch.save(checkpoint.optimizer_state, ckpt_dir / relative)
            optimizer_state_path = relative.as_posix()

        scheduler_state_path: str | None = None
        if checkpoint.scheduler_state is not None:
            relative = Path("state") / "scheduler_state.pt"
            torch.save(checkpoint.scheduler_state, ckpt_dir / relative)
            scheduler_state_path = relative.as_posix()

        rng_state_path: str | None = None
        if checkpoint.rng_state is not None:
            relative = Path("state") / "rng_state.pt"
            torch.save(checkpoint.rng_state, ckpt_dir / relative)
            rng_state_path = relative.as_posix()

        manifest = CheckpointManifest(
            checkpoint_name=checkpoint_name,
            checkpoint_type=checkpoint_type,
            epoch=checkpoint.epoch,
            global_step=checkpoint.global_step,
            model_config=checkpoint.model_config,
            segmentation_config=checkpoint.segmentation_config,
            training_config=checkpoint.training_config,
            validation_metadata=checkpoint.validation_metadata,
            metadata=checkpoint.metadata,
            segment_entries=segment_entries,
            non_segment_state_path=non_segment_state_path,
            optimizer_state_path=optimizer_state_path,
            scheduler_state_path=scheduler_state_path,
            rng_state_path=rng_state_path,
        )
        manifest.save_yaml(ckpt_dir / MANIFEST_FILENAME)
        return manifest

    def load_checkpoint(
        self,
        checkpoint_name: str,
        *,
        map_location: str | torch.device = "cpu",
        verify_hashes: bool = True,
    ) -> LoadedSegmentedCheckpoint:
        """Load a segmented checkpoint by name."""

        ckpt_dir = self.checkpoint_dir(checkpoint_name)
        manifest_path = ckpt_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"Checkpoint manifest not found: {manifest_path}")

        manifest = CheckpointManifest.load_yaml(manifest_path)
        segment_states: dict[SegmentId, dict[str, Any]] = {}
        for entry in manifest.segment_entries:
            path = ckpt_dir / entry.relative_path
            if not path.exists():
                raise FileNotFoundError(f"Segment state file not found: {path}")
            if verify_hashes:
                current_hash = _sha256_file(path)
                if current_hash != entry.sha256:
                    raise ValueError(
                        f"Hash mismatch for {path}: expected {entry.sha256}, "
                        f"got {current_hash}."
                    )
            segment_states[entry.segment_id] = torch.load(
                path,
                map_location=map_location,
                weights_only=False,
            )

        def load_optional(relative_path: str | None) -> dict[str, Any] | None:
            if relative_path is None:
                return None
            path = ckpt_dir / relative_path
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint state file not found: {path}")
            return torch.load(path, map_location=map_location, weights_only=False)

        non_segment_state = load_optional(manifest.non_segment_state_path) or {}
        optimizer_state = load_optional(manifest.optimizer_state_path)
        scheduler_state = load_optional(manifest.scheduler_state_path)
        rng_state = load_optional(manifest.rng_state_path)

        checkpoint = SegmentedCheckpoint(
            epoch=manifest.epoch,
            global_step=manifest.global_step,
            segment_states=segment_states,
            non_segment_state=non_segment_state,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            rng_state=rng_state,
            model_config=manifest.model_config,
            segmentation_config=manifest.segmentation_config,
            training_config=manifest.training_config,
            validation_metadata=manifest.validation_metadata,
            metadata=manifest.metadata,
        )
        return LoadedSegmentedCheckpoint(
            checkpoint=checkpoint,
            manifest=manifest,
            checkpoint_dir=ckpt_dir,
        )

    def save_last_checkpoint(self, checkpoint: SegmentedCheckpoint) -> CheckpointManifest:
        manifest = self.save_checkpoint(
            checkpoint_name="last_segmented_checkpoint",
            checkpoint=checkpoint,
            checkpoint_type="last_segmented",
            overwrite=True,
        )
        index = self.load_index()
        index.last_checkpoint = manifest.checkpoint_name
        self.save_index(index)
        return manifest

    def save_best_checkpoint(
        self,
        checkpoint: SegmentedCheckpoint,
        *,
        metric_name: str,
        metric_value: float,
        lower_is_better: bool = True,
    ) -> CheckpointManifest | None:
        """Save checkpoint as best if it improves the tracked metric.

        Returns the manifest when saved, otherwise ``None``.
        """

        index = self.load_index()
        if not index.should_replace_best(
            metric_value=metric_value,
            lower_is_better=lower_is_better,
        ):
            return None

        validation_metadata = dict(checkpoint.validation_metadata)
        validation_metadata[metric_name] = metric_value
        checkpoint_to_save = SegmentedCheckpoint(
            epoch=checkpoint.epoch,
            global_step=checkpoint.global_step,
            segment_states=checkpoint.segment_states,
            non_segment_state=checkpoint.non_segment_state,
            optimizer_state=checkpoint.optimizer_state,
            scheduler_state=checkpoint.scheduler_state,
            rng_state=checkpoint.rng_state,
            model_config=checkpoint.model_config,
            segmentation_config=checkpoint.segmentation_config,
            training_config=checkpoint.training_config,
            validation_metadata=validation_metadata,
            metadata=checkpoint.metadata,
        )

        manifest = self.save_checkpoint(
            checkpoint_name="best_segmented_checkpoint",
            checkpoint=checkpoint_to_save,
            checkpoint_type="best_segmented",
            overwrite=True,
        )
        index.best_checkpoint = manifest.checkpoint_name
        index.best_metric_name = metric_name
        index.best_metric_value = float(metric_value)
        index.lower_is_better = bool(lower_is_better)
        self.save_index(index)
        return manifest

    def load_last_checkpoint(self, **kwargs: Any) -> LoadedSegmentedCheckpoint:
        index = self.load_index()
        if index.last_checkpoint is None:
            raise FileNotFoundError("No last checkpoint is recorded in the index.")
        return self.load_checkpoint(index.last_checkpoint, **kwargs)

    def load_best_checkpoint(self, **kwargs: Any) -> LoadedSegmentedCheckpoint:
        index = self.load_index()
        if index.best_checkpoint is None:
            raise FileNotFoundError("No best checkpoint is recorded in the index.")
        return self.load_checkpoint(index.best_checkpoint, **kwargs)

    def load_non_segment_state(
        self,
        checkpoint_name: str,
        *,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Load only the non-segment state from a checkpoint (small, no segments loaded)."""
        ckpt_dir = self.checkpoint_dir(checkpoint_name)
        manifest = CheckpointManifest.load_yaml(ckpt_dir / MANIFEST_FILENAME)
        if manifest.non_segment_state_path is None:
            return {}
        path = ckpt_dir / manifest.non_segment_state_path
        if not path.exists():
            return {}
        return torch.load(path, map_location=map_location, weights_only=False)

    def iter_segment_states(
        self,
        checkpoint_name: str,
        *,
        map_location: str | torch.device = "cpu",
    ):
        """Yield (SegmentId, state_dict) one at a time without holding all in RAM.

        Loads each segment's .pt file individually so only one segment's weights
        are in memory between yields, preserving the single-segment invariant.
        """
        ckpt_dir = self.checkpoint_dir(checkpoint_name)
        manifest = CheckpointManifest.load_yaml(ckpt_dir / MANIFEST_FILENAME)
        for entry in manifest.segment_entries:
            path = ckpt_dir / entry.relative_path
            if not path.exists():
                raise FileNotFoundError(f"Segment state file not found: {path}")
            state = torch.load(path, map_location=map_location, weights_only=False)
            yield entry.segment_id, state
            del state
