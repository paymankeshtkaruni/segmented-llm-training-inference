"""Checkpoint manifest structures for segmented LLM checkpoints.

Phase 15 scope:
- Represent segmented checkpoint metadata in a deterministic, YAML-safe form.
- Track segment state files, non-segment state, optimizer/scheduler/RNG state,
  epoch/global step, validation metadata, and arbitrary run metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId


MANIFEST_FILENAME = "checkpoint_manifest.yaml"
INDEX_FILENAME = "checkpoint_index.yaml"


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class CheckpointSegmentEntry:
    """Manifest entry for one stored segment state file."""

    segment_id: SegmentId
    relative_path: str
    sha256: str
    num_tensors: int
    num_parameters: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id.to_dict(),
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "num_tensors": self.num_tensors,
            "num_parameters": self.num_parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointSegmentEntry":
        return cls(
            segment_id=SegmentId.from_dict(data["segment_id"]),
            relative_path=str(data["relative_path"]),
            sha256=str(data["sha256"]),
            num_tensors=int(data["num_tensors"]),
            num_parameters=int(data["num_parameters"]),
        )


@dataclass(slots=True)
class CheckpointManifest:
    """YAML-safe manifest for one segmented checkpoint."""

    checkpoint_name: str
    checkpoint_type: str
    epoch: int
    global_step: int
    created_at: str = field(default_factory=utc_now_iso)
    protocol_version: str = "1.0"
    model_config: dict[str, Any] = field(default_factory=dict)
    segmentation_config: dict[str, Any] = field(default_factory=dict)
    training_config: dict[str, Any] = field(default_factory=dict)
    validation_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    segment_entries: list[CheckpointSegmentEntry] = field(default_factory=list)
    non_segment_state_path: str | None = None
    optimizer_state_path: str | None = None
    scheduler_state_path: str | None = None
    rng_state_path: str | None = None

    def __post_init__(self) -> None:
        if self.epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {self.epoch}.")
        if self.global_step < 0:
            raise ValueError(
                f"global_step must be non-negative, got {self.global_step}."
            )
        if not self.checkpoint_name:
            raise ValueError("checkpoint_name must not be empty.")
        if not self.checkpoint_type:
            raise ValueError("checkpoint_type must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "checkpoint_name": self.checkpoint_name,
            "checkpoint_type": self.checkpoint_type,
            "created_at": self.created_at,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_config": self.model_config,
            "segmentation_config": self.segmentation_config,
            "training_config": self.training_config,
            "validation_metadata": self.validation_metadata,
            "metadata": self.metadata,
            "segment_entries": [entry.to_dict() for entry in self.segment_entries],
            "non_segment_state_path": self.non_segment_state_path,
            "optimizer_state_path": self.optimizer_state_path,
            "scheduler_state_path": self.scheduler_state_path,
            "rng_state_path": self.rng_state_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointManifest":
        entries = [
            CheckpointSegmentEntry.from_dict(item)
            for item in data.get("segment_entries", [])
        ]
        return cls(
            protocol_version=str(data.get("protocol_version", "1.0")),
            checkpoint_name=str(data["checkpoint_name"]),
            checkpoint_type=str(data["checkpoint_type"]),
            created_at=str(data["created_at"]),
            epoch=int(data["epoch"]),
            global_step=int(data["global_step"]),
            model_config=dict(data.get("model_config", {})),
            segmentation_config=dict(data.get("segmentation_config", {})),
            training_config=dict(data.get("training_config", {})),
            validation_metadata=dict(data.get("validation_metadata", {})),
            metadata=dict(data.get("metadata", {})),
            segment_entries=entries,
            non_segment_state_path=data.get("non_segment_state_path"),
            optimizer_state_path=data.get("optimizer_state_path"),
            scheduler_state_path=data.get("scheduler_state_path"),
            rng_state_path=data.get("rng_state_path"),
        )

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load_yaml(cls, path: str | Path) -> "CheckpointManifest":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid manifest YAML at {path}.")
        return cls.from_dict(data)


@dataclass(slots=True)
class CheckpointIndex:
    """Index tracking last and best segmented checkpoints."""

    last_checkpoint: str | None = None
    best_checkpoint: str | None = None
    best_metric_name: str | None = None
    best_metric_value: float | None = None
    lower_is_better: bool = True

    def should_replace_best(
        self,
        *,
        metric_value: float,
        lower_is_better: bool,
    ) -> bool:
        if self.best_metric_value is None:
            return True
        if lower_is_better:
            return metric_value < self.best_metric_value
        return metric_value > self.best_metric_value

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_checkpoint": self.last_checkpoint,
            "best_checkpoint": self.best_checkpoint,
            "best_metric_name": self.best_metric_name,
            "best_metric_value": self.best_metric_value,
            "lower_is_better": self.lower_is_better,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointIndex":
        return cls(
            last_checkpoint=data.get("last_checkpoint"),
            best_checkpoint=data.get("best_checkpoint"),
            best_metric_name=data.get("best_metric_name"),
            best_metric_value=(
                None if data.get("best_metric_value") is None else float(data["best_metric_value"])
            ),
            lower_is_better=bool(data.get("lower_is_better", True)),
        )

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load_yaml(cls, path: str | Path) -> "CheckpointIndex":
        path = Path(path)
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid checkpoint index YAML at {path}.")
        return cls.from_dict(data)
