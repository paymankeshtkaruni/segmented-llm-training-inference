"""Complete protocol configuration for YAML export/import."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .export_config import ExportConfig
from .model_config import ModelConfig
from .optimizer_config import OptimizerConfig
from .runtime_config import ExecutionConfig, RuntimeConfig
from .segmentation_config import SegmentationConfig
from .storage_config import StorageConfig
from .training_config import TrainingConfig


@dataclass(frozen=True)
class ProtocolConfig:
    """Complete config needed to establish segmented training."""

    model: ModelConfig
    segmentation: SegmentationConfig
    storage: StorageConfig
    execution: ExecutionConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    runtime_records: RuntimeConfig
    export: ExportConfig

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.model.validate()
        self.segmentation.validate_against_model(self.model)
        self.storage.validate()
        self.execution.validate()
        self.training.validate()
        self.optimizer.validate()
        self.runtime_records.validate()
        self.export.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.storage.device,
            "execution": self.execution.to_dict(),
            "segment_storage": {
                "backend": self.storage.backend,
                "segment_dir": self.storage.segment_dir,
            },
            "model": self.model.to_dict(),
            "segmentation": self.segmentation.to_dict(),
            "training": self.training.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "runtime_records": self.runtime_records.to_dict(),
            "exports": self.export.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProtocolConfig":
        device = data.get("device", "cpu")
        return cls(
            model=ModelConfig.from_dict(data.get("model", {})),
            segmentation=SegmentationConfig.from_dict(data.get("segmentation", {})),
            storage=StorageConfig.from_dict(data.get("segment_storage", {}), device=device),
            execution=ExecutionConfig.from_dict(data.get("execution", {})),
            training=TrainingConfig.from_dict(data.get("training", {})),
            optimizer=OptimizerConfig.from_dict(data.get("optimizer", {})),
            runtime_records=RuntimeConfig.from_dict(data.get("runtime_records", {})),
            export=ExportConfig.from_dict(data.get("exports", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProtocolConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        if not isinstance(data, dict):
            raise ValueError("Protocol YAML must contain a mapping at the root.")
        return cls.from_dict(data)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.to_dict(), stream, sort_keys=False)
