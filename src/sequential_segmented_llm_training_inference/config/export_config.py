"""Full-model and YAML-protocol export configuration."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class ExportConfig:
    """Artifacts exported after segmented learning."""

    export_full_model: bool = True
    full_model_export_path: str = "exports/full_model"
    export_yaml_protocol: bool = True
    yaml_protocol_path: str = "exports/segmented_training_protocol.yaml"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.export_full_model and not self.full_model_export_path:
            raise ValueError("full_model_export_path is required when export_full_model=true.")
        if self.export_yaml_protocol and not self.yaml_protocol_path:
            raise ValueError("yaml_protocol_path is required when export_yaml_protocol=true.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExportConfig":
        return cls(**data)
