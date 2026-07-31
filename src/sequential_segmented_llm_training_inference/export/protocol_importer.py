"""YAML protocol import for sequential segmented LLM training.

The importer loads a previously exported protocol YAML file and reconstructs the
core ProtocolConfig. The same YAML can also be kept as a full experiment record
including data paths, checkpoints, metrics, and runtime metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from sequential_segmented_llm_training_inference.config import ProtocolConfig
from sequential_segmented_llm_training_inference.export.protocol_exporter import validate_protocol_dict


@dataclass(frozen=True, slots=True)
class ProtocolImportResult:
    """Result returned by protocol import."""

    path: Path
    protocol: dict[str, Any]
    config: ProtocolConfig

    @property
    def protocol_version(self) -> str:
        return str(self.protocol.get("protocol_version", ""))


def load_protocol_yaml(path: str | Path) -> dict[str, Any]:
    """Load and validate a protocol YAML file as a plain dictionary."""

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Protocol YAML not found: {input_path}")
    with input_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError("Protocol YAML must contain a mapping at the root.")
    validate_protocol_dict(data)
    return data


def protocol_dict_to_config(protocol: Mapping[str, Any]) -> ProtocolConfig:
    """Convert a full protocol dictionary into the core ProtocolConfig."""

    validate_protocol_dict(protocol)
    config_data = {
        "device": protocol.get("segment_storage", {}).get("device", protocol.get("device", "cpu")),
        "execution": dict(protocol["execution"]),
        "segment_storage": dict(protocol["segment_storage"]),
        "model": dict(protocol["model"]),
        "segmentation": dict(protocol["segmentation"]),
        "training": dict(protocol["training"]),
        "optimizer": dict(protocol["optimizer"]),
        "runtime_records": dict(protocol["runtime_records"]),
        "exports": dict(protocol["exports"]),
    }

    storage = config_data["segment_storage"]
    if "device" in storage and "device" not in config_data:
        config_data["device"] = storage["device"]

    return ProtocolConfig.from_dict(config_data)


class ProtocolImporter:
    """Importer for full sequential segmented training protocol YAML files."""

    def import_protocol(self, path: str | Path) -> ProtocolImportResult:
        input_path = Path(path)
        protocol = load_protocol_yaml(input_path)
        config = protocol_dict_to_config(protocol)
        return ProtocolImportResult(path=input_path, protocol=protocol, config=config)

    def import_config(self, path: str | Path) -> ProtocolConfig:
        return self.import_protocol(path).config


def import_protocol_yaml(path: str | Path) -> ProtocolImportResult:
    """Convenience function for loading protocol YAML and core config."""

    return ProtocolImporter().import_protocol(path)


def import_protocol_config(path: str | Path) -> ProtocolConfig:
    """Convenience function for loading only the core ProtocolConfig."""

    return ProtocolImporter().import_config(path)
