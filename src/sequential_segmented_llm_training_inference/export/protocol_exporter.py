"""YAML protocol export for sequential segmented LLM training.

Phase 20 scope:
- Export the full sequential segmented training protocol as YAML.
- Include enough information to reproduce or audit training/validation/test/export.
- Accept either a ProtocolConfig object or raw section dictionaries.
- Preserve explicit user/run metadata such as data splits, checkpoints, metrics, seeds,
  software versions, and hardware/runtime metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import platform
import time

import yaml

from sequential_segmented_llm_training_inference.config import ProtocolConfig


REQUIRED_PROTOCOL_SECTIONS: tuple[str, ...] = (
    "protocol_version",
    "experiment",
    "model",
    "segmentation",
    "execution",
    "segment_storage",
    "training",
    "optimizer",
    "runtime_records",
    "loss",
    "data",
    "validation",
    "test",
    "checkpointing",
    "exports",
    "reproducibility",
    "results",
)


@dataclass(frozen=True, slots=True)
class ProtocolExportResult:
    """Result returned after exporting a YAML protocol file."""

    path: Path
    protocol_version: str
    num_top_level_sections: int
    created_at_unix: float


@dataclass(slots=True)
class ProtocolExportOptions:
    """Options for protocol YAML emission."""

    protocol_version: str = "1.0"
    sort_keys: bool = False
    include_runtime_defaults: bool = True

    def __post_init__(self) -> None:
        if not self.protocol_version:
            raise ValueError("protocol_version must not be empty.")


def _to_plain_dict(value: Any) -> dict[str, Any]:
    """Convert supported config/dataclass/mapping values into a plain dict."""

    if value is None:
        return {}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        result = value.to_dict()
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        raise TypeError(
            "Expected a mapping or object with to_dict(), "
            f"got {type(value).__name__}."
        )
    if not isinstance(result, dict):
        raise TypeError("to_dict() must return a dictionary.")
    return result


def _default_experiment() -> dict[str, Any]:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "name": "sequential_segmented_llm_training_run",
        "run_id": f"run_{int(time.time())}",
        "created_at": now,
        "description": "Sequential training and inference of a segmented LLM.",
    }


def _default_loss() -> dict[str, Any]:
    return {
        "type": "causal_cross_entropy",
        "label_shift": "internal",
        "ignore_index": -100,
    }


def _default_validation() -> dict[str, Any]:
    return {
        "mode": "segmented_forward",
        "frequency": "each_epoch",
        "metric_for_best_checkpoint": "validation_loss",
        "lower_is_better": True,
    }


def _default_test() -> dict[str, Any]:
    return {
        "mode": "segmented_forward",
        "checkpoint": "best_segmented_checkpoint",
        "metrics": ["test_loss"],
    }


def _default_checkpointing() -> dict[str, Any]:
    return {
        "save_last": True,
        "save_best": True,
        "checkpoint_dir": "checkpoints",
        "best_checkpoint_path": "checkpoints/best_segmented_checkpoint",
        "last_checkpoint_path": "checkpoints/last_segmented_checkpoint",
    }


def _default_reproducibility() -> dict[str, Any]:
    return {
        "seed": 42,
        "deterministic_mode": False,
        "save_python_rng_state": True,
        "save_torch_rng_state": True,
        "save_cuda_rng_state": True,
    }


def _default_results() -> dict[str, Any]:
    return {
        "best_validation_loss": None,
        "final_test_loss": None,
        "final_test_metrics": {},
    }


def _default_runtime_metadata() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def build_protocol_dict(
    *,
    protocol_config: ProtocolConfig,
    options: ProtocolExportOptions | None = None,
    experiment: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | None = None,
    loss: Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
    test: Mapping[str, Any] | None = None,
    checkpointing: Mapping[str, Any] | None = None,
    reproducibility: Mapping[str, Any] | None = None,
    results: Mapping[str, Any] | None = None,
    hardware_runtime: Mapping[str, Any] | None = None,
    software: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete YAML-safe protocol dictionary."""

    options = options or ProtocolExportOptions()
    base = protocol_config.to_dict()

    protocol: dict[str, Any] = {
        "protocol_version": options.protocol_version,
        "experiment": {**_default_experiment(), **dict(experiment or {})},
        "model": dict(base["model"]),
        "segmentation": dict(base["segmentation"]),
        "execution": dict(base["execution"]),
        "segment_storage": {**dict(base["segment_storage"]), "device": base.get("device", dict(base["segment_storage"]).get("device", "cpu"))},
        "training": dict(base["training"]),
        "optimizer": dict(base["optimizer"]),
        "runtime_records": dict(base["runtime_records"]),
        "loss": {**_default_loss(), **dict(loss or {})},
        "data": dict(data or {}),
        "validation": {**_default_validation(), **dict(validation or {})},
        "test": {**_default_test(), **dict(test or {})},
        "checkpointing": {**_default_checkpointing(), **dict(checkpointing or {})},
        "exports": dict(base["exports"]),
        "reproducibility": {**_default_reproducibility(), **dict(reproducibility or {})},
        "results": {**_default_results(), **dict(results or {})},
    }

    if hardware_runtime is not None or options.include_runtime_defaults:
        protocol["hardware_runtime"] = {
            **_default_runtime_metadata(),
            **dict(hardware_runtime or {}),
        }
    if software is not None:
        protocol["software"] = dict(software)
    if extra is not None:
        protocol["extra"] = dict(extra)

    validate_protocol_dict(protocol)
    return protocol


def validate_protocol_dict(protocol: Mapping[str, Any]) -> None:
    """Validate top-level protocol structure and segmented-mode invariants."""

    if not isinstance(protocol, Mapping):
        raise TypeError("protocol must be a mapping.")

    missing = [key for key in REQUIRED_PROTOCOL_SECTIONS if key not in protocol]
    if missing:
        raise ValueError(f"Protocol missing required section(s): {', '.join(missing)}")

    if protocol.get("protocol_version") in {None, ""}:
        raise ValueError("protocol_version must not be empty.")

    execution = protocol.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("execution section must be a mapping.")
    if execution.get("mode") != "strict_single_segment":
        raise ValueError("protocol execution.mode must be 'strict_single_segment'.")
    if int(execution.get("max_active_segments", -1)) != 1:
        raise ValueError("protocol execution.max_active_segments must be 1.")

    segmentation = protocol.get("segmentation")
    if not isinstance(segmentation, Mapping):
        raise ValueError("segmentation section must be a mapping.")
    if int(segmentation.get("attention_segments", 0)) <= 1:
        raise ValueError("protocol attention_segments must be > 1.")
    if int(segmentation.get("mlp_chunks", 0)) <= 1:
        raise ValueError("protocol mlp_chunks must be > 1.")

    training = protocol.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("training section must be a mapping.")
    if training.get("backward_mode", "recomputation") != "recomputation":
        raise ValueError("training.backward_mode must be 'recomputation'.")
    if (
        training.get("gradient_clip_norm") is not None
        and training.get("update_style") == "immediate_segment_update"
    ):
        raise ValueError(
            "true-global gradient clipping is incompatible with immediate_segment_update."
        )

    runtime_records = protocol.get("runtime_records")
    if not isinstance(runtime_records, Mapping):
        raise ValueError("runtime_records section must be a mapping.")
    if runtime_records.get("detach_record_tensors", True) is not True:
        raise ValueError("runtime_records.detach_record_tensors must be true.")
    if (
        runtime_records.get("restore_rng_state_during_backward", True)
        and runtime_records.get("store_rng_state", True) is not True
    ):
        raise ValueError(
            "runtime_records.restore_rng_state_during_backward requires store_rng_state=true."
        )

    validation = protocol.get("validation")
    test = protocol.get("test")
    if not isinstance(validation, Mapping) or validation.get("mode") != "segmented_forward":
        raise ValueError("validation.mode must be 'segmented_forward'.")
    if not isinstance(test, Mapping) or test.get("mode") != "segmented_forward":
        raise ValueError("test.mode must be 'segmented_forward'.")


def write_protocol_yaml(
    protocol: Mapping[str, Any],
    path: str | Path,
    *,
    sort_keys: bool = False,
) -> ProtocolExportResult:
    """Write a validated protocol mapping to YAML."""

    validate_protocol_dict(protocol)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(dict(protocol), stream, sort_keys=sort_keys)

    return ProtocolExportResult(
        path=output_path,
        protocol_version=str(protocol["protocol_version"]),
        num_top_level_sections=len(protocol),
        created_at_unix=time.time(),
    )


class ProtocolExporter:
    """Exporter for full sequential segmented training protocol YAML files."""

    def __init__(self, options: ProtocolExportOptions | None = None) -> None:
        self.options = options or ProtocolExportOptions()

    def build(self, protocol_config: ProtocolConfig, **sections: Any) -> dict[str, Any]:
        return build_protocol_dict(
            protocol_config=protocol_config,
            options=self.options,
            **sections,
        )

    def export(
        self,
        protocol_config: ProtocolConfig,
        path: str | Path,
        **sections: Any,
    ) -> ProtocolExportResult:
        protocol = self.build(protocol_config, **sections)
        return write_protocol_yaml(protocol, path, sort_keys=self.options.sort_keys)


def export_protocol_yaml(
    protocol_config: ProtocolConfig,
    path: str | Path,
    **sections: Any,
) -> ProtocolExportResult:
    """Convenience function for exporting a protocol YAML file."""

    return ProtocolExporter().export(protocol_config, path, **sections)
