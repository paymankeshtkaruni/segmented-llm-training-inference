"""Tests for Phase 20 YAML protocol export/import."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sequential_segmented_llm_training_inference.config import (
    ExecutionConfig,
    ExportConfig,
    ModelConfig,
    OptimizerConfig,
    ProtocolConfig,
    RuntimeConfig,
    SegmentationConfig,
    StorageConfig,
    TrainingConfig,
)
from sequential_segmented_llm_training_inference.export.protocol_exporter import (
    ProtocolExporter,
    build_protocol_dict,
    export_protocol_yaml,
    validate_protocol_dict,
    write_protocol_yaml,
)
from sequential_segmented_llm_training_inference.export.protocol_importer import (
    import_protocol_config,
    import_protocol_yaml,
    load_protocol_yaml,
    protocol_dict_to_config,
)


def make_protocol_config(*, device: str = "cpu", backend: str = "disk_streaming") -> ProtocolConfig:
    model = ModelConfig(
        vocab_size=128,
        max_seq_len=16,
        n_layers=2,
        d_model=32,
        n_heads=8,
        d_ff=64,
        dropout=0.1,
    )
    segmentation = SegmentationConfig(attention_segments=4, mlp_chunks=4)
    return ProtocolConfig(
        model=model,
        segmentation=segmentation,
        storage=StorageConfig(device=device, backend=backend, segment_dir="segments"),
        execution=ExecutionConfig(mode="strict_single_segment", max_active_segments=1),
        training=TrainingConfig(epochs=2, batch_size=3, gradient_accumulation_steps=1),
        optimizer=OptimizerConfig(type="segmentwise_adamw", learning_rate=1e-4),
        runtime_records=RuntimeConfig(policy="full_segment_records"),
        export=ExportConfig(
            export_full_model=True,
            full_model_export_path="exports/full_model",
            export_yaml_protocol=True,
            yaml_protocol_path="exports/protocol.yaml",
        ),
    )


def test_build_protocol_dict_contains_required_sections() -> None:
    protocol = build_protocol_dict(
        protocol_config=make_protocol_config(),
        experiment={"name": "unit_test", "run_id": "run_x"},
        data={"train_split": "train.csv", "validation_split": "val.csv", "test_split": "test.csv"},
        results={"best_validation_loss": 1.2, "final_test_loss": 1.3},
    )

    for key in (
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
    ):
        assert key in protocol

    assert protocol["experiment"]["name"] == "unit_test"
    assert protocol["data"]["train_split"] == "train.csv"
    assert protocol["validation"]["mode"] == "segmented_forward"
    assert protocol["test"]["mode"] == "segmented_forward"
    assert protocol["execution"]["max_active_segments"] == 1


def test_write_and_load_protocol_yaml_roundtrip(tmp_path: Path) -> None:
    config = make_protocol_config(device="cuda", backend="cpu_ram_offload")
    path = tmp_path / "nested" / "protocol.yaml"

    result = export_protocol_yaml(
        config,
        path,
        experiment={"name": "roundtrip", "run_id": "run_001"},
        data={"dataset_name": "dummy"},
    )

    assert result.path == path
    assert path.exists()

    loaded = load_protocol_yaml(path)
    assert loaded["experiment"]["name"] == "roundtrip"
    assert loaded["segment_storage"]["backend"] == "cpu_ram_offload"
    assert loaded["data"]["dataset_name"] == "dummy"


def test_import_protocol_yaml_returns_config(tmp_path: Path) -> None:
    config = make_protocol_config()
    path = tmp_path / "protocol.yaml"
    export_protocol_yaml(config, path)

    result = import_protocol_yaml(path)

    assert result.path == path
    assert result.protocol_version == "1.0"
    assert result.config.model.d_model == 32
    assert result.config.segmentation.attention_segments == 4
    assert result.config.execution.max_active_segments == 1


def test_protocol_dict_to_config_reconstructs_core_config() -> None:
    original = make_protocol_config(device="cuda", backend="disk_streaming")
    protocol = build_protocol_dict(protocol_config=original)

    reconstructed = protocol_dict_to_config(protocol)

    assert reconstructed.storage.device == "cuda"
    assert reconstructed.storage.backend == "disk_streaming"
    assert reconstructed.model.n_layers == original.model.n_layers
    assert reconstructed.segmentation.mlp_chunks == original.segmentation.mlp_chunks


def test_protocol_exporter_class_exports(tmp_path: Path) -> None:
    exporter = ProtocolExporter()
    path = tmp_path / "protocol.yaml"

    result = exporter.export(
        make_protocol_config(),
        path,
        checkpointing={"best_checkpoint_path": "ckpt/best"},
    )

    assert result.path.exists()
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded["checkpointing"]["best_checkpoint_path"] == "ckpt/best"


def test_validate_protocol_dict_rejects_missing_required_section() -> None:
    protocol = build_protocol_dict(protocol_config=make_protocol_config())
    protocol.pop("model")

    with pytest.raises(ValueError, match="missing required"):
        validate_protocol_dict(protocol)


def test_validate_protocol_dict_rejects_non_strict_execution() -> None:
    protocol = build_protocol_dict(protocol_config=make_protocol_config())
    protocol["execution"]["mode"] = "windowed"

    with pytest.raises(ValueError, match="strict_single_segment"):
        validate_protocol_dict(protocol)


def test_validate_protocol_dict_rejects_bad_validation_mode() -> None:
    protocol = build_protocol_dict(protocol_config=make_protocol_config())
    protocol["validation"]["mode"] = "full_model"

    with pytest.raises(ValueError, match="validation.mode"):
        validate_protocol_dict(protocol)


def test_validate_protocol_dict_rejects_bad_test_mode() -> None:
    protocol = build_protocol_dict(protocol_config=make_protocol_config())
    protocol["test"]["mode"] = "full_model"

    with pytest.raises(ValueError, match="test.mode"):
        validate_protocol_dict(protocol)


def test_load_protocol_yaml_rejects_non_mapping_root(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_protocol_yaml(path)


def test_load_protocol_yaml_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_protocol_yaml(tmp_path / "missing.yaml")


def test_write_protocol_yaml_creates_parent_directory(tmp_path: Path) -> None:
    protocol = build_protocol_dict(protocol_config=make_protocol_config())
    path = tmp_path / "a" / "b" / "protocol.yaml"

    result = write_protocol_yaml(protocol, path)

    assert result.path == path
    assert path.exists()


def test_protocol_preserves_results_and_metadata(tmp_path: Path) -> None:
    config = make_protocol_config()
    path = tmp_path / "protocol.yaml"
    export_protocol_yaml(
        config,
        path,
        results={"best_validation_loss": 0.5, "final_test_metrics": {"accuracy": 0.9}},
        software={"package": "test"},
        hardware_runtime={"device_name": "cpu"},
    )

    loaded = load_protocol_yaml(path)

    assert loaded["results"]["best_validation_loss"] == 0.5
    assert loaded["results"]["final_test_metrics"]["accuracy"] == 0.9
    assert loaded["software"]["package"] == "test"
    assert loaded["hardware_runtime"]["device_name"] == "cpu"


def test_import_protocol_config_convenience_function(tmp_path: Path) -> None:
    path = tmp_path / "protocol.yaml"
    export_protocol_yaml(make_protocol_config(), path)

    config = import_protocol_config(path)

    assert isinstance(config, ProtocolConfig)
    assert config.model.vocab_size == 128
