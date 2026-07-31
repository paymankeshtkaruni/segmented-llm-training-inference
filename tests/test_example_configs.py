"""Tests for Phase 22 example YAML configs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sequential_segmented_llm_training_inference.config import ProtocolConfig


CONFIG_NAMES = (
    "example_cpu_disk.yaml",
    "example_gpu_disk.yaml",
    "example_gpu_cpu_offload.yaml",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_config(name: str) -> dict:
    path = _repo_root() / "configs" / name
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    assert isinstance(data, dict)
    return data


def test_all_example_configs_exist() -> None:
    for name in CONFIG_NAMES:
        assert (_repo_root() / "configs" / name).exists()


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_example_config_is_valid_protocol_config(name: str) -> None:
    config = ProtocolConfig.from_dict(_load_config(name))

    assert config.execution.mode == "strict_single_segment"
    assert config.execution.max_active_segments == 1
    assert config.segmentation.attention_segments > 1
    assert config.segmentation.mlp_chunks > 1
    assert config.export.export_full_model is True
    assert config.export.export_yaml_protocol is True


def test_cpu_example_uses_disk_streaming() -> None:
    config = ProtocolConfig.from_dict(_load_config("example_cpu_disk.yaml"))

    assert config.storage.device == "cpu"
    assert config.storage.backend == "disk_streaming"


def test_gpu_disk_example_uses_cuda_disk_streaming() -> None:
    config = ProtocolConfig.from_dict(_load_config("example_gpu_disk.yaml"))

    assert config.storage.device == "cuda"
    assert config.storage.backend == "disk_streaming"


def test_gpu_cpu_offload_example_uses_cuda_cpu_ram_offload() -> None:
    config = ProtocolConfig.from_dict(_load_config("example_gpu_cpu_offload.yaml"))

    assert config.storage.device == "cuda"
    assert config.storage.backend == "cpu_ram_offload"


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_examples_use_segmented_validation_and_test(name: str) -> None:
    data = _load_config(name)

    assert data["validation"]["mode"] == "segmented_forward"
    assert data["test"]["mode"] == "segmented_forward"


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_examples_include_yaml_protocol_export(name: str) -> None:
    data = _load_config(name)

    assert data["exports"]["export_yaml_protocol"] is True
    assert data["exports"]["yaml_protocol_path"].endswith("segmented_training_protocol.yaml")


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_examples_include_inference_section(name: str) -> None:
    data = _load_config(name)

    assert data["inference"]["mode"] == "segmented_forward_generation"
    assert data["inference"]["checkpoint"] == "best_segmented_checkpoint"
