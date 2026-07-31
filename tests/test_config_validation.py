"""Tests for Phase 1 configuration validation."""

from __future__ import annotations

import pytest

from sequential_segmented_llm_training_inference.config import (
    ExecutionConfig,
    ModelConfig,
    ProtocolConfig,
    RuntimeConfig,
    SegmentationConfig,
    StorageConfig,
    TrainingConfig,
)


def valid_protocol_dict() -> dict:
    return {
        "device": "cuda",
        "execution": {"mode": "strict_single_segment", "max_active_segments": 1},
        "segment_storage": {"backend": "cpu_ram_offload"},
        "model": {
            "architecture": "gpt_decoder",
            "vocab_size": 50257,
            "max_seq_len": 512,
            "n_layers": 4,
            "d_model": 256,
            "n_heads": 8,
            "d_ff": 1024,
            "dropout": 0.1,
        },
        "segmentation": {"attention_segments": 4, "mlp_chunks": 4},
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "gradient_accumulation_steps": 1,
            "update_style": "after_full_backward",
            "precision": "fp32",
        },
        "optimizer": {"type": "segmentwise_adamw", "learning_rate": 1.0e-4},
        "runtime_records": {
            "policy": "full_segment_records",
            "store_rng_state": True,
            "restore_rng_state_during_backward": True,
            "recomputation_check": True,
        },
        "exports": {
            "export_full_model": True,
            "full_model_export_path": "exports/full_model",
            "export_yaml_protocol": True,
            "yaml_protocol_path": "exports/segmented_training_protocol.yaml",
        },
    }


def test_valid_protocol_config_passes() -> None:
    config = ProtocolConfig.from_dict(valid_protocol_dict())
    assert config.model.head_dim == 32
    assert config.segmentation.heads_per_segment(config.model) == 2
    assert config.segmentation.mlp_chunk_size(config.model) == 256


@pytest.mark.parametrize("attention_segments", [0, 1])
def test_attention_segments_must_be_greater_than_one(attention_segments: int) -> None:
    with pytest.raises(ValueError, match="attention_segments"):
        SegmentationConfig(attention_segments=attention_segments, mlp_chunks=4)


@pytest.mark.parametrize("mlp_chunks", [0, 1])
def test_mlp_chunks_must_be_greater_than_one(mlp_chunks: int) -> None:
    with pytest.raises(ValueError, match="mlp_chunks"):
        SegmentationConfig(attention_segments=4, mlp_chunks=mlp_chunks)


def test_n_heads_must_be_divisible_by_attention_segments() -> None:
    model = ModelConfig(d_model=240, n_heads=10, d_ff=1024)
    segmentation = SegmentationConfig(attention_segments=4, mlp_chunks=4)
    with pytest.raises(ValueError, match="n_heads"):
        segmentation.validate_against_model(model)


def test_d_ff_must_be_divisible_by_mlp_chunks() -> None:
    model = ModelConfig(d_model=256, n_heads=8, d_ff=1000)
    segmentation = SegmentationConfig(attention_segments=4, mlp_chunks=6)
    with pytest.raises(ValueError, match="d_ff"):
        segmentation.validate_against_model(model)


def test_d_model_must_be_divisible_by_n_heads() -> None:
    with pytest.raises(ValueError, match="d_model"):
        ModelConfig(d_model=250, n_heads=8)


def test_dropout_must_be_valid() -> None:
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig(dropout=1.0)
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig(dropout=-0.1)


def test_execution_must_be_strict_single_segment() -> None:
    with pytest.raises(ValueError, match="strict_single_segment"):
        ExecutionConfig(mode="windowed")  # type: ignore[arg-type]


def test_max_active_segments_must_be_one() -> None:
    with pytest.raises(ValueError, match="max_active_segments"):
        ExecutionConfig(max_active_segments=2)


def test_cpu_supports_only_disk_streaming() -> None:
    StorageConfig(device="cpu", backend="disk_streaming")
    with pytest.raises(ValueError, match="CPU"):
        StorageConfig(device="cpu", backend="cpu_ram_offload")


def test_gpu_supports_disk_and_cpu_ram_offload() -> None:
    StorageConfig(device="cuda", backend="disk_streaming")
    StorageConfig(device="cuda", backend="cpu_ram_offload")


def test_gradient_accumulation_rejects_immediate_update() -> None:
    with pytest.raises(ValueError, match="immediate_segment_update"):
        TrainingConfig(gradient_accumulation_steps=2, update_style="immediate_segment_update")


def test_runtime_rng_restore_requires_rng_store() -> None:
    with pytest.raises(ValueError, match="store_rng_state"):
        RuntimeConfig(store_rng_state=False, restore_rng_state_during_backward=True)


def test_protocol_yaml_roundtrip(tmp_path) -> None:
    config = ProtocolConfig.from_dict(valid_protocol_dict())
    path = tmp_path / "protocol.yaml"
    config.to_yaml(path)
    loaded = ProtocolConfig.from_yaml(path)
    assert loaded.to_dict() == config.to_dict()
