"""Tests for Phase 19 full model export."""

from __future__ import annotations

from pathlib import Path

import torch

from sequential_segmented_llm_training_inference.export.full_model_exporter import (
    FullModelExportConfig,
    FullModelExporter,
    export_full_model_from_checkpoint,
)


def _fake_checkpoint() -> dict:
    d_model = 4
    n_layers = 2
    attention_segments = 2
    mlp_chunks = 2
    head_segment_dim = 2
    mlp_chunk = 3

    segments = {}
    for layer in range(n_layers):
        for seg in range(attention_segments):
            base = 100 * layer + 10 * seg
            segments[f"layer_{layer}.attention.{seg}"] = {
                "q_proj.weight": torch.full((head_segment_dim, d_model), base + 1.0),
                "k_proj.weight": torch.full((head_segment_dim, d_model), base + 2.0),
                "v_proj.weight": torch.full((head_segment_dim, d_model), base + 3.0),
                "q_proj.bias": torch.full((head_segment_dim,), base + 4.0),
                "k_proj.bias": torch.full((head_segment_dim,), base + 5.0),
                "v_proj.bias": torch.full((head_segment_dim,), base + 6.0),
            }
        for seg in range(mlp_chunks):
            base = 1000 * layer + 100 * seg
            segments[f"layer_{layer}.mlp.{seg}"] = {
                "fc1.weight": torch.full((mlp_chunk, d_model), base + 1.0),
                "fc1.bias": torch.full((mlp_chunk,), base + 2.0),
                "fc2.weight": torch.full((d_model, mlp_chunk), base + 3.0),
            }

    return {
        "model_config": {
            "architecture": "gpt_decoder",
            "n_layers": n_layers,
            "d_model": d_model,
            "n_heads": 4,
            "d_ff": mlp_chunk * mlp_chunks,
            "vocab_size": 99,
            "max_seq_len": 16,
        },
        "segmentation_config": {
            "attention_segments": attention_segments,
            "mlp_chunks": mlp_chunks,
        },
        "non_segment_state": {
            "token_embedding.weight": torch.ones(99, d_model),
            "layers.0.attention_output_projection.weight": torch.eye(d_model),
        },
        "segments": segments,
    }


def test_export_creates_files(tmp_path: Path) -> None:
    checkpoint = _fake_checkpoint()
    result = export_full_model_from_checkpoint(checkpoint, tmp_path)

    assert result.full_model_path.exists()
    assert result.metadata_path.exists()
    assert result.num_attention_segments == 4
    assert result.num_mlp_segments == 4


def test_exported_artifact_contains_expected_sections(tmp_path: Path) -> None:
    checkpoint = _fake_checkpoint()
    result = export_full_model_from_checkpoint(checkpoint, tmp_path)

    artifact = torch.load(result.full_model_path, map_location="cpu")

    assert set(artifact) == {
        "model_config",
        "segmentation_config",
        "state_dict",
        "export_metadata",
    }
    assert artifact["model_config"]["n_layers"] == 2
    assert artifact["segmentation_config"]["attention_segments"] == 2


def test_attention_segments_are_concatenated_in_order(tmp_path: Path) -> None:
    checkpoint = _fake_checkpoint()
    result = export_full_model_from_checkpoint(checkpoint, tmp_path)
    state = torch.load(result.full_model_path, map_location="cpu")["state_dict"]

    q_weight = state["layers.0.attention.q_proj.weight"]
    q_bias = state["layers.0.attention.q_proj.bias"]

    assert q_weight.shape == (4, 4)
    assert q_bias.shape == (4,)
    assert torch.all(q_weight[:2] == 1.0)
    assert torch.all(q_weight[2:] == 11.0)
    assert torch.all(q_bias[:2] == 4.0)
    assert torch.all(q_bias[2:] == 14.0)


def test_mlp_segments_are_assembled_by_hidden_dimension(tmp_path: Path) -> None:
    checkpoint = _fake_checkpoint()
    result = export_full_model_from_checkpoint(checkpoint, tmp_path)
    state = torch.load(result.full_model_path, map_location="cpu")["state_dict"]

    fc1_weight = state["layers.0.mlp.fc1.weight"]
    fc1_bias = state["layers.0.mlp.fc1.bias"]
    fc2_weight = state["layers.0.mlp.fc2.weight"]

    assert fc1_weight.shape == (6, 4)
    assert fc1_bias.shape == (6,)
    assert fc2_weight.shape == (4, 6)
    assert torch.all(fc1_weight[:3] == 1.0)
    assert torch.all(fc1_weight[3:] == 101.0)
    assert torch.all(fc2_weight[:, :3] == 3.0)
    assert torch.all(fc2_weight[:, 3:] == 103.0)


def test_non_segment_parameters_are_copied(tmp_path: Path) -> None:
    checkpoint = _fake_checkpoint()
    result = export_full_model_from_checkpoint(checkpoint, tmp_path)
    state = torch.load(result.full_model_path, map_location="cpu")["state_dict"]

    assert "token_embedding.weight" in state
    assert "layers.0.attention_output_projection.weight" in state
    assert torch.equal(state["layers.0.attention_output_projection.weight"], torch.eye(4))


def test_export_from_checkpoint_file(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "segmented_checkpoint.pt"
    torch.save(_fake_checkpoint(), checkpoint_path)

    output_dir = tmp_path / "exported"
    result = export_full_model_from_checkpoint(checkpoint_path, output_dir)

    assert result.full_model_path.exists()
    assert result.metadata_path.exists()


def test_exporter_custom_file_names(tmp_path: Path) -> None:
    exporter = FullModelExporter(
        FullModelExportConfig(
            output_dir=tmp_path,
            full_model_filename="custom.pt",
            metadata_filename="custom.yaml",
        )
    )

    result = exporter.export_from_checkpoint_dict(_fake_checkpoint())

    assert result.full_model_path.name == "custom.pt"
    assert result.metadata_path.name == "custom.yaml"


def test_missing_segments_entry_raises(tmp_path: Path) -> None:
    exporter = FullModelExporter(FullModelExportConfig(output_dir=tmp_path))

    try:
        exporter.export_from_checkpoint_dict({"model_config": {}})
    except KeyError as exc:
        assert "segments" in str(exc)
    else:
        raise AssertionError("Expected KeyError for missing segments.")


def test_missing_attention_segment_raises(tmp_path: Path) -> None:
    checkpoint = _fake_checkpoint()
    checkpoint["segments"].pop("layer_0.attention.1")

    exporter = FullModelExporter(FullModelExportConfig(output_dir=tmp_path))

    try:
        exporter.export_from_checkpoint_dict(checkpoint)
    except ValueError as exc:
        assert "attention segments" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing attention segment.")


def test_strict_missing_required_parameter_raises(tmp_path: Path) -> None:
    checkpoint = _fake_checkpoint()
    del checkpoint["segments"]["layer_0.mlp.0"]["fc1.weight"]

    exporter = FullModelExporter(FullModelExportConfig(output_dir=tmp_path, strict=True))

    try:
        exporter.export_from_checkpoint_dict(checkpoint)
    except KeyError as exc:
        assert "fc1.weight" in str(exc)
    else:
        raise AssertionError("Expected KeyError for missing fc1.weight.")
