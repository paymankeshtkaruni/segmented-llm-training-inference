"""Tests for Phase 21 CLI commands."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml

from sequential_segmented_llm_training_inference.cli import (
    export_full_model,
    export_protocol,
    infer,
    test as test_cli,
    train,
    validate,
)
from sequential_segmented_llm_training_inference.cli.common import parse_override_items


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _install_fake_module(module_name: str, attr_name: str, calls: list[dict]) -> None:
    module = types.ModuleType(module_name)

    def fake_callable(**kwargs):
        calls.append(kwargs)
        return {"ok": True, "kwargs": kwargs}

    setattr(module, attr_name, fake_callable)
    sys.modules[module_name] = module


def test_parse_override_items() -> None:
    assert parse_override_items(["a=1", "b=two"]) == {"a": "1", "b": "two"}


def test_parse_override_rejects_invalid_item() -> None:
    with pytest.raises(Exception, match="KEY=VALUE"):
        parse_override_items(["bad"])


def test_train_cli_dispatches_to_train_from_config(tmp_path: Path) -> None:
    config = _write_yaml(tmp_path / "config.yaml", {"training": {"epochs": 1}})
    calls: list[dict] = []
    _install_fake_module(
        "sequential_segmented_llm_training_inference.training.trainer",
        "train_from_config",
        calls,
    )

    result = train.main(["--config", str(config), "--override", "x=y"])

    assert result["ok"] is True
    assert calls
    assert calls[0]["config_path"] == config
    assert calls[0]["config"] == {"training": {"epochs": 1}}
    assert calls[0]["overrides"] == {"x": "y"}


def test_validate_cli_dispatches_to_validate_from_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = _write_yaml(tmp_path / "config.yaml", {"x": 1})
    calls: list[dict] = []
    _install_fake_module(
        "sequential_segmented_llm_training_inference.training.validator",
        "validate_from_checkpoint",
        calls,
    )

    result = validate.main(
        ["--checkpoint", str(checkpoint), "--config", str(config), "--split", "validation"]
    )

    assert result["ok"] is True
    assert calls[0]["checkpoint"] == checkpoint
    assert calls[0]["split"] == "validation"


def test_test_cli_dispatches_to_test_from_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    calls: list[dict] = []
    _install_fake_module(
        "sequential_segmented_llm_training_inference.training.tester",
        "test_from_checkpoint",
        calls,
    )

    result = test_cli.main(["--checkpoint", str(checkpoint)])

    assert result["ok"] is True
    assert calls[0]["checkpoint"] == checkpoint
    assert calls[0]["split"] == "test"


def test_export_full_model_cli_dispatches(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output = tmp_path / "exports" / "full_model"
    calls: list[dict] = []
    _install_fake_module(
        "sequential_segmented_llm_training_inference.export.full_model_exporter",
        "export_full_model",
        calls,
    )

    result = export_full_model.main(["--checkpoint", str(checkpoint), "--out", str(output)])

    assert result["ok"] is True
    assert calls[0]["checkpoint"] == checkpoint
    assert calls[0]["output"] == output


def test_export_protocol_cli_dispatches(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    output = tmp_path / "exports" / "protocol.yaml"
    calls: list[dict] = []
    _install_fake_module(
        "sequential_segmented_llm_training_inference.export.protocol_exporter",
        "export_protocol_yaml",
        calls,
    )

    result = export_protocol.main(["--run-dir", str(run_dir), "--out", str(output)])

    assert result["ok"] is True
    assert calls[0]["checkpoint"] == run_dir
    assert calls[0]["output"] == output


def test_infer_cli_dispatches_when_inferencer_exists(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    calls: list[dict] = []
    _install_fake_module(
        "sequential_segmented_llm_training_inference.training.inferencer",
        "infer_from_checkpoint",
        calls,
    )

    result = infer.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--prompt",
            "hello",
            "--max-new-tokens",
            "4",
        ]
    )

    assert result["ok"] is True
    assert calls[0]["checkpoint"] == checkpoint
    assert calls[0]["prompt"] == "hello"
    assert calls[0]["max_new_tokens"] == 4


def test_train_help_exits_cleanly(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        train.main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "segllm train" in captured.out
