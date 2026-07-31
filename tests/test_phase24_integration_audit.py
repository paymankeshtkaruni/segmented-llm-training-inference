"""Tests for Phase 24 final integration audit utilities."""

from __future__ import annotations

from pathlib import Path

import yaml

from sequential_segmented_llm_training_inference.integration.audit import (
    EXPECTED_CONFIG_FILES,
    EXPECTED_MODULES,
    run_full_audit,
)


def test_expected_module_list_is_not_empty() -> None:
    assert EXPECTED_MODULES
    assert any(module.endswith(".execution.forward_engine") for module in EXPECTED_MODULES)
    assert any(module.endswith(".execution.backward_engine") for module in EXPECTED_MODULES)
    assert any(module.endswith(".training.trainer") for module in EXPECTED_MODULES)


def test_expected_config_files_are_declared() -> None:
    assert set(EXPECTED_CONFIG_FILES) == {
        "configs/example_cpu_disk.yaml",
        "configs/example_gpu_disk.yaml",
        "configs/example_gpu_cpu_offload.yaml",
    }


def test_audit_can_run_without_pytest_and_write_reports(tmp_path: Path) -> None:
    repo = Path.cwd()
    output_dir = tmp_path / "reports"

    result = run_full_audit(repo, run_pytest=False, output_dir=output_dir)

    assert result.checks["repo_root_exists"]
    assert output_dir.joinpath("phase24_audit_report.json").exists()
    assert output_dir.joinpath("phase24_audit_report.md").exists()


def test_example_configs_are_valid_yaml_if_present() -> None:
    repo = Path.cwd()
    for relative in EXPECTED_CONFIG_FILES:
        path = repo / relative
        if not path.exists():
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert data["execution"]["mode"] == "strict_single_segment"
        assert data["execution"]["max_active_segments"] == 1
        assert data["segmentation"]["attention_segments"] > 1
        assert data["segmentation"]["mlp_chunks"] > 1
