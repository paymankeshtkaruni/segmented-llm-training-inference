"""Final integration and package audit utilities.

Phase 24 scope:
- Verify expected package files/modules are present.
- Verify important modules import successfully.
- Verify example YAML configs are readable and structurally valid.
- Optionally run pytest and capture the result.
- Write a machine-readable and human-readable audit report.

This phase intentionally does not add new model/training features.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


PACKAGE_NAME = "sequential_segmented_llm_training_inference"


EXPECTED_MODULES: tuple[str, ...] = (
    f"{PACKAGE_NAME}",
    f"{PACKAGE_NAME}.config",
    f"{PACKAGE_NAME}.config.model_config",
    f"{PACKAGE_NAME}.config.segmentation_config",
    f"{PACKAGE_NAME}.config.training_config",
    f"{PACKAGE_NAME}.config.storage_config",
    f"{PACKAGE_NAME}.config.optimizer_config",
    f"{PACKAGE_NAME}.config.runtime_config",
    f"{PACKAGE_NAME}.config.export_config",
    f"{PACKAGE_NAME}.config.protocol_config",
    f"{PACKAGE_NAME}.segments.segment_ids",
    f"{PACKAGE_NAME}.segments.attention_segment",
    f"{PACKAGE_NAME}.segments.mlp_segment",
    f"{PACKAGE_NAME}.segments.segment_factory",
    f"{PACKAGE_NAME}.model.embeddings",
    f"{PACKAGE_NAME}.model.layer_norms",
    f"{PACKAGE_NAME}.model.output_head",
    f"{PACKAGE_NAME}.storage.segment_store",
    f"{PACKAGE_NAME}.storage.disk_segment_store",
    f"{PACKAGE_NAME}.storage.cpu_ram_segment_store",
    f"{PACKAGE_NAME}.storage.runtime_records",
    f"{PACKAGE_NAME}.storage.checkpoint_store",
    f"{PACKAGE_NAME}.execution.segment_loader",
    f"{PACKAGE_NAME}.execution.forward_engine",
    f"{PACKAGE_NAME}.execution.backward_engine",
    f"{PACKAGE_NAME}.execution.rng",
    f"{PACKAGE_NAME}.optimization.segment_optimizer",
    f"{PACKAGE_NAME}.optimization.segmentwise_sgd",
    f"{PACKAGE_NAME}.optimization.segmentwise_adamw",
    f"{PACKAGE_NAME}.training.losses",
    f"{PACKAGE_NAME}.training.trainer",
    f"{PACKAGE_NAME}.training.validator",
    f"{PACKAGE_NAME}.training.tester",
    f"{PACKAGE_NAME}.training.profiler",
    f"{PACKAGE_NAME}.export.full_model_exporter",
    f"{PACKAGE_NAME}.export.protocol_exporter",
    f"{PACKAGE_NAME}.export.protocol_importer",
    f"{PACKAGE_NAME}.cli.train",
    f"{PACKAGE_NAME}.cli.validate",
    f"{PACKAGE_NAME}.cli.test",
    f"{PACKAGE_NAME}.cli.export_full_model",
    f"{PACKAGE_NAME}.cli.export_protocol",
)

EXPECTED_CONFIG_FILES: tuple[str, ...] = (
    "configs/example_cpu_disk.yaml",
    "configs/example_gpu_disk.yaml",
    "configs/example_gpu_cpu_offload.yaml",
)


@dataclass(slots=True)
class AuditResult:
    """Container for Phase 24 package audit results."""

    repo_root: str
    started_at_unix: float
    finished_at_unix: float | None = None
    passed: bool = False
    checks: dict[str, bool] = field(default_factory=dict)
    imported_modules: list[str] = field(default_factory=list)
    failed_imports: dict[str, str] = field(default_factory=dict)
    config_files_checked: list[str] = field(default_factory=list)
    config_errors: dict[str, str] = field(default_factory=dict)
    pytest_returncode: int | None = None
    pytest_output_tail: str | None = None
    notes: list[str] = field(default_factory=list)

    def finish(self) -> None:
        self.finished_at_unix = time.time()
        self.passed = (
            all(self.checks.values())
            and not self.failed_imports
            and not self.config_errors
            and self.pytest_returncode in (None, 0)
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"# Phase 24 Integration Audit — {status}",
            "",
            f"- Repository: `{self.repo_root}`",
            f"- Started: `{self.started_at_unix}`",
            f"- Finished: `{self.finished_at_unix}`",
            "",
            "## Checks",
            "",
        ]

        for name, ok in sorted(self.checks.items()):
            marker = "✅" if ok else "❌"
            lines.append(f"- {marker} `{name}`")

        lines.extend(["", "## Module Imports", ""])
        if self.failed_imports:
            lines.append("### Failed imports")
            for module, error in sorted(self.failed_imports.items()):
                lines.append(f"- `{module}`: `{error}`")
        else:
            lines.append(f"All {len(self.imported_modules)} expected modules imported.")

        lines.extend(["", "## Config Files", ""])
        if self.config_errors:
            for path, error in sorted(self.config_errors.items()):
                lines.append(f"- ❌ `{path}`: `{error}`")
        else:
            lines.append(f"All {len(self.config_files_checked)} expected config files loaded.")

        if self.pytest_returncode is not None:
            lines.extend(
                [
                    "",
                    "## Pytest",
                    "",
                    f"- Return code: `{self.pytest_returncode}`",
                    "",
                    "```text",
                    self.pytest_output_tail or "",
                    "```",
                ]
            )

        if self.notes:
            lines.extend(["", "## Notes", ""])
            for note in self.notes:
                lines.append(f"- {note}")

        lines.append("")
        return "\\n".join(lines)


def _ensure_src_on_path(repo_root: Path) -> None:
    src = repo_root / "src"
    src_text = str(src)
    if src.exists() and src_text not in sys.path:
        sys.path.insert(0, src_text)


def _check_repo_layout(repo_root: Path, result: AuditResult) -> None:
    result.checks["repo_root_exists"] = repo_root.exists()
    result.checks["pyproject_exists"] = (repo_root / "pyproject.toml").exists()
    result.checks["src_package_exists"] = (repo_root / "src" / PACKAGE_NAME).exists()
    result.checks["tests_dir_exists"] = (repo_root / "tests").exists()
    result.checks["configs_dir_exists"] = (repo_root / "configs").exists()


def _check_module_imports(repo_root: Path, result: AuditResult) -> None:
    _ensure_src_on_path(repo_root)

    for module_name in EXPECTED_MODULES:
        try:
            importlib.import_module(module_name)
            result.imported_modules.append(module_name)
        except Exception as exc:  # noqa: BLE001 - audit must capture all import failures
            result.failed_imports[module_name] = f"{type(exc).__name__}: {exc}"


def _validate_config_structure(config: dict[str, Any], path: str) -> None:
    required_top_level = {
        "device",
        "execution",
        "segment_storage",
        "model",
        "segmentation",
        "training",
        "optimizer",
        "runtime_records",
        "loss",
    }
    missing = required_top_level - set(config)
    if missing:
        raise ValueError(f"missing top-level keys: {sorted(missing)}")

    if config["execution"].get("mode") != "strict_single_segment":
        raise ValueError("execution.mode must be strict_single_segment")
    if int(config["execution"].get("max_active_segments", -1)) != 1:
        raise ValueError("execution.max_active_segments must be 1")

    attention_segments = int(config["segmentation"].get("attention_segments", 0))
    mlp_chunks = int(config["segmentation"].get("mlp_chunks", 0))
    if attention_segments <= 1:
        raise ValueError("attention_segments must be > 1")
    if mlp_chunks <= 1:
        raise ValueError("mlp_chunks must be > 1")

    n_heads = int(config["model"].get("n_heads", 0))
    d_model = int(config["model"].get("d_model", 0))
    d_ff = int(config["model"].get("d_ff", 0))
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    if n_heads % attention_segments != 0:
        raise ValueError("n_heads must be divisible by attention_segments")
    if d_ff % mlp_chunks != 0:
        raise ValueError("d_ff must be divisible by mlp_chunks")

    device = config["device"]
    backend = config["segment_storage"].get("backend")
    if device == "cpu" and backend != "disk_streaming":
        raise ValueError("CPU configs must use disk_streaming backend")
    if device == "cuda" and backend not in {"disk_streaming", "cpu_ram_offload"}:
        raise ValueError("CUDA configs must use disk_streaming or cpu_ram_offload")

    if path.endswith("example_cpu_disk.yaml") and device != "cpu":
        raise ValueError("example_cpu_disk.yaml must use device=cpu")
    if path.endswith("example_gpu_disk.yaml") and backend != "disk_streaming":
        raise ValueError("example_gpu_disk.yaml must use disk_streaming")
    if path.endswith("example_gpu_cpu_offload.yaml") and backend != "cpu_ram_offload":
        raise ValueError("example_gpu_cpu_offload.yaml must use cpu_ram_offload")


def _check_config_files(repo_root: Path, result: AuditResult) -> None:
    for relative in EXPECTED_CONFIG_FILES:
        path = repo_root / relative
        if not path.exists():
            result.config_errors[relative] = "file does not exist"
            continue

        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("YAML root must be a mapping")
            _validate_config_structure(data, relative)
            result.config_files_checked.append(relative)
        except Exception as exc:  # noqa: BLE001
            result.config_errors[relative] = f"{type(exc).__name__}: {exc}"


def _run_pytest(repo_root: Path, result: AuditResult, pytest_args: list[str]) -> None:
    command = [sys.executable, "-m", "pytest", *pytest_args]
    env = dict(**__import__("os").environ)
    src = str(repo_root / "src")
    env["PYTHONPATH"] = src + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    process = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
    )
    result.pytest_returncode = process.returncode
    output = process.stdout
    result.pytest_output_tail = "\\n".join(output.splitlines()[-80:])


def write_audit_reports(result: AuditResult, output_dir: Path) -> None:
    """Write JSON and Markdown audit reports."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase24_audit_report.json").write_text(
        result.to_json() + "\\n",
        encoding="utf-8",
    )
    (output_dir / "phase24_audit_report.md").write_text(
        result.to_markdown(),
        encoding="utf-8",
    )


def run_full_audit(
    repo_root: str | Path,
    *,
    run_pytest: bool = True,
    pytest_args: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> AuditResult:
    """Run the final integration audit.

    Args:
        repo_root: Repository root.
        run_pytest: Whether to run pytest.
        pytest_args: Pytest arguments. Defaults to ["tests"].
        output_dir: Report output directory. Defaults to repo_root / "audit_reports".

    Returns:
        AuditResult with all collected information.
    """

    root = Path(repo_root).expanduser().resolve()
    result = AuditResult(repo_root=str(root), started_at_unix=time.time())

    _check_repo_layout(root, result)
    if result.checks.get("src_package_exists"):
        _check_module_imports(root, result)
    else:
        result.notes.append("Skipped module imports because src package directory was missing.")

    if result.checks.get("configs_dir_exists"):
        _check_config_files(root, result)
    else:
        result.notes.append("Skipped config validation because configs directory was missing.")

    if run_pytest:
        _run_pytest(root, result, pytest_args or ["tests"])

    result.finish()

    report_dir = Path(output_dir).expanduser().resolve() if output_dir else root / "audit_reports"
    write_audit_reports(result, report_dir)

    return result
