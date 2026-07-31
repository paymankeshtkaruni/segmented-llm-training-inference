"""CLI entry point for Phase 24 final integration audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from sequential_segmented_llm_training_inference.integration.audit import run_full_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final integration/package audit.")
    parser.add_argument(
        "--repo",
        default=".",
        help="Repository root. Default: current directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for audit reports. Default: <repo>/audit_reports.",
    )
    parser.add_argument(
        "--no-pytest",
        action="store_true",
        help="Skip pytest execution.",
    )
    parser.add_argument(
        "--pytest-args",
        nargs="*",
        default=["tests"],
        help="Arguments passed to pytest. Default: tests",
    )
    args = parser.parse_args()

    result = run_full_audit(
        Path(args.repo),
        run_pytest=not args.no_pytest,
        pytest_args=args.pytest_args,
        output_dir=args.output_dir,
    )

    print(result.to_markdown())
    raise SystemExit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
