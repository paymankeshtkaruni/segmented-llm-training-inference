"""CLI entry point for YAML protocol export."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from sequential_segmented_llm_training_inference.cli.common import (
    PACKAGE_NAME,
    call_function_or_class,
    ensure_parent_dir,
    existing_path_arg,
    first_available,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="segllm export-protocol",
        description="Export the full sequential segmented training protocol as YAML.",
    )
    parser.add_argument("--run-dir", required=True, type=existing_path_arg)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def run(args: argparse.Namespace) -> object:
    ensure_parent_dir(args.out)
    _, _, target = first_available(
        [
            (f"{PACKAGE_NAME}.export.protocol_exporter", "export_protocol_yaml"),
            (f"{PACKAGE_NAME}.export.protocol_exporter", "export_protocol"),
            (f"{PACKAGE_NAME}.export.protocol_exporter", "ProtocolExporter"),
        ]
    )
    return call_function_or_class(
        target,
        checkpoint=args.run_dir,
        output=args.out,
        action_names=("export", "run"),
    )


def main(argv: Sequence[str] | None = None) -> object:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
