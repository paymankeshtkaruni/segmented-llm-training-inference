"""CLI entry point for exporting a full model from trained segments."""

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
        prog="segllm export-full-model",
        description="Export a normal full model from a segmented checkpoint.",
    )
    parser.add_argument("--checkpoint", required=True, type=existing_path_arg)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def run(args: argparse.Namespace) -> object:
    ensure_parent_dir(args.out)
    _, _, target = first_available(
        [
            (f"{PACKAGE_NAME}.export.full_model_exporter", "export_full_model"),
            (f"{PACKAGE_NAME}.export.full_model_exporter", "export_full_model_from_checkpoint"),
            (f"{PACKAGE_NAME}.export.full_model_exporter", "FullModelExporter"),
        ]
    )
    return call_function_or_class(
        target,
        checkpoint=args.checkpoint,
        output=args.out,
        action_names=("export", "run"),
    )


def main(argv: Sequence[str] | None = None) -> object:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
