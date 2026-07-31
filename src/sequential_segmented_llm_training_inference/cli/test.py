"""CLI entry point for final segmented testing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from sequential_segmented_llm_training_inference.cli.common import (
    PACKAGE_NAME,
    call_function_or_class,
    existing_path_arg,
    first_available,
    load_yaml,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="segllm test",
        description="Run final segmented test using segmented forward only.",
    )
    parser.add_argument("--checkpoint", required=True, type=existing_path_arg)
    parser.add_argument("--config", type=existing_path_arg, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def run(args: argparse.Namespace) -> object:
    _, _, target = first_available(
        [
            (f"{PACKAGE_NAME}.training.tester", "test_from_checkpoint"),
            (f"{PACKAGE_NAME}.training.tester", "Tester"),
        ]
    )
    config = load_yaml(args.config) if args.config is not None else None
    return call_function_or_class(
        target,
        config_path=args.config,
        checkpoint=args.checkpoint,
        output=args.output,
        config=config,
        extra_kwargs={"split": args.split},
        action_names=("test", "run"),
    )


def main(argv: Sequence[str] | None = None) -> object:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
