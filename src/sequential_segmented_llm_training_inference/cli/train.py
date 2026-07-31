"""CLI entry point for segmented training."""

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
    parse_override_items,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="segllm train",
        description="Run sequential segmented LLM training.",
    )
    parser.add_argument("--config", required=True, type=existing_path_arg)
    parser.add_argument("--resume", type=existing_path_arg, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Optional KEY=VALUE override passed to the trainer.",
    )
    return parser


def run(args: argparse.Namespace) -> object:
    _, _, target = first_available(
        [
            (f"{PACKAGE_NAME}.training.trainer", "train_from_config"),
            (f"{PACKAGE_NAME}.training.trainer", "Trainer"),
        ]
    )
    config = load_yaml(args.config)
    return call_function_or_class(
        target,
        config_path=args.config,
        checkpoint=args.resume,
        config=config,
        extra_kwargs={
            "run_dir": args.run_dir,
            "overrides": parse_override_items(args.override),
        },
        action_names=("train", "run"),
    )


def main(argv: Sequence[str] | None = None) -> object:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
