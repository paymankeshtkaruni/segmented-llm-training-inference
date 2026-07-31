"""CLI entry point for segmented inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from sequential_segmented_llm_training_inference.cli.common import (
    PACKAGE_NAME,
    call_function_or_class,
    existing_path_arg,
    first_available,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="segllm infer",
        description="Run segmented LLM inference/generation.",
    )
    parser.add_argument("--checkpoint", required=True, type=existing_path_arg)
    parser.add_argument("--protocol", default=None, type=existing_path_arg)
    parser.add_argument("--prompt", default="", help="Prompt text or serialized input.")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample", action="store_true", help="Enable token sampling.")
    return parser


def run(args: argparse.Namespace) -> object:
    _, _, target = first_available(
        [
            (f"{PACKAGE_NAME}.training.inferencer", "infer_from_checkpoint"),
            (f"{PACKAGE_NAME}.inference.generation", "SegmentedAutoregressiveGenerator"),
        ]
    )
    return call_function_or_class(
        target,
        checkpoint=args.checkpoint,
        extra_kwargs={
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "do_sample": args.sample,
        },
        action_names=("infer", "generate", "run"),
    )


def main(argv: Sequence[str] | None = None) -> object:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
