"""Shared helpers for Phase 21 CLI entry points.

The CLI layer intentionally stays thin. It parses arguments, loads YAML
configuration/protocol files, then dispatches to the package-level trainer,
validator, tester, and exporters implemented in earlier phases.

The dispatch helpers accept several conventional callable/class names so the
CLI remains stable even if an earlier phase exposes either a function-style or
class-style API.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import yaml


PACKAGE_NAME = "sequential_segmented_llm_training_inference"


class CLIError(RuntimeError):
    """Raised when a CLI command cannot dispatch to the requested component."""


def path_arg(value: str) -> Path:
    """Convert a command-line path argument to an expanded Path."""

    return Path(value).expanduser()


def existing_path_arg(value: str) -> Path:
    """Convert and validate an existing command-line path."""

    path = path_arg(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"path does not exist: {path}")
    return path


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for an output path."""

    path.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""

    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise CLIError(f"YAML file must contain a mapping at top level: {path}")

    return data


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write a dictionary as YAML."""

    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def parse_override_items(items: Sequence[str] | None) -> dict[str, str]:
    """Parse repeated KEY=VALUE overrides.

    The CLI leaves values as strings because typed validation belongs to the
    config/protocol layer.
    """

    overrides: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise CLIError(f"override must have form KEY=VALUE, got {item!r}")
        key, value = item.split("=", maxsplit=1)
        key = key.strip()
        if not key:
            raise CLIError(f"override key must not be empty: {item!r}")
        overrides[key] = value
    return overrides


def _import_attr(module_name: str, attr_name: str) -> Any | None:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    return getattr(module, attr_name, None)


def first_available(candidates: Iterable[tuple[str, str]]) -> tuple[str, str, Any]:
    """Return the first importable attribute among candidate module/attr pairs."""

    checked: list[str] = []
    for module_name, attr_name in candidates:
        checked.append(f"{module_name}:{attr_name}")
        attr = _import_attr(module_name, attr_name)
        if attr is not None:
            return module_name, attr_name, attr

    checked_text = "\n  - ".join(checked)
    raise CLIError(f"No supported implementation found. Checked:\n  - {checked_text}")


def call_function_or_class(
    target: Any,
    *,
    config_path: Path | None = None,
    checkpoint: Path | None = None,
    output: Path | None = None,
    protocol: Path | None = None,
    config: dict[str, Any] | None = None,
    extra_kwargs: dict[str, Any] | None = None,
    action_names: Sequence[str] = ("run",),
) -> Any:
    """Call either a function target or instantiate/call a class target."""

    kwargs: dict[str, Any] = {}
    if config_path is not None:
        kwargs["config_path"] = config_path
    if checkpoint is not None:
        kwargs["checkpoint"] = checkpoint
    if output is not None:
        kwargs["output"] = output
    if protocol is not None:
        kwargs["protocol"] = protocol
    if config is not None:
        kwargs["config"] = config
    kwargs.update(extra_kwargs or {})

    if callable(target) and not isinstance(target, type):
        try:
            return target(**kwargs)
        except TypeError:
            # Fallback for simpler function APIs.
            positional = [v for v in (config_path, checkpoint, output, protocol) if v is not None]
            return target(*positional)

    if isinstance(target, type):
        instance = _instantiate_component(target, kwargs)
        for action_name in action_names:
            action = getattr(instance, action_name, None)
            if callable(action):
                try:
                    return action()
                except TypeError:
                    return action(**kwargs)
        raise CLIError(
            f"Class {target.__name__} does not expose any action method from "
            f"{tuple(action_names)!r}."
        )

    raise CLIError(f"Target is not callable: {target!r}")


def _instantiate_component(component_cls: type, kwargs: dict[str, Any]) -> Any:
    """Instantiate a class target using common factory conventions."""

    config_path = kwargs.get("config_path")
    config = kwargs.get("config")
    checkpoint = kwargs.get("checkpoint")
    output = kwargs.get("output")
    protocol = kwargs.get("protocol")

    if config_path is not None and hasattr(component_cls, "from_config_path"):
        return component_cls.from_config_path(config_path)
    if config is not None and hasattr(component_cls, "from_config"):
        return component_cls.from_config(config)
    if checkpoint is not None and hasattr(component_cls, "from_checkpoint"):
        return component_cls.from_checkpoint(checkpoint)

    try:
        return component_cls(**kwargs)
    except TypeError:
        pass

    for candidate in (config_path, config, checkpoint, output, protocol):
        if candidate is not None:
            try:
                return component_cls(candidate)
            except TypeError:
                continue

    return component_cls()
