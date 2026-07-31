"""RNG state capture and restoration for deterministic recomputation.

Phase 13 scope:
- Capture and restore Python, Torch CPU, and optional Torch CUDA RNG state.
- Provide a small tracker/context API for forward/backward recomputation.
- Keep this module independent of the forward/backward engines so it can be used
  by segment records and later training/checkpointing code.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import random
from typing import Any, Iterator, Optional

import torch
from torch import Tensor


PythonRandomState = object


@dataclass(slots=True)
class RngState:
    """Container for reproducible RNG state used during segment recomputation.

    Attributes:
        python_state: State returned by ``random.getstate()``.
        torch_cpu_state: Tensor returned by ``torch.random.get_rng_state()``.
        torch_cuda_states: Optional list returned by ``torch.cuda.get_rng_state_all()``.
            It is ``None`` when CUDA is not captured or unavailable.
    """

    python_state: PythonRandomState
    torch_cpu_state: Tensor
    torch_cuda_states: Optional[list[Tensor]] = None

    @property
    def has_cuda_state(self) -> bool:
        """Return True if CUDA RNG states were captured."""

        return self.torch_cuda_states is not None

    def clone(self) -> "RngState":
        """Return a deep-enough clone for safe storage and reuse."""

        cuda_states = None
        if self.torch_cuda_states is not None:
            cuda_states = [state.clone() for state in self.torch_cuda_states]
        return RngState(
            python_state=self.python_state,
            torch_cpu_state=self.torch_cpu_state.clone(),
            torch_cuda_states=cuda_states,
        )

    def to_checkpoint(self) -> dict[str, Any]:
        """Return a torch-save/YAML-protocol-compatible checkpoint dictionary.

        Python's native random state is not human-readable YAML, but it is safe to
        preserve in torch checkpoints or pickle-based manifests. The YAML protocol
        should store whether RNG capture was enabled and where the binary checkpoint
        stores the actual state.
        """

        return {
            "python_state": self.python_state,
            "torch_cpu_state": self.torch_cpu_state.clone(),
            "torch_cuda_states": (
                None
                if self.torch_cuda_states is None
                else [state.clone() for state in self.torch_cuda_states]
            ),
        }

    @classmethod
    def from_checkpoint(cls, data: dict[str, Any]) -> "RngState":
        """Reconstruct an RNG state from :meth:`to_checkpoint` output."""

        required = {"python_state", "torch_cpu_state", "torch_cuda_states"}
        missing = required - set(data)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Missing RNG checkpoint field(s): {missing_text}")

        torch_cpu_state = data["torch_cpu_state"]
        if not isinstance(torch_cpu_state, Tensor):
            raise TypeError("torch_cpu_state must be a torch.Tensor.")

        torch_cuda_states = data["torch_cuda_states"]
        if torch_cuda_states is not None:
            if not isinstance(torch_cuda_states, list):
                raise TypeError("torch_cuda_states must be None or a list of tensors.")
            if not all(isinstance(state, Tensor) for state in torch_cuda_states):
                raise TypeError("all torch_cuda_states entries must be tensors.")

        return cls(
            python_state=data["python_state"],
            torch_cpu_state=torch_cpu_state.clone(),
            torch_cuda_states=(
                None if torch_cuda_states is None else [s.clone() for s in torch_cuda_states]
            ),
        )


def cuda_rng_capture_available() -> bool:
    """Return True when CUDA is available and can expose RNG state."""

    return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)


def capture_rng_state(*, include_cuda: bool = True) -> RngState:
    """Capture Python, Torch CPU, and optional Torch CUDA RNG state."""

    cuda_states: Optional[list[Tensor]] = None
    if include_cuda and cuda_rng_capture_available():
        cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]

    return RngState(
        python_state=random.getstate(),
        torch_cpu_state=torch.random.get_rng_state().clone(),
        torch_cuda_states=cuda_states,
    )


def restore_rng_state(state: RngState, *, restore_cuda: bool = True) -> None:
    """Restore a previously captured RNG state."""

    if not isinstance(state, RngState):
        raise TypeError(f"state must be RngState, got {type(state).__name__}.")

    random.setstate(state.python_state)  # type: ignore[arg-type]
    torch.random.set_rng_state(state.torch_cpu_state.clone())

    if restore_cuda and state.torch_cuda_states is not None:
        if not cuda_rng_capture_available():
            raise RuntimeError("CUDA RNG state was captured, but CUDA is unavailable now.")
        if len(state.torch_cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "Captured CUDA RNG state count does not match current CUDA device count: "
                f"captured={len(state.torch_cuda_states)}, current={torch.cuda.device_count()}."
            )
        torch.cuda.set_rng_state_all([s.clone() for s in state.torch_cuda_states])


@contextmanager
def restored_rng_state(
    state: RngState,
    *,
    restore_cuda: bool = True,
    preserve_current: bool = True,
) -> Iterator[None]:
    """Temporarily restore RNG state inside a context.

    Args:
        state: Captured RNG state to restore for the context body.
        restore_cuda: Whether CUDA state should be restored when available.
        preserve_current: If True, restore the caller's current RNG state after the
            context exits. This is useful for local recomputation checks.
    """

    previous = capture_rng_state(include_cuda=restore_cuda) if preserve_current else None
    restore_rng_state(state, restore_cuda=restore_cuda)
    try:
        yield
    finally:
        if previous is not None:
            restore_rng_state(previous, restore_cuda=restore_cuda)


class RngStateTracker:
    """Small helper for storing named RNG states during segmented execution."""

    def __init__(self) -> None:
        self._states: dict[str, RngState] = {}

    def capture(self, key: str, *, include_cuda: bool = True) -> RngState:
        """Capture and store an RNG state under a deterministic key."""

        self._validate_key(key)
        state = capture_rng_state(include_cuda=include_cuda)
        self._states[key] = state.clone()
        return state

    def set(self, key: str, state: RngState) -> None:
        """Store a cloned RNG state under a key."""

        self._validate_key(key)
        if not isinstance(state, RngState):
            raise TypeError(f"state must be RngState, got {type(state).__name__}.")
        self._states[key] = state.clone()

    def get(self, key: str) -> RngState:
        """Return a cloned RNG state by key."""

        self._validate_key(key)
        try:
            return self._states[key].clone()
        except KeyError as exc:
            raise KeyError(f"No RNG state stored for key {key!r}.") from exc

    def restore(self, key: str, *, restore_cuda: bool = True) -> None:
        """Restore a stored RNG state by key."""

        restore_rng_state(self.get(key), restore_cuda=restore_cuda)

    def has(self, key: str) -> bool:
        """Return True if the key exists."""

        self._validate_key(key)
        return key in self._states

    def clear(self) -> None:
        """Remove all stored RNG states."""

        self._states.clear()

    def keys(self) -> tuple[str, ...]:
        """Return stored keys in deterministic sorted order."""

        return tuple(sorted(self._states))

    def __len__(self) -> int:
        return len(self._states)

    @staticmethod
    def _validate_key(key: str) -> None:
        if not isinstance(key, str):
            raise TypeError(f"key must be a string, got {type(key).__name__}.")
        if not key:
            raise ValueError("key must not be empty.")
