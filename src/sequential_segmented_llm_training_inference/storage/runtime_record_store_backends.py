"""Runtime record storage backends for memory-minimal training.

InMemoryRecordBackend: current dict-based behaviour (default, no change).
DiskRecordBackend: spills each shared tensor to disk under a root_dir.

Keys use dot-separated naming: dots become path separators so that
    "layer_0.attention_input" → root_dir/layer_0/attention_input.pt
    "embedding_output"        → root_dir/embedding_output.pt
    "pre_final_norm"          → root_dir/pre_final_norm.pt

evict_layer(layer_id) removes every subdirectory/file whose name starts
with "layer_{layer_id}" — this covers all keys from one transformer layer.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import Tensor


class RuntimeRecordBackend(ABC):
    """Abstract base for shared-tensor storage backends."""

    @abstractmethod
    def put_tensor(self, key: str, tensor: Tensor) -> None: ...

    @abstractmethod
    def get_tensor(self, key: str) -> Tensor: ...

    @abstractmethod
    def has_tensor(self, key: str) -> bool: ...

    @abstractmethod
    def evict_layer(self, layer_id: int) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...


class InMemoryRecordBackend(RuntimeRecordBackend):
    """Dict-backed backend — identical to the original shared_tensors dict."""

    def __init__(self) -> None:
        self._store: dict[str, Tensor] = {}

    def put_tensor(self, key: str, tensor: Tensor) -> None:
        self._store[key] = tensor

    def get_tensor(self, key: str) -> Tensor:
        try:
            return self._store[key]
        except KeyError as exc:
            raise KeyError(f"Record tensor not found in memory: {key!r}") from exc

    def has_tensor(self, key: str) -> bool:
        return key in self._store

    def evict_layer(self, layer_id: int) -> None:
        prefix_dot = f"layer_{layer_id}."
        prefix_dunder = f"layer_{layer_id}__"
        to_del = [
            k for k in self._store
            if k.startswith(prefix_dot) or k.startswith(prefix_dunder)
        ]
        for k in to_del:
            del self._store[k]

    def clear(self) -> None:
        self._store.clear()

    # Expose raw store for the _assert_records_detached check in trainer
    @property
    def store(self) -> dict[str, Tensor]:
        return self._store


class DiskRecordBackend(RuntimeRecordBackend):
    """Writes each tensor as a .pt file under root_dir.

    Key → path mapping:
        "layer_0.attention_input"  → root_dir/layer_0/attention_input.pt
        "embedding_output"         → root_dir/embedding_output.pt

    evict_layer(layer_id) deletes every child of root_dir whose name
    starts with "layer_{layer_id}".
    clear() wipes and recreates root_dir.
    """

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        # Replace dots with / to build subdirectory hierarchy.
        safe = key.replace(".", "/")
        return self.root_dir / (safe + ".pt")

    def put_tensor(self, key: str, tensor: Tensor) -> None:
        path = self._key_to_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor.cpu().detach().clone(), path)

    def get_tensor(self, key: str) -> Tensor:
        path = self._key_to_path(key)
        if not path.exists():
            raise KeyError(
                f"Record tensor not found on disk: {key!r} (expected {path})"
            )
        return torch.load(path, map_location="cpu", weights_only=False)

    def has_tensor(self, key: str) -> bool:
        return self._key_to_path(key).exists()

    def evict_layer(self, layer_id: int) -> None:
        prefix = f"layer_{layer_id}"
        if not self.root_dir.exists():
            return
        for child in list(self.root_dir.iterdir()):
            if child.name.startswith(prefix):
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)

    def clear(self) -> None:
        if self.root_dir.exists():
            shutil.rmtree(self.root_dir, ignore_errors=True)
        self.root_dir.mkdir(parents=True, exist_ok=True)
