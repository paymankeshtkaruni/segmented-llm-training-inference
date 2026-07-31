"""Disk-backed per-segment gradient store for memory-minimal after_full_backward.

Supports read-modify-write accumulation: if a segment's gradient already
exists on disk, loading + adding + saving keeps only one segment resident
at a time.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import torch
from torch import Tensor

from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId


class DiskGradientStore:
    """Stores per-segment parameter gradients on disk.

    Each segment's gradient dict is saved as a single .pt file.
    Accumulation is performed by read-modify-write (load existing + add new + save).
    """

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._keys: set[SegmentId] = set()

    def _path(self, segment_id: SegmentId) -> Path:
        return self.root_dir / f"{segment_id.to_path_name()}.pt"

    def accumulate(self, segment_id: SegmentId, gradients: dict[str, Tensor]) -> None:
        """Add gradients for segment_id to disk (read-modify-write if exists)."""
        path = self._path(segment_id)
        if path.exists():
            existing: dict[str, Tensor] = torch.load(
                path, map_location="cpu", weights_only=False
            )
            merged: dict[str, Tensor] = {}
            for name, grad in gradients.items():
                g_cpu = grad.cpu().detach()
                if name in existing:
                    merged[name] = existing[name] + g_cpu
                else:
                    merged[name] = g_cpu.clone()
            # Keep any keys in existing that are not in gradients (shouldn't
            # happen in normal use, but be safe).
            for name in existing:
                if name not in merged:
                    merged[name] = existing[name]
            torch.save(merged, path)
        else:
            torch.save(
                {name: g.cpu().detach().clone() for name, g in gradients.items()},
                path,
            )
        self._keys.add(segment_id)

    def load(self, segment_id: SegmentId) -> dict[str, Tensor]:
        """Load gradient dict for segment_id from disk."""
        path = self._path(segment_id)
        if not path.exists():
            raise KeyError(f"No gradient on disk for {segment_id}")
        return torch.load(path, map_location="cpu", weights_only=False)

    def evict(self, segment_id: SegmentId) -> None:
        """Delete the on-disk gradient file for segment_id."""
        path = self._path(segment_id)
        path.unlink(missing_ok=True)
        self._keys.discard(segment_id)

    def save_scaled(self, segment_id: SegmentId, scaled: dict[str, Tensor]) -> None:
        """Overwrite the gradient file for segment_id with pre-scaled values."""
        torch.save(scaled, self._path(segment_id))

    def __iter__(self) -> Iterator[SegmentId]:
        return iter(sorted(self._keys))

    def __contains__(self, segment_id: object) -> bool:
        return segment_id in self._keys

    def __len__(self) -> int:
        return len(self._keys)

    def clear(self) -> None:
        """Delete all gradient files and reset the key set."""
        if self.root_dir.exists():
            shutil.rmtree(self.root_dir, ignore_errors=True)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._keys.clear()
