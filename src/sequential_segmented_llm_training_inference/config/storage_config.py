"""Segment storage and compute device configuration."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal

DeviceName = Literal["cpu", "cuda"]
SegmentStorageBackend = Literal["disk_streaming", "cpu_ram_offload"]


@dataclass(frozen=True)
class StorageConfig:
    """Defines where inactive segments live and where active segments execute."""

    device: DeviceName = "cpu"
    backend: SegmentStorageBackend = "disk_streaming"
    segment_dir: str = "segments"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be either 'cpu' or 'cuda'.")
        if self.backend not in {"disk_streaming", "cpu_ram_offload"}:
            raise ValueError("backend must be 'disk_streaming' or 'cpu_ram_offload'.")
        if self.device == "cpu" and self.backend != "disk_streaming":
            raise ValueError("CPU execution supports only disk_streaming segment storage.")
        if not self.segment_dir:
            raise ValueError("segment_dir must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, device: str | None = None) -> "StorageConfig":
        merged = dict(data)
        if device is not None:
            merged["device"] = device
        return cls(**merged)
