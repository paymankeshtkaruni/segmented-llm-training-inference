"""
Stores — where the *inactive* slices live (the offload targets).

Per the device→target rule (DESIGN §4): on GPU the constraint is VRAM so inactive
state is parked in **host RAM** (`cpu_ram`); on CPU the constraint is host RAM so it
is parked on **disk**. A store holds any per-segment payload — segment weights,
accumulated gradients, or optimizer state (m,v) — as a CPU/disk tensor dict, keyed
by a `SegmentKey`. Only the *active* segment is ever materialized on the compute
device (loader.py); everything else sits in a store.

Design choices:
- One uniform `SegmentStore` interface with two backends so the rest of the code is
  device-agnostic; `make_store(kind)` picks `cpu_ram` (GPU runs) or `disk` (CPU runs).
- Payloads are always stored on **CPU** (`cpu_ram`) or **disk**, never on the GPU —
  that is the whole point (keep VRAM free).
- Disk backend: one `.pt` file per key; `torch.load(map_location="cpu")` so loading
  never touches the GPU. Read-modify-write `accumulate()` for gradient accumulation
  without holding more than one segment's grads at once.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

StateDict = Dict[str, torch.Tensor]


@dataclass(frozen=True)
class SegmentKey:
    """Identifies one segment payload. layer_id = -1 for global (embedding/head)."""
    layer_id: int
    kind: str          # 'attention' | 'mlp' | 'embedding' | 'output_head'
    seg: int

    def fname(self) -> str:
        return f"L{self.layer_id}__{self.kind}__s{self.seg}.pt"


def _to_cpu(sd: StateDict) -> StateDict:
    return {k: v.detach().to("cpu", copy=True) for k, v in sd.items()}


class SegmentStore:
    """Abstract: put/get/accumulate/evict per-segment CPU tensor dicts."""
    def put(self, key: SegmentKey, sd: StateDict) -> None: ...
    def get(self, key: SegmentKey) -> Optional[StateDict]: ...
    def has(self, key: SegmentKey) -> bool: ...
    def accumulate(self, key: SegmentKey, sd: StateDict) -> None: ...
    def evict(self, key: SegmentKey) -> None: ...
    def keys(self) -> Iterable[SegmentKey]: ...


class CpuRamStore(SegmentStore):
    """GPU runs: park payloads in host RAM (free on GPU; keeps VRAM at one slice)."""
    def __init__(self) -> None:
        self._d: Dict[SegmentKey, StateDict] = {}

    def put(self, key, sd):
        self._d[key] = _to_cpu(sd)

    def get(self, key):
        sd = self._d.get(key)
        return None if sd is None else {k: v.clone() for k, v in sd.items()}

    def has(self, key):
        return key in self._d

    def accumulate(self, key, sd):
        if key not in self._d:
            self._d[key] = _to_cpu(sd)
        else:
            cur = self._d[key]
            for k, v in sd.items():
                cur[k] = cur[k] + v.detach().to("cpu")

    def evict(self, key):
        self._d.pop(key, None)

    def keys(self):
        return list(self._d.keys())


class DiskStore(SegmentStore):
    """CPU runs: park payloads on disk (host RAM is the constraint). One .pt per key,
    loaded with map_location='cpu'. accumulate() is read-modify-write so at most one
    segment's payload is in RAM at a time."""
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._keys: set[SegmentKey] = set()

    def _path(self, key):
        return self.root / key.fname()

    def put(self, key, sd):
        torch.save(_to_cpu(sd), self._path(key))
        self._keys.add(key)

    def get(self, key):
        p = self._path(key)
        return torch.load(p, map_location="cpu") if p.exists() else None

    def has(self, key):
        return key in self._keys

    def accumulate(self, key, sd):
        if not self.has(key):
            torch.save(_to_cpu(sd), self._path(key)); self._keys.add(key)
        else:
            cur = torch.load(self._path(key), map_location="cpu")
            for k, v in sd.items():
                cur[k] = cur[k] + v.detach().to("cpu")
            torch.save(cur, self._path(key))
            del cur

    def evict(self, key):
        p = self._path(key)
        if p.exists():
            p.unlink()
        self._keys.discard(key)

    def keys(self):
        return list(self._keys)


def make_store(kind: str, root: Optional[Path] = None) -> SegmentStore:
    """kind: 'cpu_ram' (GPU runs) or 'disk' (CPU runs)."""
    if kind == "cpu_ram":
        return CpuRamStore()
    if kind == "disk":
        if root is None:
            raise ValueError("disk store needs a root path")
        return DiskStore(root)
    raise ValueError(f"unknown store kind {kind!r}")


def default_store_kind(device: str) -> str:
    """The device→target rule: GPU→cpu_ram, CPU→disk."""
    return "cpu_ram" if str(device).startswith("cuda") else "disk"


if __name__ == "__main__":
    import tempfile
    torch.manual_seed(0)
    k = SegmentKey(0, "mlp", 1)
    sd = {"w": torch.randn(4, 4), "b": torch.randn(4)}
    for kind, root in [("cpu_ram", None), ("disk", Path(tempfile.mkdtemp()) / "store")]:
        s = make_store(kind, root)
        s.put(k, sd)
        got = s.get(k)
        ok_rt = all(torch.allclose(got[n], sd[n]) for n in sd)
        s.accumulate(k, sd)                       # now should be 2*sd
        acc = s.get(k)
        ok_acc = all(torch.allclose(acc[n], 2 * sd[n]) for n in sd)
        s.evict(k)
        ok_ev = not s.has(k)
        print(f"{kind:8} round-trip={ok_rt} accumulate=2x:{ok_acc} evict={ok_ev}")
    print("default_store_kind: cuda->%s cpu->%s" %
          (default_store_kind("cuda"), default_store_kind("cpu")))
