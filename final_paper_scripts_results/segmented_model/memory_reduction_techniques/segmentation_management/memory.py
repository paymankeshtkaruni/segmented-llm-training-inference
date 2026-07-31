"""
Memory helpers — freeing and measuring, with the CPU/GPU distinctions that matter.

`free_device` is called between segments to actually return memory:
  * gc.collect()            — UNCONDITIONAL. Reference cycles (autograd closures,
                              context managers) accumulate on CPU and GPU alike, and
                              they pin tensors until collected.
  * torch.cuda.empty_cache  — CUDA ONLY. Returns torch's cached VRAM to the driver so
                              nvidia-smi / the next segment sees it freed.
  * malloc_trim(0)          — LINUX/CPU ONLY. glibc keeps freed heap pages; trim returns
                              them to the kernel so process RSS (the CPU constraint)
                              actually drops. No-op meaning on the GPU path.

The samplers (host RSS, process VRAM via nvidia-smi, torch VRAM counters) are used by
the cost/measurement runs. They are torch-free except `torch_vram` (guarded).
"""

from __future__ import annotations

import ctypes
import gc
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


def free_device(device: str) -> None:
    """Return memory to the OS/driver between segments (see module docstring)."""
    gc.collect()
    dev = str(device)
    if dev.startswith("cuda"):
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# point measurements
# --------------------------------------------------------------------------- #
def host_rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 1024**2


def smi_process_vram_mb() -> float:
    """VRAM (MB) used by THIS process per nvidia-smi (0.0 if unavailable)."""
    pid = os.getpid()
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL).decode()
    except Exception:
        return 0.0
    tot = 0.0
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == pid:
            try:
                tot += float(parts[1])
            except ValueError:
                pass
    return tot


def torch_vram_mb(device) -> Dict[str, float]:
    import torch
    if device is None or torch.device(device).type != "cuda":
        return {"alloc": 0.0, "peak_alloc": 0.0, "reserved": 0.0}
    return {"alloc": torch.cuda.memory_allocated(device) / 1024**2,
            "peak_alloc": torch.cuda.max_memory_allocated(device) / 1024**2,
            "reserved": torch.cuda.memory_reserved(device) / 1024**2}


# --------------------------------------------------------------------------- #
# background samplers (continuous timelines)
# --------------------------------------------------------------------------- #
@dataclass
class Sampler:
    fn: Callable[[], float]
    interval_s: float = 0.02
    peak: float = 0.0
    timeline: List[Tuple[float, float]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None
    _t0: float = 0.0

    def _loop(self):
        while not self._stop.is_set():
            v = self.fn()
            self.peak = max(self.peak, v)
            self.timeline.append((time.perf_counter() - self._t0, v))
            self._stop.wait(self.interval_s)

    def start(self):
        self._t0 = time.perf_counter()
        self.peak = self.fn()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> float:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self.peak


if __name__ == "__main__":
    print("host_rss_mb:", round(host_rss_mb(), 1))
    print("smi_process_vram_mb:", smi_process_vram_mb())
    free_device("cpu")
    print("free_device('cpu') ok (gc + malloc_trim)")
