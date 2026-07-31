"""Runtime and memory profiling for segmented LLM training.

Phase 23 scope from the implementation plan:
- Track training/validation/test losses and task metrics.
- Track step/forward/backward/segment load/unload/recompute/optimizer/checkpoint/export timings.
- Count segment loads and saves.
- Estimate CPU memory and track GPU peak memory when CUDA is available.
- Export profiler results as a plain dictionary suitable for YAML/JSON protocol export.

This module is intentionally dependency-light and does not own training logic.
"""

from __future__ import annotations

import contextlib
import resource
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, MutableMapping, Optional


Number = int | float

# Maximum number of per-segment events kept in RAM at any time.  The peaks
# (per_segment_peak_bytes, segment_type_peak_bytes) are maintained separately
# and are never trimmed, so long-run statistics are preserved.
_MAX_TIMELINE_EVENTS: int = 2000


def _now() -> float:
    """Return a monotonic timestamp in seconds."""

    return time.perf_counter()


def _safe_float(value: Number) -> float:
    """Convert a numeric value to float for metric storage."""

    if not isinstance(value, (int, float)):
        raise TypeError(f"Metric value must be numeric, got {type(value).__name__}.")
    return float(value)


def get_cpu_peak_memory_bytes() -> int:
    """Return current process peak resident set size in bytes.

    On Linux ru_maxrss is KiB. On macOS it is bytes. The package is primarily
    developed/tested on Linux, but this function handles both conservatively.
    """

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes. Values below 10 MiB are almost
    # certainly KiB for a live Python process in our context.
    if usage < 10 * 1024 * 1024:
        return int(usage * 1024)
    return int(usage)


def get_gpu_peak_memory_bytes() -> Optional[int]:
    """Return CUDA peak memory allocated in bytes if CUDA is available."""

    try:
        import torch
    except Exception:
        return None

    try:
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated())
    except Exception:
        return None


def get_gpu_memory_snapshot() -> Optional[dict[str, int]]:
    """Return a full CUDA memory snapshot (bytes) if CUDA is available.

    Includes current and peak values for both allocated (tensors actually in
    use) and reserved (the larger caching-allocator pool) memory, since the
    two diverge significantly under segment load/unload churn.
    """

    try:
        import torch
    except Exception:
        return None

    try:
        if not torch.cuda.is_available():
            return None
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    except Exception:
        return None


def _get_cpu_current_memory_bytes() -> Optional[int]:
    """Return current process RSS in bytes using psutil, or None if unavailable."""
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _get_gpu_current_memory_bytes() -> Optional[int]:
    """Return current CUDA memory_allocated in bytes, or None if unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.memory_allocated())
    except Exception:
        return None


@dataclass(slots=True)
class TimerStats:
    """Accumulated timing statistics for one named profiler section."""

    total_seconds: float = 0.0
    count: int = 0
    last_seconds: float = 0.0

    def add(self, seconds: Number) -> None:
        seconds_f = _safe_float(seconds)
        if seconds_f < 0:
            raise ValueError(f"Timer duration must be non-negative, got {seconds_f}.")
        self.total_seconds += seconds_f
        self.last_seconds = seconds_f
        self.count += 1

    @property
    def average_seconds(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_seconds / self.count

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total_seconds": self.total_seconds,
            "count": self.count,
            "last_seconds": self.last_seconds,
            "average_seconds": self.average_seconds,
        }


@dataclass(slots=True)
class SegmentedTrainingProfiler:
    """Profiler for strict single-segment training runs.

    The profiler stores all values in plain Python containers so it can be saved
    into checkpoints or exported into the YAML protocol.
    """

    timers: MutableMapping[str, TimerStats] = field(default_factory=dict)
    counters: MutableMapping[str, int] = field(default_factory=dict)
    scalars: MutableMapping[str, list[float]] = field(default_factory=dict)
    metadata: MutableMapping[str, Any] = field(default_factory=dict)
    per_segment_peak_bytes: MutableMapping[str, int] = field(default_factory=dict)
    segment_type_peak_bytes: MutableMapping[str, int] = field(default_factory=dict)
    # Baseline RSS captured after setup, before any computation.
    # All cpu_net_* values are (current_rss - cpu_baseline_bytes).
    cpu_baseline_bytes: int = field(default=0)
    # Running peak of (current_rss - cpu_baseline_bytes); updated every snapshot.
    _cpu_net_peak_bytes: int = field(default=0)
    # Per-segment memory events: one dict per before_load/after_load/after_release event.
    # Keys: segment_key, segment_type, event, memory_bytes, absolute_memory_bytes, elapsed_s
    segment_memory_events: list = field(default_factory=list)
    # Per-batch (step-level) memory snapshots: one dict per training step.
    # Keys: step_id, phase, cpu_rss_bytes, gpu_allocated_bytes
    batch_memory_snapshots: list = field(default_factory=list)

    DEFAULT_TIMER_NAMES = (
        "step",
        "forward",
        "backward",
        "segment_load",
        "segment_save",
        "segment_unload",
        "segment_recomputation",
        "optimizer_update",
        "checkpoint",
        "full_model_export",
        "yaml_protocol_export",
        "validation",
        "test",
        "inference",
    )

    DEFAULT_COUNTER_NAMES = (
        "segment_loads",
        "segment_saves",
        "segment_unloads",
        "recompute_mismatches",
        "training_steps",
        "validation_runs",
        "test_runs",
        "checkpoint_saves",
        "full_model_exports",
        "yaml_protocol_exports",
    )

    DEFAULT_SCALAR_NAMES = (
        "training_loss",
        "validation_loss",
        "test_loss",
        "task_metric",
        "gpu_peak_memory_bytes",
        "cpu_peak_memory_bytes",
        "cpu_net_peak_memory_bytes",
        "disk_io_bytes",
        "mem_timeline_bytes",
        "mem_timeline_elapsed_s",
        "step_memory_bytes",
        "cpu_memory_bytes",
        "gpu_memory_bytes",
    )

    def __post_init__(self) -> None:
        for name in self.DEFAULT_TIMER_NAMES:
            self.timers.setdefault(name, TimerStats())
        for name in self.DEFAULT_COUNTER_NAMES:
            self.counters.setdefault(name, 0)
        for name in self.DEFAULT_SCALAR_NAMES:
            self.scalars.setdefault(name, [])

    def capture_cpu_baseline(self) -> None:
        """Record current RSS as the computation baseline.

        Call once after all framework/model setup is complete but before any
        training or inference computation begins.  All cpu_net_* readings are
        then (current_rss - baseline), isolating computation memory from
        Python/PyTorch/package overhead.
        """
        try:
            import psutil
            self.cpu_baseline_bytes = int(psutil.Process().memory_info().rss)
        except Exception:
            self.cpu_baseline_bytes = 0
        self._cpu_net_peak_bytes = 0

    def cpu_net_peak_memory_bytes(self) -> int:
        """Return the peak (current_rss - baseline) seen so far in bytes."""
        return self._cpu_net_peak_bytes

    def cpu_raw_peak_memory_bytes(self) -> int:
        """Return the raw (baseline-independent) peak process RSS seen so far.

        This is the all-time peak resident set size of the process — it includes
        the Python/PyTorch overhead AND any resident model/segment weights, with
        no baseline subtraction. Use this for ``peak_cpu_total_mb`` and as the
        minuend for a "from-start" net figure that fairly counts resident
        weights. Falls back to a fresh ``ru_maxrss`` reading if no memory
        snapshot has been recorded yet.
        """
        recorded = self.scalars.get("cpu_peak_memory_bytes", [])
        live = get_cpu_peak_memory_bytes()
        if recorded:
            return int(max(max(recorded), live))
        return int(live)

    def reset_cuda_peak_memory(self) -> None:
        """Reset CUDA peak memory stats when CUDA is available."""

        try:
            import torch
        except Exception:
            return

        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            return

    def add_time(self, name: str, seconds: Number) -> None:
        """Add elapsed seconds to a named timer."""

        if not name:
            raise ValueError("Timer name must not be empty.")
        self.timers.setdefault(name, TimerStats()).add(seconds)

    @contextlib.contextmanager
    def time_section(self, name: str) -> Iterator[None]:
        """Context manager that records elapsed seconds for a named section."""

        start = _now()
        try:
            yield
        finally:
            self.add_time(name, _now() - start)

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment a named counter."""

        if not name:
            raise ValueError("Counter name must not be empty.")
        if not isinstance(amount, int):
            raise TypeError(f"Counter amount must be int, got {type(amount).__name__}.")
        if amount < 0:
            raise ValueError(f"Counter amount must be non-negative, got {amount}.")
        self.counters[name] = self.counters.get(name, 0) + amount

    def record_scalar(self, name: str, value: Number) -> None:
        """Append a scalar metric value."""

        if not name:
            raise ValueError("Scalar name must not be empty.")
        self.scalars.setdefault(name, []).append(_safe_float(value))

    def record_training_loss(self, loss: Number) -> None:
        self.record_scalar("training_loss", loss)

    def record_validation_loss(self, loss: Number) -> None:
        self.record_scalar("validation_loss", loss)

    def record_test_loss(self, loss: Number) -> None:
        self.record_scalar("test_loss", loss)

    def record_task_metric(self, value: Number) -> None:
        self.record_scalar("task_metric", value)

    def record_segment_load(self, seconds: Number | None = None) -> None:
        self.increment("segment_loads")
        if seconds is not None:
            self.add_time("segment_load", seconds)

    def record_segment_unload(self, seconds: Number | None = None) -> None:
        self.increment("segment_unloads")
        if seconds is not None:
            self.add_time("segment_unload", seconds)

    def record_segment_save(self, seconds: Number | None = None) -> None:
        self.increment("segment_saves")
        if seconds is not None:
            self.add_time("segment_save", seconds)

    def record_recompute_mismatch(self, count: int = 1) -> None:
        self.increment("recompute_mismatches", count)

    def record_segment_memory_event(
        self,
        segment_key: str,
        segment_type: str,
        memory_bytes: int,
        event: str,
        elapsed_s: float,
        absolute_memory_bytes: int | None = None,
    ) -> None:
        """Record memory usage at a segment load or release event.

        Args:
            segment_key: Unique key for the segment (e.g. ``segment_id.to_key()``).
            segment_type: Type of the segment (e.g. ``"attention"``, ``"mlp"``).
            memory_bytes: **Footprint delta** — bytes added by loading this segment.
                On CPU this is (RSS after load) − (RSS before load), so it reflects
                only the segment's own parameters rather than the entire process.
                On GPU this is memory_allocated after load (already near-zero baseline).
            event: Either ``"load"`` or ``"release"``.
            elapsed_s: Seconds elapsed since the loader was created (for timeline X axis).
            absolute_memory_bytes: Absolute device-memory reading after the event.
                Used for the timeline so the shape of total memory over time is visible.
                Defaults to ``memory_bytes`` when not provided.
        """
        timeline_bytes = absolute_memory_bytes if absolute_memory_bytes is not None else memory_bytes

        if event == "load":
            # Peaks track the footprint (delta), not the absolute process size.
            existing_seg = self.per_segment_peak_bytes.get(segment_key, 0)
            if memory_bytes > existing_seg:
                self.per_segment_peak_bytes[segment_key] = memory_bytes
            existing_type = self.segment_type_peak_bytes.get(segment_type, 0)
            if memory_bytes > existing_type:
                self.segment_type_peak_bytes[segment_type] = memory_bytes

        # Timeline uses the absolute reading so total memory shape is visible.
        self.record_scalar("mem_timeline_bytes", timeline_bytes)
        self.record_scalar("mem_timeline_elapsed_s", elapsed_s)

        # Detailed per-segment event list (all three event types).
        self.segment_memory_events.append({
            "segment_key": segment_key,
            "segment_type": segment_type,
            "event": event,
            "memory_bytes": memory_bytes,
            "absolute_memory_bytes": timeline_bytes,
            "elapsed_s": elapsed_s,
        })

        # Rolling window: cap in-RAM list so it does not grow without bound
        # across a long training run.  Peaks are tracked separately and never
        # trimmed, so long-run statistics are still preserved.
        if len(self.segment_memory_events) > _MAX_TIMELINE_EVENTS:
            self.segment_memory_events = self.segment_memory_events[-_MAX_TIMELINE_EVENTS:]
            for _key in ("mem_timeline_bytes", "mem_timeline_elapsed_s"):
                _lst = self.scalars.get(_key)
                if _lst and len(_lst) > _MAX_TIMELINE_EVENTS:
                    self.scalars[_key] = _lst[-_MAX_TIMELINE_EVENTS:]

    def record_memory_snapshot(self) -> None:
        """Record CPU and GPU peak and current memory snapshots."""

        self.record_scalar("cpu_peak_memory_bytes", get_cpu_peak_memory_bytes())

        # Net CPU peak: (current_rss - baseline), computed only when psutil is
        # available and a baseline has been captured.
        cpu_current = _get_cpu_current_memory_bytes()
        if cpu_current is not None:
            self.record_scalar("cpu_memory_bytes", cpu_current)
            if self.cpu_baseline_bytes > 0:
                net = max(0, cpu_current - self.cpu_baseline_bytes)
                if net > self._cpu_net_peak_bytes:
                    self._cpu_net_peak_bytes = net
                self.record_scalar("cpu_net_peak_memory_bytes", float(self._cpu_net_peak_bytes))

        gpu_snapshot = get_gpu_memory_snapshot()
        if gpu_snapshot is not None:
            self.record_scalar("gpu_peak_memory_bytes", gpu_snapshot["peak_allocated_bytes"])
            self.record_scalar("gpu_allocated_bytes", gpu_snapshot["allocated_bytes"])
            self.record_scalar("gpu_reserved_bytes", gpu_snapshot["reserved_bytes"])
            self.record_scalar("gpu_peak_reserved_bytes", gpu_snapshot["peak_reserved_bytes"])

        gpu_current = _get_gpu_current_memory_bytes()
        if gpu_current is not None:
            self.record_scalar("gpu_memory_bytes", gpu_current)

    def record_batch_memory_snapshot(
        self,
        step_id: int,
        phase: str = "training",
        gpu_peak_bytes: Optional[int] = None,
    ) -> None:
        """Record a per-batch (step-level) memory snapshot.

        Captures current RSS (CPU) and current allocated VRAM (GPU) at the end of
        each training step so memory can be plotted per batch alongside the
        per-segment timeline.

        Args:
            gpu_peak_bytes: Peak VRAM allocated during this step
                (torch.cuda.max_memory_allocated after a per-step reset).
                None on CPU or when not tracked.
        """
        cpu_current = _get_cpu_current_memory_bytes()
        gpu_current = _get_gpu_current_memory_bytes()
        self.batch_memory_snapshots.append({
            "step_id": step_id,
            "phase": phase,
            "cpu_rss_bytes": cpu_current,
            "gpu_allocated_bytes": gpu_current,
            "gpu_peak_bytes": gpu_peak_bytes,
        })

    def set_metadata(self, key: str, value: Any) -> None:
        if not key:
            raise ValueError("Metadata key must not be empty.")
        self.metadata[key] = value

    def latest_scalar(self, name: str) -> Optional[float]:
        values = self.scalars.get(name, [])
        if not values:
            return None
        return values[-1]

    def mean_scalar(self, name: str) -> Optional[float]:
        values = self.scalars.get(name, [])
        if not values:
            return None
        return sum(values) / len(values)

    def summary(self) -> dict[str, Any]:
        """Return a compact summary of the latest and average scalar values."""

        scalar_summary: dict[str, dict[str, float | int | None]] = {}
        for name, values in self.scalars.items():
            scalar_summary[name] = {
                "count": len(values),
                "latest": values[-1] if values else None,
                "mean": (sum(values) / len(values)) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }

        return {
            "timers": {name: stats.to_dict() for name, stats in self.timers.items()},
            "counters": dict(self.counters),
            "scalars": scalar_summary,
            "metadata": dict(self.metadata),
            "per_segment_peak_bytes": dict(self.per_segment_peak_bytes),
            "segment_type_peak_bytes": dict(self.segment_type_peak_bytes),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return full profiler state as a plain dictionary."""

        return {
            "timers": {name: stats.to_dict() for name, stats in self.timers.items()},
            "counters": dict(self.counters),
            "scalars": {name: list(values) for name, values in self.scalars.items()},
            "metadata": dict(self.metadata),
            "per_segment_peak_bytes": dict(self.per_segment_peak_bytes),
            "segment_type_peak_bytes": dict(self.segment_type_peak_bytes),
            "segment_memory_events": list(self.segment_memory_events),
            "batch_memory_snapshots": list(self.batch_memory_snapshots),
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SegmentedTrainingProfiler":
        """Reconstruct a profiler from a dictionary produced by :meth:`to_dict`."""

        profiler = cls()
        profiler.timers.clear()
        for name, stats in data.get("timers", {}).items():
            timer = TimerStats(
                total_seconds=float(stats.get("total_seconds", 0.0)),
                count=int(stats.get("count", 0)),
                last_seconds=float(stats.get("last_seconds", 0.0)),
            )
            profiler.timers[name] = timer

        profiler.counters.clear()
        profiler.counters.update(
            {str(name): int(value) for name, value in data.get("counters", {}).items()}
        )

        profiler.scalars.clear()
        profiler.scalars.update(
            {
                str(name): [float(v) for v in values]
                for name, values in data.get("scalars", {}).items()
            }
        )

        profiler.metadata.clear()
        profiler.metadata.update(dict(data.get("metadata", {})))

        profiler.per_segment_peak_bytes.clear()
        profiler.per_segment_peak_bytes.update(
            {str(k): int(v) for k, v in data.get("per_segment_peak_bytes", {}).items()}
        )

        profiler.segment_type_peak_bytes.clear()
        profiler.segment_type_peak_bytes.update(
            {str(k): int(v) for k, v in data.get("segment_type_peak_bytes", {}).items()}
        )

        profiler.cpu_baseline_bytes = int(data.get("cpu_baseline_bytes", 0))
        profiler._cpu_net_peak_bytes = int(data.get("_cpu_net_peak_bytes", 0))
        profiler.segment_memory_events = list(data.get("segment_memory_events", []))
        profiler.batch_memory_snapshots = list(data.get("batch_memory_snapshots", []))

        return profiler
