"""
C — torch-free ONNX inference-cost (LARGE model).

This module imports ONLY onnxruntime + numpy + psutil + subprocess + stdlib.
It deliberately does NOT import torch or `_common` (which pulls torch). A guard at
the end of every run asserts torch never entered the process, so the measured cost
is genuinely torch-free. The one-time ONNX export (export_onnx.py) is a SEPARATE
job whose torch usage is not part of this measurement.

Loads `full_model/onnx/large_model.onnx` and runs N forward passes on full-length
synthetic batches (batch 4, seq 512) via onnxruntime, measuring:
  CPU: host RAM (RSS) — framework baseline (python+onnxruntime) vs model+data.
  GPU: + process VRAM (nvidia-smi). onnxruntime creates its CUDA context + loads
       weights at session creation; we record before/after-session and peak.
Outputs <prefix>_metrics.json + <prefix>_trace.csv (with memory timelines).
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import onnxruntime as ort

FULL_MODEL = Path(__file__).resolve().parent.parent
ONNX_PATH = FULL_MODEL / "onnx" / "large_model.onnx"
OUTPUTS = FULL_MODEL / "outputs"

# Cost params — fixed, matching the A2/B2 large-model cost runs.
BATCH = 4
SEQ_LEN = 512
N_STEPS = 3
VOCAB = 50257


def _rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 1024**2


def _smi_vram_mb() -> float:
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


@dataclass
class _Sampler:
    fn: Callable[[], float]
    interval_s: float = 0.02
    peak: float = 0.0
    timeline: List[Tuple[float, float]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: object = None
    _t0: float = 0.0

    def _loop(self) -> None:
        while not self._stop.is_set():
            v = self.fn()
            self.peak = max(self.peak, v)
            self.timeline.append((time.perf_counter() - self._t0, v))
            self._stop.wait(self.interval_s)

    def start(self) -> "_Sampler":
        self._t0 = time.perf_counter()
        self.peak = self.fn()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.peak


def _synth(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, VOCAB, size=(BATCH, SEQ_LEN), dtype=np.int64)


def run(device: str, profile_memory: bool, out_dir: Path, prefix: str) -> None:
    is_cuda = device == "cuda"
    provider = "CUDAExecutionProvider" if is_cuda else "CPUExecutionProvider"
    if not ONNX_PATH.exists():
        raise FileNotFoundError(f"{ONNX_PATH} not found — run export_onnx first")

    if is_cuda:
        # Load CUDA/cuDNN/cuBLAS .so from the nvidia-* pip packages so onnxruntime's
        # CUDA provider can link them. This loads SHARED LIBRARIES only — it does
        # NOT import torch, so the run stays torch-free.
        ort.preload_dlls()
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider not available in onnxruntime")

    print(f"[1] C onnx inference-cost (torch-free) on {device}  provider={provider}  "
          f"bs={BATCH} seq={SEQ_LEN} n_steps={N_STEPS} profile_memory={profile_memory}")

    # baseline: python + onnxruntime (+CUDA libs if GPU) loaded; NO torch
    host_baseline = _rss_mb()
    cpu_s = gpu_s = None
    if profile_memory:
        cpu_s = _Sampler(_rss_mb).start()
        if is_cuda:
            gpu_s = _Sampler(_smi_vram_mb).start()
    vram_before = _smi_vram_mb() if is_cuda else 0.0

    # session creation: onnxruntime makes its CUDA context + loads the weights
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "8"))  # avoid affinity spam
    sess = ort.InferenceSession(str(ONNX_PATH), sess_options=so, providers=[provider])
    active = sess.get_providers()
    print(f"    active providers: {active}")
    if is_cuda and "CUDAExecutionProvider" not in active:
        raise RuntimeError(f"CUDA EP did not activate (silent CPU fallback); active={active}")
    host_after_session = _rss_mb()
    vram_after_session = _smi_vram_mb() if is_cuda else 0.0

    sess.run(["logits"], {"input_ids": _synth(0)})  # warmup
    step_times: List[float] = []
    trace: List[dict] = []
    for step in range(N_STEPS):
        x = _synth(1 + step)
        t0 = time.perf_counter()
        sess.run(["logits"], {"input_ids": x})
        dt = time.perf_counter() - t0
        step_times.append(dt)
        row = {"step": step, "step_time_s": round(dt, 6)}
        if profile_memory:
            row["host_rss_mb"] = round(_rss_mb(), 1)
            if is_cuda:
                row["nvidia_smi_vram_mb"] = round(_smi_vram_mb(), 1)
        trace.append(row)

    memory = {}
    if profile_memory:
        host_peak = cpu_s.stop()
        memory["baseline"] = {
            "_comment": "framework = python + onnxruntime, NO torch; "
                        "subtract from peak for model+data",
            "host_framework_mb": round(host_baseline, 1),
            "host_after_session_mb": round(host_after_session, 1),
        }
        memory["host_cpu_ram"] = {
            "peak_mb": round(host_peak, 1),
            "framework_baseline_mb": round(host_baseline, 1),
            "net_model_data_mb": round(host_peak - host_baseline, 1),
            "timeline": [[round(t, 4), round(v, 1)] for t, v in cpu_s.timeline],
        }
        if is_cuda:
            vram_peak = gpu_s.stop()
            memory["baseline"]["vram_before_session_mb"] = round(vram_before, 1)
            memory["baseline"]["vram_after_session_mb"] = round(vram_after_session, 1)
            memory["vram"] = {
                "nvidia_smi_peak_mb": round(vram_peak, 1),
                "before_session_mb": round(vram_before, 1),
                "after_session_mb": round(vram_after_session, 1),  # context + weights
                "net_model_data_mb": round(vram_peak - vram_before, 1),
                "activations_mb": round(vram_peak - vram_after_session, 1),
                "nvidia_smi_timeline": [[round(t, 4), round(v, 1)] for t, v in gpu_s.timeline],
            }

    measured = step_times[1:] if len(step_times) > 1 else step_times
    avg = sum(measured) / len(measured) if measured else float("nan")
    metrics = {
        "run": "C_onnx_inference_cost", "mode": "infer", "runtime": "onnxruntime",
        "torch_free": True, "device": device, "provider": provider, "model": "large",
        "batch_size": BATCH, "seq_len": SEQ_LEN, "n_steps": N_STEPS,
        "profile_memory": profile_memory,
        "step_times_s": [round(t, 6) for t in step_times],
        "avg_step_time_s": avg,
        "throughput_samples_per_s": BATCH / avg if avg and avg > 0 else float("nan"),
        "memory": memory,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(out_dir / f"{prefix}_metrics.json", "w"), indent=2)
    if trace:
        with open(out_dir / f"{prefix}_trace.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
            w.writeheader(); w.writerows(trace)

    print(f"[done] avg_step={avg*1000:.1f}ms")
    if profile_memory:
        h = memory["host_cpu_ram"]
        print(f"       host RAM: peak={h['peak_mb']}MB = framework {h['framework_baseline_mb']}MB "
              f"+ model+data {h['net_model_data_mb']}MB")
        if is_cuda:
            v = memory["vram"]
            print(f"       VRAM: peak={v['nvidia_smi_peak_mb']}MB "
                  f"(after-session {v['after_session_mb']}MB = context+weights; "
                  f"activations {v['activations_mb']}MB)")
    print(f"       -> {out_dir/(prefix + '_metrics.json')}")

    # PROVE torch-free: torch must never have been imported in this process.
    import sys
    assert "torch" not in sys.modules, "torch leaked into the torch-free C run!"
    print("       [torch-free OK] 'torch' not in sys.modules")
