#!/usr/bin/env python
"""Measured bandwidth of the SEGMENT-STORE PATH, exactly as the engine uses it.

Sec. IX attributes the streamed training step at 0.84B to "transport under 10%,
orchestration the rest". That split was computed against an *assumed* ~20 GB/s
PCIe figure, which a referee can rightly object to: the assumed number is a link
peak, while the engine's store copies go through **pageable** host memory (the
store hands out `torch.Tensor.clone()`s on the host, and `module.to(device)`
copies them one parameter at a time), so the achieved rate is what matters, not
the link rating. This script measures the achieved rate.

Three paths are timed, all through the real objects (`StrictSegmentLoader`,
`CpuRamStore`/`DiskStore`), never a synthetic copy:

  FETCH  `loader.load_segment(key)` — store.get(key) host clone (or torch.load
         from disk) + module.load_state_dict + module.to(device). This is the
         per-segment arrival cost the step pays.
  PARK   `loader.release_segment(save=True)` — device -> host store write-back,
         including the store's own copy. This is the departure cost.
  SPLIT  (cuda only) the same fetch, decomposed: store.get(key) alone vs the
         .to(device) of the resulting tensors, so the report can say how much of
         the fetch is host-side copying and how much is host-to-device transfer.

Plus hardware context on cuda: a plain 256 MiB pageable H2D copy and the same
copy from pinned memory, which bracket what the link could give.

Bound to the streaming technique code 11111111100 — the same code the streamed
training modes (T3/T6/T9/T12) run under — so the measured path is the measured
mode's path.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import socket
import statistics
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "segmentation_management"))

from techniques import Tech                                              # noqa: E402
from config import get_preset                                           # noqa: E402
from modules import ReferenceGPTDecoder                                 # noqa: E402
from forward_engine import populate_from_reference                      # noqa: E402
from loader import StrictSegmentLoader                                  # noqa: E402
from stores import make_store                                           # noqa: E402
from optimizer import all_segment_keys                                  # noqa: E402

# the streamed training modes' technique code (stream on, free_device off) — the
# configuration whose step time Sec. IX decomposes.
STREAM_TECH_CODE = "11111111100"
GB = 1e9            # decimal GB, so GB/s is comparable with vendor link ratings
MIB = 1024.0 ** 2
RAW_COPY_BYTES = 256 * 1024 * 1024      # 256 MiB fp32 hardware-context copy
RAW_COPY_REPS = 5


def tech_from_code(code: str) -> Tech:
    names = [f.name for f in fields(Tech)]
    if len(code) != len(names):
        raise ValueError(f"tech code length {len(code)} != {len(names)} flags")
    return Tech(**{n: c == "1" for n, c in zip(names, code)})


def _sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _module_bytes(module) -> int:
    """Payload actually moved for one segment: parameters + buffers."""
    return (sum(p.numel() * p.element_size() for p in module.parameters())
            + sum(b.numel() * b.element_size() for b in module.buffers()))


def _sd_bytes(sd) -> int:
    return sum(v.numel() * v.element_size() for v in sd.values())


def _rates(total_bytes: int, seconds: float) -> dict:
    return {"bytes": int(total_bytes), "seconds": seconds,
            "gbs": (total_bytes / GB) / seconds if seconds > 0 else float("nan"),
            "mibs": (total_bytes / MIB) / seconds if seconds > 0 else float("nan")}


def _median_pass(passes: list) -> dict:
    """The median pass BY RATE, reported whole — not a per-field median, so the
    seconds and the GB/s quoted in the paper come from the same pass."""
    ordered = sorted(passes, key=lambda p: p["gbs"])
    med = ordered[(len(ordered) - 1) // 2]
    return {"gbs": med["gbs"], "mibs": med["mibs"], "seconds": med["seconds"],
            "bytes": med["bytes"], "pass_index": passes.index(med)}


def fetch_pass(loader, keys, device: str) -> dict:
    """Time load_segment(key) for every key in engine order; no write-back."""
    total_bytes, total_s, per_key_max = 0, 0.0, 0
    for key in keys:
        _sync(device)
        t0 = time.perf_counter()
        module = loader.load_segment(key)
        _sync(device)
        total_s += time.perf_counter() - t0
        nb = _module_bytes(module)
        total_bytes += nb
        per_key_max = max(per_key_max, nb)
        loader.release_segment(save=False)
    out = _rates(total_bytes, total_s)
    out["largest_segment_bytes"] = per_key_max
    return out


def park_pass(loader, keys, device: str) -> dict:
    """Time release_segment(save=True) — the device -> store write-back."""
    total_bytes, total_s = 0, 0.0
    for key in keys:
        module = loader.load_segment(key)          # untimed: we time the departure
        nb = _module_bytes(module)
        _sync(device)
        t0 = time.perf_counter()
        loader.release_segment(save=True)
        _sync(device)
        total_s += time.perf_counter() - t0
        total_bytes += nb
    return _rates(total_bytes, total_s)


def split_pass(store, keys, device: str) -> dict:
    """Decompose the fetch on cuda: store.get() host clone vs .to(device).

    The loader does both back to back; timing them apart is what answers "is the
    store path bounded by the link, or by host-side copying?".
    """
    total_bytes, host_s, h2d_s = 0, 0.0, 0.0
    for key in keys:
        t0 = time.perf_counter()
        sd = store.get(key)                        # host clone / torch.load
        host_s += time.perf_counter() - t0
        nb = _sd_bytes(sd)
        _sync(device)
        t0 = time.perf_counter()
        sd_dev = {k: v.to(device) for k, v in sd.items()}
        _sync(device)
        h2d_s += time.perf_counter() - t0
        total_bytes += nb
        del sd, sd_dev
    return {"bytes": int(total_bytes),
            "host_copy": _rates(total_bytes, host_s),
            "host_to_device": _rates(total_bytes, h2d_s),
            "sum_seconds": host_s + h2d_s}


def raw_h2d(device: str) -> dict:
    """Hardware context: 256 MiB fp32 pageable vs pinned host -> device."""
    n = RAW_COPY_BYTES // 4
    src = torch.empty(n, dtype=torch.float32)
    pageable, pinned = [], []
    for _ in range(RAW_COPY_REPS):
        _sync(device)
        t0 = time.perf_counter()
        dst = src.to(device)
        _sync(device)
        pageable.append((RAW_COPY_BYTES / GB) / (time.perf_counter() - t0))
        del dst
    src_pin = src.pin_memory()
    for _ in range(RAW_COPY_REPS):
        _sync(device)
        t0 = time.perf_counter()
        dst = src_pin.to(device)
        _sync(device)
        pinned.append((RAW_COPY_BYTES / GB) / (time.perf_counter() - t0))
        del dst
    del src, src_pin
    return {"bytes": RAW_COPY_BYTES, "reps": RAW_COPY_REPS,
            "raw_pageable_h2d_gbs": statistics.median(pageable),
            "raw_pinned_h2d_gbs": statistics.median(pinned)}


def fs_info(path: Path) -> dict:
    """Filesystem the disk store sits on (the CPU cells' parking target)."""
    st = os.statvfs(path)
    info = {"path": str(path), "f_bsize": st.f_bsize, "f_frsize": st.f_frsize,
            "free_bytes": st.f_bavail * st.f_frsize, "fstype": None, "mount_point": None}
    try:                                    # longest mount-point prefix wins
        best = ""
        with open("/proc/mounts") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 3 and str(path).startswith(parts[1]) and len(parts[1]) > len(best):
                    best, info["fstype"], info["mount_point"] = parts[1], parts[2], parts[1]
    except OSError:
        pass
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cuda", "cpu"], required=True)
    ap.add_argument("--store", choices=["cpu_ram", "disk"], required=True)
    ap.add_argument("--preset", default="large_8x2x2x8")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, default=Path("results/measurement_checks"))
    a = ap.parse_args()

    out_dir = Path(a.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # disk store root sits under out-dir, i.e. on the same filesystem the CPU cells
    # park to (measure_one.py's _work convention); deleted at the end like that scratch.
    store_root = out_dir / f"_store_bw_{a.preset}"

    preset = get_preset(a.preset)
    m = replace(preset["model"], dropout=0.0)      # dropout is irrelevant to transport
    s = preset["seg"]
    tech = tech_from_code(STREAM_TECH_CODE)

    torch.manual_seed(0)
    ref = ReferenceGPTDecoder(m)
    ref_param_bytes = sum(p.numel() * p.element_size() for p in ref.parameters())
    if a.store == "disk" and store_root.exists():
        shutil.rmtree(store_root)
    store = make_store("cpu_ram") if a.store == "cpu_ram" else make_store("disk", store_root)
    populate_from_reference(ref, m, s, store)
    del ref                                        # the reference is never needed again
    gc.collect()

    loader = StrictSegmentLoader(m, s, store, a.device, tech=tech)
    keys = all_segment_keys(m, s)

    result = {
        "script": "store_bandwidth.py", "preset": a.preset, "device": a.device,
        "store": a.store, "tech_code": STREAM_TECH_CODE, "passes": a.passes,
        "hostname": socket.gethostname(), "torch": torch.__version__,
        "n_segments": len(keys), "reference_param_bytes": ref_param_bytes,
        "device_name": (torch.cuda.get_device_name(0)
                        if a.device == "cuda" and torch.cuda.is_available() else "cpu"),
    }
    try:
        fetches = [fetch_pass(loader, keys, a.device) for _ in range(a.passes)]
        parks = [park_pass(loader, keys, a.device) for _ in range(a.passes)]
        result["fetch_passes"] = fetches
        result["fetch_median"] = _median_pass(fetches)
        result["park_passes"] = parks
        result["park_median"] = _median_pass(parks)
        result["segment_bytes_total"] = fetches[0]["bytes"]
        result["largest_segment_bytes"] = fetches[0]["largest_segment_bytes"]
        if a.device == "cuda":
            result["fetch_split"] = split_pass(store, keys, a.device)
            result["raw_h2d"] = raw_h2d(a.device)
            result["max_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        if a.store == "disk":
            result["store_fs"] = fs_info(store_root)
        result["note"] = (
            "loader fetch = store.get(key) host clone (cpu_ram) or torch.load(map_location=cpu) "
            "(disk) + load_state_dict + module.to(device); park = release_segment(save=True), "
            "i.e. state_dict() off the device plus the store's own host copy. Host memory is "
            "PAGEABLE throughout (no pinning anywhere in the engine), which is why the fetch "
            "rate sits below the raw pinned H2D figure. GB = 1e9 bytes. Backs the "
            "transport-vs-orchestration attribution of Sec. IX.")
    finally:
        if a.store == "disk" and store_root.exists():
            shutil.rmtree(store_root, ignore_errors=True)

    out = out_dir / f"store_bandwidth_{a.preset}_{a.device}_{a.store}.json"
    json.dump(result, open(out, "w"), indent=2)
    fm, pm = result["fetch_median"], result["park_median"]
    print(f"[store_bandwidth] {a.preset} {a.device}/{a.store}: {result['n_segments']} segments, "
          f"{result['segment_bytes_total']/GB:.3f} GB/pass | FETCH {fm['gbs']:.3f} GB/s "
          f"({fm['mibs']:.0f} MiB/s, {fm['seconds']:.2f} s) | PARK {pm['gbs']:.3f} GB/s "
          f"({pm['mibs']:.0f} MiB/s, {pm['seconds']:.2f} s)")
    if a.device == "cuda":
        sp, raw = result["fetch_split"], result["raw_h2d"]
        print(f"[store_bandwidth] split: host-copy {sp['host_copy']['seconds']:.2f} s "
              f"({sp['host_copy']['gbs']:.3f} GB/s) vs H2D {sp['host_to_device']['seconds']:.2f} s "
              f"({sp['host_to_device']['gbs']:.3f} GB/s) | raw 256MiB pageable "
              f"{raw['raw_pageable_h2d_gbs']:.2f} GB/s, pinned {raw['raw_pinned_h2d_gbs']:.2f} GB/s "
              f"-> {out}")
    else:
        print(f"[store_bandwidth] store fs: "
              f"{result.get('store_fs', {}).get('fstype')} -> {out}")


if __name__ == "__main__":
    main()
