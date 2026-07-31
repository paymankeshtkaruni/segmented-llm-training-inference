"""
Backward VRAM accumulation probe — MEASURE (don't guess) where on-device memory
goes during one segmented backward of the LARGE model.

Logs torch.cuda.memory_allocated() at every segment load/release (tagged by phase,
layer, kind), plus explicit checkpoints around _record_layer_inputs and the CE
backward, then prints: the resident floor after recording, the running peak, and the
per-layer allocated trend (does it climb with depth?). This pinpoints the accumulation
the headline number reflects, so any fix is evidence-driven.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

SEG_PKG = Path(__file__).resolve().parent.parent / "segmentation_management"
sys.path.insert(0, str(SEG_PKG))
from trainer import SegmentedTrainer        # noqa: E402


def MB(x): return x / (1024 ** 2)


class MemTraceProfiler:
    def __init__(self):
        self.phase = "?"; self.trace = []  # (phase, layer, kind, alloc_mb, reserved_mb)

    def set_phase(self, p): self.phase = p

    def on_load(self, key): pass

    def on_release(self, key):
        self.trace.append((self.phase, key.layer_id, key.kind,
                           MB(torch.cuda.memory_allocated()),
                           MB(torch.cuda.memory_reserved())))


def main():
    dev = "cuda"
    tr = SegmentedTrainer("large_8x2x2x8", dev, Path("/tmp/seg_mem_probe_work"),
                          store_kind="cpu_ram")
    m, s = tr.m, tr.s
    B, T = 4, m.max_seq_len
    pad = tr.tok.pad_token_id
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, m.vocab_size, (B, T), generator=g)
    if pad is not None: x[x == pad] = (int(pad) + 1) % m.vocab_size
    y = x.clone(); y[:, 0] = -100
    x, y = x.to(dev), y.to(dev)

    prof = MemTraceProfiler(); tr.loader.profiler = prof
    bwd = tr.bwd

    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    a0 = MB(torch.cuda.memory_allocated())
    print(f"[probe] large {m.n_params_estimate/1e6:.0f}M  B={B} T={T}  d_model={m.d_model} "
          f"layers={m.n_layers} vocab={m.vocab_size}")
    print(f"  baseline allocated (shared params resident): {a0:.0f} MB")

    # --- measure the record step in isolation ---
    prof.set_phase("record")
    li, hlast = bwd._record_layer_inputs(x, pad)
    torch.cuda.synchronize()
    li_bytes = sum(t.element_size() * t.nelement() for t in li)
    print(f"  after _record_layer_inputs: allocated={MB(torch.cuda.memory_allocated()):.0f} MB "
          f"| layer_inputs list = {len(li)} x {list(li[0].shape)} = {MB(li_bytes):.0f} MB resident")
    del li, hlast
    torch.cuda.empty_cache(); torch.cuda.synchronize()

    # --- full backward with per-segment trace ---
    prof.set_phase("backward")
    torch.cuda.reset_peak_memory_stats()
    _ = bwd.backward(x, y, pad_token_id=pad)
    torch.cuda.synchronize()
    peak = MB(torch.cuda.max_memory_allocated())
    print(f"  backward peak allocated (torch): {peak:.0f} MB   reserved peak: "
          f"{MB(torch.cuda.max_memory_reserved()):.0f} MB")

    # per-layer trend: allocated at the LAST release within each layer
    bylayer = {}
    for ph, L, kind, alloc, res in prof.trace:
        if ph == "backward":
            bylayer[L] = (alloc, res)
    layers = sorted([L for L in bylayer if L >= 0], reverse=True)
    print("  per-layer allocated (MB) at layer end [L: alloc / reserved]:")
    show = layers[:3] + ["..."] + layers[-3:] if len(layers) > 6 else layers
    for L in show:
        if L == "...": print("     ..."); continue
        a, r = bylayer[L]; print(f"     L{L:>2}: {a:7.0f} / {r:7.0f}")
    # phase peaks within trace
    mx = max((a for _, _, _, a, _ in prof.trace), default=0)
    print(f"  max allocated seen at any segment boundary: {mx:.0f} MB")


if __name__ == "__main__":
    main()
