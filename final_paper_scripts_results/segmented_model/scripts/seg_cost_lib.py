"""
Segmented cost profiler (A2 / D) — honest, fine-grained, CATEGORY-SEPARATED memory.

Tracks the REAL memory flow of EVERY operation in a training step (every embedding-cat,
residual add, norm, attention head-group, MLP chunk, CE vocab-slice, autograd.grad, ...
— not just a per-phase total), on ONE synchronized clock, with VRAM and CPU-RAM kept
as TWO separate series for TWO plots, and without blending physically-different memory:

  VRAM (device):
    * CUDA context            — driver / non-torch VRAM           = nvidia-smi - reserved (~const)
    * torch cache             — reserved pool slack, NOT live data = reserved - allocated
    * torch allocated/RESIDENT — params + records + grads carried  = allocated floor between ops
    * torch allocated/OPERATION— the current op's own transient    = allocated rise during the op
  CPU-RAM (host / python process):
    * RSS                     — process resident set (the constrained resource on CPU runs)

How (no cheating, no curated peak):
  * one high-frequency thread samples (torch.allocated, torch.reserved, RSS) together
    -> two synchronized timelines showing every change in VRAM and CPU-RAM;
  * the loader hook marks every segment load/release AND the engines' optrace marks
    every non-segment op -> a labeled, ordered per-operation trace with exact memory;
  * a slow thread samples nvidia-smi to calibrate the CUDA context (verified:
    context + reserved ~= nvidia-smi).
The verified segmentation_management engines run UNCHANGED (marks are no-ops here only
because we install the hook; default behavior is untouched — identity tests pass).
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import torch

SEG_PKG = Path(__file__).resolve().parent.parent / "segmentation_management"
sys.path.insert(0, str(SEG_PKG))
import optrace                                           # noqa: E402
from trainer import SegmentedTrainer                    # noqa: E402
from memory import host_rss_mb, smi_process_vram_mb     # noqa: E402

_MB = 1024 ** 2


class MemFlow:
    """Synchronized VRAM + CPU-RAM recorder with per-operation annotation.
    Serves as BOTH the loader profiler hook (on_load/on_release) and the optrace
    operation hook (op_mark), so every move — segment or not — lands in one trace."""

    def __init__(self, is_cuda: bool, fast_iv=0.001, slow_iv=0.03):
        self.is_cuda = is_cuda
        self.fast_iv, self.slow_iv = fast_iv, slow_iv
        self.fast = []          # (t, vram_alloc_mb, vram_reserved_mb, rss_mb)  — hi-freq, BOTH
        self.slow = []          # (t, smi_vram_mb)                              — lo-freq (context)
        self.moves = []         # (t, phase, label, vram_alloc_mb, vram_reserved_mb, rss_mb)
        self.phase = "?"
        self._t0 = {}
        self._stop = threading.Event(); self._tf = self._ts = None

    # ---- precise point read (main thread, at an op boundary) ----
    def _point(self):
        a = torch.cuda.memory_allocated() / _MB if self.is_cuda else 0.0
        r = torch.cuda.memory_reserved() / _MB if self.is_cuda else 0.0
        return a, r, host_rss_mb()

    # ---- background samplers ----
    def _floop(self):
        ma, mr = torch.cuda.memory_allocated, torch.cuda.memory_reserved
        cuda = self.is_cuda
        while not self._stop.is_set():
            a = ma() / _MB if cuda else 0.0
            r = mr() / _MB if cuda else 0.0
            self.fast.append((time.perf_counter(), a, r, host_rss_mb()))
            self._stop.wait(self.fast_iv)

    def _sloop(self):
        while not self._stop.is_set():
            self.slow.append((time.perf_counter(), smi_process_vram_mb()))
            self._stop.wait(self.slow_iv)

    def start(self, fast=True, slow=True):
        if fast:
            self._tf = threading.Thread(target=self._floop, daemon=True); self._tf.start()
        if slow and self.is_cuda:
            self._ts = threading.Thread(target=self._sloop, daemon=True); self._ts.start()
        return self

    def stop(self):
        self._stop.set()
        for t in (self._tf, self._ts):
            if t: t.join(timeout=2.0)

    # ---- annotation interfaces ----
    def set_phase(self, phase):
        self.phase = phase
        a, r, h = self._point(); self.moves.append((time.perf_counter(), phase, "<phase>", a, r, h))

    def op_mark(self, label):                       # optrace hook: non-segment ops
        a, r, h = self._point()
        self.moves.append((time.perf_counter(), self.phase, label, a, r, h))

    def on_load(self, key):                          # loader hook: segment acquired
        self._t0[key] = time.perf_counter()
        a, r, h = self._point()
        self.moves.append((self._t0[key], self.phase,
                           f"load|{key.kind}|L{key.layer_id}|s{key.seg}", a, r, h))

    def on_release(self, key):                        # loader hook: segment freed
        a, r, h = self._point()
        self.moves.append((time.perf_counter(), self.phase,
                           f"release|{key.kind}|L{key.layer_id}|s{key.seg}", a, r, h))

    # ---- analysis ----
    def _fast_in(self, t0, t1):
        return [(a, r, h) for t, a, r, h in self.fast if t0 <= t < t1]

    def context_mb(self):
        if not self.is_cuda or not self.slow or not self.fast:
            return 0.0
        ctx = []
        for t, smi in self.slow:
            near = min(self.fast, key=lambda r: abs(r[0] - t))
            if smi > 0:
                ctx.append(smi - near[2])             # smi - reserved
        ctx = sorted(c for c in ctx if c > 0)
        return ctx[len(ctx) // 2] if ctx else 0.0


def _phase_breakdown(mf, ctx, t0, t1):
    fa = mf._fast_in(t0, t1)
    if not fa:
        return None
    al = [a for a, _, _ in fa]; rs = [r for _, r, _ in fa]; hs = [h for _, _, h in fa]
    floor, apeak, rpeak = min(al), max(al), max(rs)
    return {
        "vram": {
            "context_mb": round(ctx, 1),
            "torch_cache_mb": round(max(0.0, rpeak - apeak), 1),
            "alloc_resident_mb": round(floor, 1),
            "alloc_operation_mb": round(apeak - floor, 1),
            "alloc_peak_mb": round(apeak, 1),
            "reserved_peak_mb": round(rpeak, 1),
            "total_peak_mb": round(ctx + rpeak, 1),
        },
        "cpu_ram": {
            "rss_floor_mb": round(min(hs), 1),
            "rss_peak_mb": round(max(hs), 1),
            "rss_operation_mb": round(max(hs) - min(hs), 1),
        },
        "time_ms": round((t1 - t0) * 1000, 1),
    }


import re as _re
_LSEG = _re.compile(r"^(L-?\d+|s\d+|h\d+|c\d+)$")
def _op_family(label):
    """Collapse a move label to its operation family by dropping layer/segment/slice
    indices, so e.g. 'bwd|L35|mlp_chunk0' and 'bwd|L0|mlp_chunk1' both map to
    'bwd|mlp_chunk' and 'load|attention|L9|s1' maps to 'load|attention'."""
    toks = []
    for tok in label.split("|"):
        if _LSEG.match(tok):
            continue
        toks.append(_re.sub(r"\d+$", "", tok))      # strip trailing index (chunk0 -> chunk)
    return "|".join(toks)


def _synth(batch, seq, vocab, pad, device, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (batch, seq), generator=g)
    if pad is not None:
        ids[ids == pad] = (int(pad) + 1) % vocab
    lab = ids.clone(); lab[:, 0] = -100
    return ids.to(device), lab.to(device)


def _fwd(tr, x, pad):
    tr.fwd.train()
    with torch.no_grad():
        return tr.fwd.forward_hidden(x, pad_token_id=pad)


def _val(tr, x, y, pad):
    tr.fwd.eval()
    with torch.no_grad():
        h = tr.fwd.forward_hidden(x, pad_token_id=pad); tr.fwd.chunked_ce(h, y)
    tr.fwd.train()


def run_cost(preset, device, out_dir: Path, prefix, batch=4, n_steps=3, seq_len=None, from_scratch=True):
    is_cuda = str(device).startswith("cuda")
    out_dir.mkdir(parents=True, exist_ok=True)
    host_baseline = host_rss_mb()
    if is_cuda:
        torch.zeros(1, device=device); torch.cuda.synchronize()

    # start sampling BEFORE building the model, so the timeline covers A2 from the
    # very beginning (build -> warmup -> every step) end to end.
    mf = MemFlow(is_cuda); mf.start()
    mf.set_phase("build")
    tr = SegmentedTrainer(preset, device, out_dir / "_work",
                          store_kind=("cpu_ram" if is_cuda else "disk"), from_scratch=from_scratch)
    m, s = tr.m, tr.s
    seq = seq_len or m.max_seq_len
    pad = tr.tok.pad_token_id
    tr.loader.profiler = mf
    optrace.set_hook(mf.op_mark)                      # engines' per-op marks -> mf
    host_after_build = host_rss_mb()

    mf.set_phase("warmup")
    xw, yw = _synth(batch, seq, m.vocab_size, pad, device, 0)
    grw = tr.bwd.backward(xw, yw, pad_token_id=pad); tr.opt.step(grw, clip_norm=1.0)
    del xw, yw, grw

    windows = []; step_times = []; last_grads = None
    for step in range(n_steps):
        x, y = _synth(batch, seq, m.vocab_size, pad, device, 1 + step)
        s0 = time.perf_counter()
        for ph, fn in [
            ("forward",    lambda: _fwd(tr, x, pad)),
            ("backward",   lambda: tr.bwd.backward(x, y, pad_token_id=pad)),
            ("optimizer",  lambda: tr.opt.step(last_grads, clip_norm=1.0)),
            ("validation", lambda: _val(tr, x, y, pad)),
        ]:
            mf.set_phase(ph)
            # high-water mark: the allocator records EVERY allocation's peak, so the
            # phase peak cannot be missed (unlike 1 kHz sampling). Reset at phase start.
            if is_cuda:
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            out = fn()
            if ph == "backward":
                last_grads = out
            if is_cuda:
                torch.cuda.synchronize()
            hw_a = torch.cuda.max_memory_allocated() / _MB if is_cuda else 0.0
            hw_r = torch.cuda.max_memory_reserved() / _MB if is_cuda else 0.0
            windows.append((ph, step, t0, time.perf_counter(), hw_a, hw_r))
        step_times.append(time.perf_counter() - s0)
        # faithful: real training releases the step's grads before the next backward
        last_grads = None; del x, y
        gc.collect()
        if is_cuda:
            torch.cuda.empty_cache()
    mf.stop(); optrace.set_hook(None)

    ctx = mf.context_mb()
    rss_hw_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB->MB, no-miss lifetime peak
    per_phase = {}
    for ph in ("forward", "backward", "optimizer", "validation"):
        wins = [(t0, t1, hwa, hwr) for p, st, t0, t1, hwa, hwr in windows if p == ph]
        if wins:
            t0, t1, hwa, hwr = wins[-1]                 # steady-state step
            bd = _phase_breakdown(mf, ctx, t0, t1)
            if bd:
                # GUARANTEED (no-miss) peak from the allocator high-water mark
                bd["vram"]["hw_alloc_peak_mb"] = round(hwa, 1)
                bd["vram"]["hw_reserved_peak_mb"] = round(hwr, 1)
                bd["vram"]["hw_total_peak_mb"] = round(ctx + hwr, 1)   # +context = true total
                per_phase[ph] = bd

    # per-operation working set (every move): for each move, op-working = alloc rise
    # from the previous move's alloc floor; aggregate the max per (phase,op-label-base).
    st0 = mf.fast[0][0] if mf.fast else (windows[0][2] if windows else 0.0)   # build start = t0
    per_op = defaultdict(lambda: {"count": 0, "vram_op_max_mb": 0.0, "rss_op_max_mb": 0.0,
                                  "vram_alloc_max_mb": 0.0})
    prev_a = prev_h = None
    for t, ph, label, a, r, h in mf.moves:
        if ph in ("forward", "backward", "optimizer", "validation") and label != "<phase>":
            base = _op_family(label)                 # op family (layer/seg index stripped)
            key = f"{ph}|{base}"
            d = per_op[key]; d["count"] += 1
            d["vram_alloc_max_mb"] = max(d["vram_alloc_max_mb"], a)
            if prev_a is not None:
                d["vram_op_max_mb"] = max(d["vram_op_max_mb"], a - prev_a)
                d["rss_op_max_mb"] = max(d["rss_op_max_mb"], h - prev_h)
        prev_a, prev_h = a, h

    metrics = {
        "run": "segmented_cost", "preset": preset, "device": device,
        "model_config": m.to_dict(), "seg_config": s.to_dict(),
        "batch_size": batch, "seq_len": seq, "n_steps": n_steps,
        "avg_step_time_s": sum(step_times) / len(step_times),
        "measurement": "synchronized VRAM+CPU-RAM, per-operation; categories not blended",
        "baseline": {"host_framework_mb": round(host_baseline, 1),
                     "host_after_build_mb": round(host_after_build, 1),
                     "cuda_context_mb": round(ctx, 1)},
        "overall": {
            # sampled (flow): nvidia-smi total + torch counters from the 1 kHz thread
            "vram_smi_peak_mb": round(max((v for _, v in mf.slow), default=0.0), 1),
            "vram_reserved_peak_mb": round(max((r for _, _, r, _ in mf.fast), default=0.0), 1),
            "vram_alloc_peak_mb": round(max((a for _, a, _, _ in mf.fast), default=0.0), 1),
            "rss_peak_sampled_mb": round(max((h for _, _, _, h in mf.fast), default=0.0), 1),
            # GUARANTEED no-miss high-water marks (cannot miss any allocation):
            "vram_hw_reserved_peak_mb": round(max((w[5] for w in windows), default=0.0), 1),
            "vram_hw_total_peak_mb": round(ctx + max((w[5] for w in windows), default=0.0), 1),
            "rss_hw_peak_mb": round(rss_hw_mb, 1),
        },
        "per_phase": per_phase,
        "per_operation": {k: {"count": v["count"],
                              "vram_op_max_mb": round(v["vram_op_max_mb"], 1),
                              "vram_alloc_max_mb": round(v["vram_alloc_max_mb"], 1),
                              "rss_op_max_mb": round(v["rss_op_max_mb"], 1)}
                          for k, v in sorted(per_op.items())},
        # TWO synchronized timelines for TWO plots (every change), same clock:
        "vram_timeline": [[round(t - st0, 4), round(a, 1), round(r, 1)] for t, a, r, _ in mf.fast],
        "cpu_ram_timeline": [[round(t - st0, 4), round(h, 1)] for t, _, _, h in mf.fast],
        # full labeled per-operation trace (every move):
        "moves": [[round(t - st0, 4), ph, lbl, round(a, 1), round(r, 1), round(h, 1)]
                  for t, ph, lbl, a, r, h in mf.moves],
    }
    json.dump(metrics, open(out_dir / f"{prefix}_metrics.json", "w"), indent=2)

    print(f"[seg-cost] {preset} on {device}  {m.n_params_estimate/1e6:.0f}M  seg={s.code}  bs={batch} seq={seq}")
    print(f"  CUDA context={ctx:.0f}MB  fast-samples={len(mf.fast)}  moves={len(mf.moves)}")
    o = metrics["overall"]
    print(f"  OVERALL VRAM: smi={o['vram_smi_peak_mb']:.0f} reserved_sampled={o['vram_reserved_peak_mb']:.0f} "
          f"| HW(no-miss) total={o['vram_hw_total_peak_mb']:.0f}  ||  "
          f"CPU-RAM RSS sampled={o['rss_peak_sampled_mb']:.0f} HW={o['rss_hw_peak_mb']:.0f} MB")
    print(f"  {'phase':11} {'ctx':>5} {'cache':>6} {'resident':>9} {'op':>6} {'=vram':>7} | {'rss_floor':>9} {'rss_op':>7} {'time':>7}")
    for ph in ("forward", "backward", "optimizer", "validation"):
        if ph in per_phase:
            v = per_phase[ph]["vram"]; c = per_phase[ph]["cpu_ram"]
            print(f"  {ph:11} {v['context_mb']:5.0f} {v['torch_cache_mb']:6.0f} "
                  f"{v['alloc_resident_mb']:9.0f} {v['alloc_operation_mb']:6.0f} {v['total_peak_mb']:7.0f} | "
                  f"{c['rss_floor_mb']:9.0f} {c['rss_operation_mb']:7.0f} {per_phase[ph]['time_ms']:6.0f}ms")
    print(f"  -> {out_dir/(prefix+'_metrics.json')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="large_8x2x2x8")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="seg_cost")
    p.add_argument("--from-reference", action="store_true",
                   help="init by building the FULL reference model then slicing (causes the build "
                        "spike). Default: from-scratch, one segment at a time (no full model).")
    a = p.parse_args()
    run_cost(a.preset, a.device, a.out_dir, a.prefix, batch=a.batch, n_steps=a.n_steps,
             seq_len=a.seq_len, from_scratch=not a.from_reference)
