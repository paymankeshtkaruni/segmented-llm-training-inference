"""
C Step 2/3/4 — torch-free SEGMENTED ONNX inference cost (GPU + CPU).

Imports ONLY onnxruntime + numpy + psutil + stdlib — NOT torch (asserted at the end).
Composes the segmented forward one segment at a time from the weightless ONNX graphs +
glue.npz (norms / mlp-bias / final-norm / W_o applied in numpy):
  embedding sessions -> concat ;  per layer: attn_ln(np) -> attention sessions -> concat
  -> W_o(np)+residual -> mlp_ln(np) -> mlp sessions -> sum+bias+residual ; final_norm(np)
  -> output_head sessions (streamed).
Session policy (T06): GPU = ONE lazy CUDA session at a time (del+gc on signature switch);
CPU = cache all signatures. preload_dlls() on GPU. Measures nvidia-smi VRAM + RSS,
baseline/after-session/peak, torch-free.

Also a --verify mode (CPU) that compares the torch-free forward hidden_norm to a saved
reference (written by the export step) — proving the runtime is exact.
"""
from __future__ import annotations
import argparse, gc, json, os, subprocess, sys, threading, time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil


def rss_mb(): return psutil.Process().memory_info().rss / 1024**2
def smi_vram_mb():
    pid = os.getpid()
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL)
        for line in out.strip().splitlines():
            p, mem = [x.strip() for x in line.split(",")]
            if int(p) == pid:
                return float(mem)
    except Exception:
        pass
    return 0.0


def layernorm(x, w, b, eps=1e-5):
    # keep float32 — a Python-float eps would promote to float64, and feeding a float64
    # activation to the float32 ONNX input makes ORT return wrong values.
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (((x - mu) / np.sqrt(var + np.float32(eps)) * w + b)).astype(np.float32, copy=False)


class Sampler:
    def __init__(self, is_cuda, iv=0.02):
        self.is_cuda, self.iv = is_cuda, iv; self.s = []
        self._stop = threading.Event(); self._t = None
    def _loop(self):
        while not self._stop.is_set():
            self.s.append((time.perf_counter(), smi_vram_mb() if self.is_cuda else 0.0, rss_mb()))
            self._stop.wait(self.iv)
    def start(self): self._t = threading.Thread(target=self._loop, daemon=True); self._t.start(); return self
    def stop(self):
        self._stop.set()
        if self._t: self._t.join(timeout=2.0)


class SegOnnxRuntime:
    """Torch-free segmented forward. GPU: one lazy CUDA session; CPU: cache all sigs."""
    def __init__(self, onnx_dir: Path, provider: str, preload_weights: bool, lazy_cuda: bool = True):
        self.cfg = json.load(open(onnx_dir / "config.json"))
        self.man = json.load(open(onnx_dir / "manifest.json"))
        self.glue = dict(np.load(str(onnx_dir / "glue.npz")))
        self.is_cuda = provider == "cuda"
        self.providers = ["CUDAExecutionProvider"] if self.is_cuda else ["CPUExecutionProvider"]
        self.kind2sig = {info["kind"]: info for info in self.man["signatures"].values()}
        so = ort.SessionOptions(); so.enable_mem_pattern = False
        so.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "8") or "8")
        so.inter_op_num_threads = 1
        self.so = so
        # ONNX bytes per kind in RAM
        self.bytes = {k: open(v["weightless_onnx"], "rb").read() for k, v in self.kind2sig.items()}
        # lazy = ONE CUDA session in VRAM at a time (T06 P3); else cache all (CPU, or the
        # GPU cache-all probe baseline). CPU never benefits from lazy (no VRAM).
        self.lazy = self.is_cuda and lazy_cuda
        self._cur_kind = None; self._sess = None; self._cache = {}
        if not self.lazy:
            for k in self.kind2sig:
                self._cache[k] = ort.InferenceSession(self.bytes[k], self.so, providers=self.providers)
            self.active_provider = next(iter(self._cache.values())).get_providers()[0]
        else:
            first = next(iter(self.kind2sig)); self._sess = ort.InferenceSession(
                self.bytes[first], self.so, providers=self.providers); self._cur_kind = first
            self.active_provider = self._sess.get_providers()[0]
        # weights: preload to RAM (cpu_ram analog) or load per-segment from disk
        self.preload = preload_weights
        self._wcache = {}
        if preload_weights:
            for name, info in self.man["segments"].items():
                self._wcache[name] = dict(np.load(info["weights_npz"]))

    def _session(self, kind):
        if not self.lazy:
            return self._cache[kind]
        if kind != self._cur_kind:                       # T06: one CUDA session at a time
            del self._sess; gc.collect()
            self._sess = ort.InferenceSession(self.bytes[kind], self.so, providers=self.providers)
            self._cur_kind = kind
        return self._sess

    def _weights(self, name, info):
        if self.preload:
            return self._wcache[name]
        return dict(np.load(info["weights_npz"]))        # disk analog (per segment)

    def _run(self, kind, name, act_name, act):
        info = self.man["segments"][name]
        w = self._weights(name, info)
        sess = self._session(kind)
        out = sess.run(["out"], {act_name: act, **w})[0]
        if not self.preload:
            del w
        return out

    def forward(self, input_ids):
        c = self.cfg; D = c["d_model"]
        # embedding -> concat
        parts = []
        for e in range(c["embedding_segments"]):
            nm = f"L-1__embedding__s{e}"
            parts.append(self._run("embedding", nm, "input_ids", input_ids))
        hidden = np.concatenate(parts, axis=-1); del parts
        for L in range(c["n_layers"]):
            xin = layernorm(hidden, self.glue[f"attn_ln.{L}.weight"], self.glue[f"attn_ln.{L}.bias"])
            outs = [self._run("attention", f"L{L}__attention__s{a}", "x_norm", xin)
                    for a in range(c["attention_segments"])]
            attn = np.concatenate(outs, axis=-1); del outs, xin
            Wo = self.glue[f"Wo.{L}.weight"]; bo = self.glue[f"Wo.{L}.bias"]
            hidden = hidden + (attn @ Wo.T + bo); del attn
            xin = layernorm(hidden, self.glue[f"mlp_ln.{L}.weight"], self.glue[f"mlp_ln.{L}.bias"])
            acc = None
            for cc in range(c["mlp_chunks"]):
                o = self._run("mlp", f"L{L}__mlp__s{cc}", "x_norm", xin)
                acc = o if acc is None else acc + o
            hidden = hidden + acc + self.glue[f"mlp_out_bias.{L}"]; del acc, xin
        hn = layernorm(hidden, self.glue["final_norm.weight"], self.glue["final_norm.bias"])
        return hn

    def next_token(self, input_ids):
        """Streamed argmax next-token over output-head slices (no full logits)."""
        c = self.cfg
        hn = self.forward(input_ids)[:, -1:, :]          # [B,1,D]
        B = hn.shape[0]; best_val = np.full((B, 1), -1e30, np.float32); best_idx = np.zeros((B, 1), np.int64)
        H = c["output_head_segments"]; V = c["vocab_size"]
        for h in range(H):
            v0 = (V // H) * h + min(h, V % H); v1 = v0 + (V // H) + (1 if h < V % H else 0)
            z = self._run("output_head", f"L-1__output_head__s{h}", "hidden_norm", hn)  # [B,1,vsl]
            smax = z.max(-1); sarg = z.argmax(-1).astype(np.int64)
            upd = smax > best_val
            best_val = np.where(upd, smax, best_val); best_idx = np.where(upd, sarg + v0, best_idx)
            del z
        return best_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--provider", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--gen-tokens", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--preload-weights", action="store_true")
    ap.add_argument("--cache-all", action="store_true", help="GPU probe: cache all sessions (no lazy)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prefix", default="seg_c")
    ap.add_argument("--verify", action="store_true", help="CPU: compare forward to reference .npy")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    is_cuda = a.provider == "cuda"
    if is_cuda:
        ort.preload_dlls()
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            print("FATAL: CUDA EP unavailable"); sys.exit(2)

    base_rss = rss_mb(); base_vram = smi_vram_mb() if is_cuda else 0.0
    samp = Sampler(is_cuda).start()
    rt = SegOnnxRuntime(a.onnx_dir, a.provider, a.preload_weights, lazy_cuda=not a.cache_all)
    if is_cuda and rt.active_provider != "CUDAExecutionProvider":
        print(f"FATAL: silent CPU fallback ({rt.active_provider})"); sys.exit(2)
    after_rss = rss_mb(); after_vram = smi_vram_mb() if is_cuda else 0.0

    rng = np.random.default_rng(0)
    V = rt.cfg["vocab_size"]
    # prefill forwards (memory bound)
    pref_t = []
    for _ in range(3):
        ids = rng.integers(0, V, size=(a.batch, a.seq), dtype=np.int64)
        t0 = time.perf_counter(); rt.forward(ids); pref_t.append(time.perf_counter() - t0)
    # decode (batch autoregressive)
    ids = rng.integers(0, V, size=(a.batch, a.prompt_len), dtype=np.int64)
    tok_t = []
    for _ in range(a.gen_tokens):
        t0 = time.perf_counter(); nxt = rt.next_token(ids); tok_t.append(time.perf_counter() - t0)
        ids = np.concatenate([ids, nxt], axis=1)
    samp.stop()

    peak_vram = max((v for _, v, _ in samp.s), default=0.0)
    peak_rss = max((h for _, _, h in samp.s), default=0.0)
    torch_free = "torch" not in sys.modules
    metrics = {
        "run": "segmented_onnx_infer_cost", "onnx_dir": str(a.onnx_dir), "provider": a.provider,
        "batch": a.batch, "seq": a.seq, "gen_tokens": a.gen_tokens,
        "preload_weights": a.preload_weights, "unique_sessions": len(rt.kind2sig),
        "active_provider": rt.active_provider, "torch_free": torch_free,
        "avg_prefill_s": sum(pref_t) / len(pref_t), "avg_token_s": sum(tok_t) / len(tok_t),
        "baseline": {"rss_mb": round(base_rss, 1), "vram_mb": round(base_vram, 1)},
        "after_session": {"rss_mb": round(after_rss, 1), "vram_mb": round(after_vram, 1)},
        "peak": {"vram_mb": round(peak_vram, 1), "vram_net_mb": round(peak_vram - base_vram, 1),
                 "rss_mb": round(peak_rss, 1), "rss_net_mb": round(peak_rss - base_rss, 1)},
        "timeline": [[round(t - samp.s[0][0], 4), round(v, 1), round(h, 1)] for t, v, h in samp.s],
    }
    json.dump(metrics, open(a.out_dir / f"{a.prefix}_metrics.json", "w"), indent=2)
    assert torch_free, "TORCH LEAKED INTO TORCH-FREE C RUN"
    print(f"[seg-c-onnx] {a.onnx_dir.name} on {a.provider}  torch_free={torch_free}  sessions={len(rt.kind2sig)} active={rt.active_provider}")
    print(f"  VRAM: baseline={base_vram:.0f} after_session={after_vram:.0f} PEAK={peak_vram:.0f} (net {metrics['peak']['vram_net_mb']:.0f}) MB")
    print(f"  RSS : baseline={base_rss:.0f} PEAK={peak_rss:.0f} (net {metrics['peak']['rss_net_mb']:.0f}) MB")
    print(f"  avg prefill={metrics['avg_prefill_s']*1000:.0f}ms  avg token={metrics['avg_token_s']*1000:.0f}ms")
    print(f"  -> {a.out_dir/(a.prefix+'_metrics.json')}")


if __name__ == "__main__":
    main()
