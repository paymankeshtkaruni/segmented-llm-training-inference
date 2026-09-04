#!/usr/bin/env python
"""Inference-cost measurement for the RESIDENT ONNX modes.

O1 full   : one full-model session (weights baked, external data) — the fastest
            torch-free mode; whole model resident.
O2 baked  : one session per segment, weights baked — resident, segmented
            execution (glue applied in numpy exactly like the weights-as-inputs
            runtime, but no per-call weight feeding).

Same conventions as seg_c_onnx_cost: prompt prefill + greedy decode, batch 1,
RSS/VRAM sampling, per-token time. Torch-free at runtime (onnxruntime + numpy).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
from seg_c_onnx_cost import Sampler, layernorm, rss_mb, smi_vram_mb   # noqa: E402


def _so(threads: int):
    so = ort.SessionOptions()
    so.enable_mem_pattern = False
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return so


class FullRuntime:
    def __init__(self, onnx_dir: Path, provider: str, threads: int):
        self.cfg = json.load(open(onnx_dir / "config.json"))
        prov = ["CUDAExecutionProvider"] if provider == "cuda" else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(onnx_dir / self.cfg["model"]),
                                         _so(threads), providers=prov)
        self.active_provider = self.sess.get_providers()[0]

    def next_token(self, ids):
        logits = self.sess.run(["logits"], {"input_ids": ids})[0]
        return logits[:, -1].argmax(-1).astype(np.int64).reshape(-1, 1)


class BakedRuntime:
    """Per-segment baked sessions; forward mirrors the weights-as-inputs runtime."""

    def __init__(self, onnx_dir: Path, provider: str, threads: int):
        self.cfg = json.load(open(onnx_dir / "config.json"))
        self.glue = dict(np.load(str(onnx_dir / "glue.npz")))
        man = json.load(open(onnx_dir / "manifest.json"))
        prov = ["CUDAExecutionProvider"] if provider == "cuda" else ["CPUExecutionProvider"]
        so = _so(threads)
        self.sess = {name: ort.InferenceSession(info["onnx"], so, providers=prov)
                     for name, info in man["segments"].items()}
        self.act = {name: info["activation_input"] for name, info in man["segments"].items()}
        self.active_provider = next(iter(self.sess.values())).get_providers()[0]

    def _run(self, name, x):
        return self.sess[name].run(["out"], {self.act[name]: x})[0]

    def forward(self, input_ids):
        c = self.cfg
        parts = [self._run(f"L-1__embedding__s{e}", input_ids)
                 for e in range(c["embedding_segments"])]
        hidden = np.concatenate(parts, axis=-1); del parts
        for L in range(c["n_layers"]):
            xin = layernorm(hidden, self.glue[f"attn_ln.{L}.weight"], self.glue[f"attn_ln.{L}.bias"])
            outs = [self._run(f"L{L}__attention__s{a}", xin)
                    for a in range(c["attention_segments"])]
            attn = np.concatenate(outs, axis=-1); del outs, xin
            Wo = self.glue[f"Wo.{L}.weight"]; bo = self.glue[f"Wo.{L}.bias"]
            hidden = hidden + (attn @ Wo.T + bo); del attn
            xin = layernorm(hidden, self.glue[f"mlp_ln.{L}.weight"], self.glue[f"mlp_ln.{L}.bias"])
            acc = None
            for cc in range(c["mlp_chunks"]):
                o = self._run(f"L{L}__mlp__s{cc}", xin)
                acc = o if acc is None else acc + o
            hidden = hidden + acc + self.glue[f"mlp_out_bias.{L}"]; del acc, xin
        hn = layernorm(hidden, self.glue["final_norm.weight"], self.glue["final_norm.bias"])
        return hn

    def next_token(self, ids):
        c = self.cfg
        hn = self.forward(ids)[:, -1:, :]
        B = hn.shape[0]
        best_val = np.full((B, 1), -1e30, np.float32)
        best_idx = np.zeros((B, 1), np.int64)
        H = c["output_head_segments"]; V = c["vocab_size"]
        for h in range(H):
            v0 = (V // H) * h + min(h, V % H)
            z = self._run(f"L-1__output_head__s{h}", hn)
            smax = z.max(-1); sarg = z.argmax(-1).astype(np.int64)
            upd = smax > best_val
            best_val = np.where(upd, smax, best_val)
            best_idx = np.where(upd, sarg + v0, best_idx)
            del z
        return best_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["full", "baked"])
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--provider", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--prefix", default="onnx_resident")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    is_cuda = a.provider == "cuda"
    samp = Sampler(is_cuda).start()
    t0 = time.perf_counter()
    rt = (FullRuntime if a.mode == "full" else BakedRuntime)(a.onnx_dir, a.provider, a.threads)
    build_s = time.perf_counter() - t0
    cfg = rt.cfg
    rng = np.random.default_rng(7)
    ids = rng.integers(0, cfg["vocab_size"], (1, a.prompt_len)).astype(np.int64)

    t0 = time.perf_counter()
    nxt = rt.next_token(ids)                      # prefill + first token
    prefill_s = time.perf_counter() - t0
    ids = np.concatenate([ids, nxt], axis=1)
    t0 = time.perf_counter()
    for _ in range(a.gen_tokens - 1):
        win = ids[:, -cfg["max_seq_len"]:] if ids.shape[1] > cfg["max_seq_len"] else ids
        nxt = rt.next_token(win)
        ids = np.concatenate([ids, nxt], axis=1)
    gen_s = time.perf_counter() - t0
    samp.stop()
    vram_peak = max((v for _, v, _ in samp.s), default=0.0)
    rss_peak = max((r for _, _, r in samp.s), default=0.0)

    met = {"run": "onnx_resident_cost", "mode": a.mode, "provider": rt.active_provider,
           "prompt_len": a.prompt_len, "gen_tokens": a.gen_tokens,
           "build_s": round(build_s, 2), "prefill_plus_first_s": round(prefill_s, 3),
           "per_token_s": round(gen_s / max(1, a.gen_tokens - 1), 3),
           "rss_peak_mb": round(rss_peak, 1),
           "vram_peak_mb": round(vram_peak, 1) if is_cuda else None}
    out = a.out_dir / f"{a.prefix}_metrics.json"
    json.dump(met, open(out, "w"), indent=2)
    print(json.dumps(met, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
