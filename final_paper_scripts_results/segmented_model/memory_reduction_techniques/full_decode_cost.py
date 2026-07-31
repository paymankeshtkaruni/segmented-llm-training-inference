#!/usr/bin/env python
"""
FULL-MODEL decode cost — per-token time of greedy generation, measured under the SAME
protocol as the segmented inference ladder (infer_cost_lib.py): batch 1, random
256-token prompt, 1 warmup token, then 8 timed greedy tokens, per_token = total/8.

WHY: the dumbbell figure (comp_cost_dumbbell) annotates each endpoint with
(peak memory, time). The training rows pair full step vs segmented step on one
protocol; this run supplies the missing like-for-like decode timing for the
inference rows (the existing full_infer profiler times a batched forward, not decode).

The full model is the project's reference GPTDecoder (src package, unchanged),
executed the standard resident way: full forward over the growing sequence each
step, no KV cache — the same "standard implementation" the paper baselines against
(mirrors full_model/scripts/_infer_lib.generate_batch, at batch 1).

Usage:
  python full_decode_cost.py --preset large_8x2x2x8 --device cuda
  python full_decode_cost.py --preset large_8x2x2x8 --device cpu
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE / "segmentation_management"))
sys.path.insert(0, str(REPO / "src"))
from config import get_preset                             # noqa: E402

_MB = 1024 ** 2


def build_full_model(m):
    from sequential_segmented_llm_training_inference.model.full_model import GPTDecoder
    return GPTDecoder(vocab_size=m.vocab_size, d_model=m.d_model, n_heads=m.n_heads,
                      n_layers=m.n_layers, d_ff=m.d_ff, max_seq_len=m.max_seq_len,
                      dropout=m.dropout)


@torch.no_grad()
def greedy_decode(model, ids, n_new, device, pad=None):
    """Standard resident decode: full forward over the growing sequence each step."""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = []
    for _ in range(n_new):
        logits, _ = model(x, pad_token_id=pad)
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="large_8x2x2x8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--gen-tokens", type=int, default=8)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    is_cuda = a.device.startswith("cuda")
    out_dir = Path(a.out_dir) if a.out_dir else (
        HERE / "results" / f"full_decode_{'gpu' if is_cuda else 'cpu'}")
    out_dir.mkdir(parents=True, exist_ok=True)

    p = get_preset(a.preset)
    m = p["model"]
    torch.manual_seed(42)
    model = build_full_model(m).to(a.device)
    model.eval()
    n_params = sum(pp.numel() for pp in model.parameters())

    # identical prompt construction to infer_cost_lib.run_infer_cost
    g = torch.Generator().manual_seed(0)
    plen = min(a.prompt_len, max(1, m.max_seq_len - a.gen_tokens))
    ids = torch.randint(0, m.vocab_size, (plen,), generator=g).tolist()

    if is_cuda:
        torch.cuda.synchronize()
    greedy_decode(model, ids, 1, a.device)                      # warmup token
    if is_cuda:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = greedy_decode(model, ids, a.gen_tokens, a.device)
    if is_cuda:
        torch.cuda.synchronize()
    gen_s = time.perf_counter() - t0

    n = max(1, len(out))
    metrics = {
        "run": "full_decode_cost", "preset": a.preset, "device": a.device,
        "model_config": m.to_dict(), "params_million": round(n_params / 1e6, 1),
        "prompt_len": len(ids), "gen_tokens": n,
        "protocol": "identical to segmented infer_cost_lib: batch 1, greedy, warmup 1 "
                    "token, full forward per step (standard resident implementation, "
                    "no KV cache)",
        "generate_total_s": round(gen_s, 3),
        "per_token_s": round(gen_s / n, 3),
        "vram_hw_reserved_peak_mb": (round(torch.cuda.max_memory_reserved() / _MB, 1)
                                     if is_cuda else None),
        "rss_hw_peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
    }
    json.dump(metrics, open(out_dir / "full_decode_metrics.json", "w"), indent=2)
    print(f"[full-decode] {a.preset} on {a.device}  params={n_params/1e6:.0f}M  "
          f"prompt={len(ids)} gen={n}  per-token={metrics['per_token_s']:.3f}s  "
          f"-> {out_dir/'full_decode_metrics.json'}")


if __name__ == "__main__":
    main()
