#!/usr/bin/env python
"""ONNX rows of the fresh inference-accuracy table (compact quality config).

Drives the torch-free SegOnnxRuntime (seg_c export format, fresh trained weights)
through the same greedy exact-match task as the torch engines. Batches contain
only prompts of IDENTICAL length (no padding), so the export needs no pad
masking; causality is baked into the exported attention graphs (is_causal=True).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "segmentation_management"))
sys.path.insert(0, str(HERE.parent / "scripts"))

from data import build_tokenizer, LazyLogDataset, TEST_CSV     # noqa: E402
from config import PROMPT_TEMPLATE, get_preset                 # noqa: E402
from seg_c_onnx_cost import SegOnnxRuntime                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--provider", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--preload-weights", action="store_true")
    ap.add_argument("--preset", default="small_8x2x2x8")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results" / "quality_fresh_pair")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    p = get_preset(a.preset)
    tok = build_tokenizer(p["tokenizer"])
    eos = tok.eos_token_id
    max_seq = p["model"].max_seq_len

    rt = SegOnnxRuntime(a.onnx_dir, a.provider, a.preload_weights)
    print(f"[onnx] provider={rt.active_provider} preload={a.preload_weights}", flush=True)

    test = LazyLogDataset(TEST_CSV)
    ex = [test[i] for i in range(len(test))]
    prompts = [tok.encode(PROMPT_TEMPLATE.format(data=e["data"]), add_special_tokens=False)
               for e in ex]
    by_len = defaultdict(list)
    for i, pr in enumerate(prompts):
        by_len[len(pr)].append(i)

    preds = [None] * len(ex)
    pred_ids = [None] * len(ex)
    t0 = time.time()
    done_n = 0
    for plen in sorted(by_len):
        idxs = by_len[plen]
        for b0 in range(0, len(idxs), a.batch):
            bidx = idxs[b0:b0 + a.batch]
            ids = np.array([prompts[i] for i in bidx], dtype=np.int64)   # [B, plen], no padding
            outs = [[] for _ in bidx]
            done = [False] * len(bidx)
            for _ in range(a.max_new):
                win = ids[:, -max_seq:] if ids.shape[1] > max_seq else ids
                nxt = rt.next_token(win)                                  # [B,1] int64
                for j in range(len(bidx)):
                    if not done[j]:
                        t = int(nxt[j, 0])
                        if t == eos:
                            done[j] = True
                        else:
                            outs[j].append(t)
                ids = np.concatenate([ids, nxt], axis=1)
                if all(done):
                    break
            for j, i in enumerate(bidx):
                preds[i] = tok.decode(outs[j], skip_special_tokens=True).strip()
                pred_ids[i] = list(map(int, outs[j]))
            done_n += len(bidx)
        print(f"  len={plen:4d}  done {done_n}/{len(ex)}  ({time.time()-t0:.0f}s)", flush=True)

    labels = [e["label"].strip() for e in ex]
    exact = sum(int(x == y) for x, y in zip(preds, labels))
    em = exact / len(ex)
    dt = time.time() - t0
    mode = "preload" if a.preload_weights else "stream"
    print(f"[ONNX {a.provider} {mode}] exact-match={em:.4f} (n={len(ex)})  {dt:.0f}s", flush=True)

    out = a.out_dir / f"onnx_accuracy_{a.provider}_{mode}.json"
    json.dump({"run": "onnx_inference_accuracy_fresh", "onnx_dir": str(a.onnx_dir),
               "provider": a.provider, "mode": mode, "preset": a.preset,
               "n_test": len(ex), "exact_match": em, "eval_time_s": dt,
               "note": "equal-length batches (no padding); causal baked into export"},
              open(out, "w"), indent=2)
    json.dump({"engine": f"onnx_{a.provider}_{mode}", "preds": preds,
               "pred_ids": pred_ids},
              open(a.out_dir / f"preds_onnx_{a.provider}_{mode}.json", "w"))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
