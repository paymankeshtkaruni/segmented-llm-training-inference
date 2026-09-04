#!/usr/bin/env python
"""Full-model exact-match under the REPAIRED (equal-length, no-padding) protocol.

The fresh matched-pair full-model run (seg_full_train_b1.py) reported its
exact-match under the old left-padded B1 loop, which is not comparable to the
repaired equal-length numbers of the segmented/exported/ONNX engines
(0.9548). This script loads the saved full-model checkpoint and evaluates it
with the SAME batched_greedy_eval used for every other engine, then reports
agreement against the segmented predictions (different weights, so
disagreements are expected; the comparable number is the exact-match).

Writes results/quality_fresh_pair/full_model_repaired.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent                  # memory_reduction_techniques/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "segmentation_management"))

from config import get_preset                            # noqa: E402
from modules import ReferenceGPTDecoder                  # noqa: E402
from data import build_tokenizer                         # noqa: E402
from eval_quality_inference import batched_greedy_eval   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="full-model state_dict saved by seg_full_train_b1.py --save-ckpt")
    ap.add_argument("--preset", default="small_8x2x2x8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out-dir", type=Path,
                    default=HERE / "results" / "quality_fresh_pair")
    a = ap.parse_args()

    p = get_preset(a.preset)
    m = p["model"]
    tok = build_tokenizer(p["tokenizer"])
    model = ReferenceGPTDecoder(m).to(a.device)
    model.load_state_dict(torch.load(a.checkpoint, map_location=a.device))
    model.eval()

    @torch.no_grad()
    def full_next(ids, pad):
        logits, _ = model(ids, pad_token_id=pad)
        return logits[:, -1].argmax(-1)

    t0 = time.perf_counter()
    preds, pred_ids, labels, em, n = batched_greedy_eval(
        full_next, tok, m.max_seq_len, a.device, batch=a.batch)
    t = time.perf_counter() - t0
    print(f"[full/repaired] exact-match={em:.4f} (n={n})  {t:.0f}s", flush=True)

    seg = json.load(open(a.out_dir / "preds_torch.json"))["preds"]
    agree = sum(int(x == y) for x, y in zip(preds, seg))
    out = {
        "run": "full_model_repaired_eval",
        "checkpoint": str(a.checkpoint),
        "protocol": "equal-length batches, no padding (identical to all other engines)",
        "n_test": n,
        "exact_match": em,
        "agreement_vs_segmented_preds": agree / n,
        "note": "full model trained independently of the segmented checkpoint; "
                "agreement below 1.0 is expected (different weights), the "
                "comparable number is exact_match",
        "eval_time_s": round(t, 1),
    }
    dst = a.out_dir / "full_model_repaired.json"
    json.dump(out, open(dst, "w"), indent=2)
    print(json.dumps(out, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
