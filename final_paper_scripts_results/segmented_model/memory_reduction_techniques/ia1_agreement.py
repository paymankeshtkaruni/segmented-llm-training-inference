#!/usr/bin/env python
"""I-A1 — ONNX vs torch prediction agreement + near-tie margin analysis.

Loads the prediction dumps of the torch segmented engine and one ONNX config
(all four ONNX configs produce identical predictions), reports:
  - agreement over all test examples,
  - for every disagreement: the first divergent generation step, and the torch
    logit margin between torch's token and ONNX's token at that step (computed
    by teacher-forcing the shared prefix through the exported model).
A small margin (fp-noise scale, relative to the logit range) proves the
disagreements are near-ties, not model differences.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "segmentation_management"))

from config import get_preset, PROMPT_TEMPLATE                 # noqa: E402
from data import build_tokenizer, LazyLogDataset, TEST_CSV     # noqa: E402
from export import reassemble_model                            # noqa: E402
from stores import make_store, SegmentKey                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch-preds", type=Path, required=True)
    ap.add_argument("--onnx-preds", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results" / "quality_fresh_pair")
    a = ap.parse_args()

    tp = json.load(open(a.torch_preds))
    op = json.load(open(a.onnx_preds))
    n = len(tp["preds"])
    agree_text = sum(int(x == y) for x, y in zip(tp["preds"], op["preds"]))
    agree_ids = sum(int(x == y) for x, y in zip(tp["pred_ids"], op["pred_ids"]))
    print(f"[agreement] text: {agree_text}/{n} ({agree_text/n:.6f})  "
          f"token-ids: {agree_ids}/{n} ({agree_ids/n:.6f})", flush=True)

    # ---- margin analysis on token-level disagreements ---------------------------
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    preset = get_preset(ck["preset"])
    m, s = preset["model"], preset["seg"]
    store = make_store("cpu_ram")
    for kstr, sd in ck["params"].items():
        layer, kind, seg = kstr.split("|")
        store.put(SegmentKey(int(layer), kind, int(seg)), sd)
    model = reassemble_model(store, ck["shared"], m, s).to(a.device).eval()
    tok = build_tokenizer(preset["tokenizer"])
    test = LazyLogDataset(TEST_CSV)

    rows = []
    for i in range(n):
        ti, oi = tp["pred_ids"][i], op["pred_ids"][i]
        if ti == oi:
            continue
        k = next((j for j in range(min(len(ti), len(oi))) if ti[j] != oi[j]),
                 min(len(ti), len(oi)))
        prompt = tok.encode(PROMPT_TEMPLATE.format(data=test[i]["data"]),
                            add_special_tokens=False)
        prefix = prompt + ti[:k]
        ids = torch.tensor([prefix[-m.max_seq_len:]], dtype=torch.long, device=a.device)
        with torch.no_grad():
            logits, _ = model(ids, pad_token_id=None)
        lg = logits[0, -1]
        top2 = torch.topk(lg, 2)
        t_tok = ti[k] if k < len(ti) else int(tok.eos_token_id)
        o_tok = oi[k] if k < len(oi) else int(tok.eos_token_id)
        margin = float(lg[t_tok] - lg[o_tok])
        rows.append({"example": i, "divergence_step": k,
                     "torch_token": t_tok, "onnx_token": o_tok,
                     "logit_margin_torch_minus_onnx": margin,
                     "top1_top2_gap": float(top2.values[0] - top2.values[1]),
                     "logit_scale_max_abs": float(lg.abs().max()),
                     "torch_pred": tp["preds"][i], "onnx_pred": op["preds"][i],
                     "label": tp["labels"][i]})
    margins = [abs(r["logit_margin_torch_minus_onnx"]) for r in rows]
    print(f"[margins] {len(rows)} token-level disagreements; "
          f"max |margin| = {max(margins) if margins else 0:.3e}, "
          f"median = {sorted(margins)[len(margins)//2] if margins else 0:.3e}", flush=True)

    out = a.out_dir / "ia1_agreement_report.json"
    json.dump({"run": "ia1_onnx_torch_agreement", "n_test": n,
               "agreement_text": agree_text / n, "agreement_token_ids": agree_ids / n,
               "n_disagreements_text": n - agree_text,
               "n_disagreements_token_ids": n - agree_ids,
               "margin_rows": rows}, open(out, "w"), indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
