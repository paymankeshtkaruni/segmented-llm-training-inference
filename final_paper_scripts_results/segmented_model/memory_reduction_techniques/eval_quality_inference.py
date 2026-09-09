#!/usr/bin/env python
"""Fresh inference-accuracy table (compact quality config, fresh T1 checkpoint).

Rows:
  1. exported model — segmented checkpoint reassembled into a full model,
     batched greedy exact-match on the complete test set (b1 protocol);
  2. segmented engine — the same evaluation through the segmented inference
     engine (resident mode, batched greedy);
  3. per-example prediction agreement between the two (identical weights).

The full-model row of the paper table comes from the matched fresh training run
(seg_full_train_b1.py output). Metric and loop are IDENTICAL for both engines:
equal-length batches (no padding), greedy argmax, EOS stop, string exact-match on labels.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "segmentation_management"))

from techniques import Tech                                              # noqa: E402
from config import get_preset, PROMPT_TEMPLATE                           # noqa: E402
from data import build_tokenizer, LazyLogDataset, TEST_CSV               # noqa: E402
from forward_engine import SegmentedForwardEngine, SharedParams          # noqa: E402
from loader import StrictSegmentLoader                                   # noqa: E402
from stores import make_store, SegmentKey                                # noqa: E402
from export import reassemble_model                                      # noqa: E402
from inference import SegmentedGenerator                                 # noqa: E402

# resident, no streaming, no KV cache (batched recompute decode), base on
RESIDENT_CODE = "11100111001"


def tech_from_code(code: str) -> Tech:
    names = [f.name for f in fields(Tech)]
    return Tech(**{n: c == "1" for n, c in zip(names, code)})


def load_checkpoint(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    preset = get_preset(ck["preset"])
    m, s = preset["model"], preset["seg"]
    store = make_store("cpu_ram")
    for kstr, sd in ck["params"].items():
        layer, kind, seg = kstr.split("|")
        store.put(SegmentKey(int(layer), kind, int(seg)), sd)
    return ck, m, s, store, ck["shared"]


def batched_greedy_eval(next_token_batch, tok, max_seq_len, device, batch=64, max_new=40):
    """Equal-length batches (NO padding): batches contain only prompts of identical
    length, so every real token sits at its natural position. (The earlier b1-style
    left-padding shifted positions for padded rows and caused 70/11654 spurious
    disagreements vs the unpadded ONNX evaluation.)"""
    from collections import defaultdict
    pad, eos = tok.pad_token_id, tok.eos_token_id
    test = LazyLogDataset(TEST_CSV)
    ex = [test[i] for i in range(len(test))]
    prompts = [tok.encode(PROMPT_TEMPLATE.format(data=e["data"]), add_special_tokens=False)
               for e in ex]
    by_len = defaultdict(list)
    for i, pr in enumerate(prompts):
        by_len[len(pr)].append(i)
    groups = [idxs[b0:b0 + batch] for plen in sorted(by_len)
              for idxs in [by_len[plen]] for b0 in range(0, len(idxs), batch)]
    preds = [None] * len(ex)
    pred_ids = [None] * len(ex)
    for bidx in groups:
        bp = [prompts[i] for i in bidx]
        B = len(bp)
        ids = torch.tensor(bp, dtype=torch.long, device=device)
        outs = [[] for _ in bp]
        done = [False] * B
        for _ in range(max_new):
            if ids.size(1) > max_seq_len:
                ids = ids[:, -max_seq_len:]
            nxt = next_token_batch(ids, pad)                 # [B] LongTensor
            for j in range(B):
                if not done[j]:
                    t = int(nxt[j])
                    if t == eos:
                        done[j] = True
                    else:
                        outs[j].append(t)
            ids = torch.cat([ids, nxt.view(B, 1)], dim=1)
            if all(done):
                break
        for j, i in enumerate(bidx):
            preds[i] = tok.decode(outs[j], skip_special_tokens=True).strip()
            pred_ids[i] = list(outs[j])
    labels = [e["label"].strip() for e in ex]
    exact = sum(int(p == l) for p, l in zip(preds, labels))
    return preds, pred_ids, labels, exact / len(ex), len(ex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=HERE / "results" / "quality_fresh_pair" / "T1" / "checkpoints" / "best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results" / "quality_fresh_pair")
    a = ap.parse_args()

    ck, m, s, store, shared_state = load_checkpoint(a.checkpoint)
    tok = build_tokenizer(get_preset(ck["preset"])["tokenizer"])

    # ---- 1. exported (reassembled) model ---------------------------------------
    exported = reassemble_model(store, shared_state, m, s).to(a.device).eval()

    def full_next(ids, pad):
        with torch.no_grad():
            logits, _ = exported(ids, pad_token_id=pad)
        return logits[:, -1].argmax(-1)

    t0 = time.time()
    p_exp, ids_exp, labels, em_exp, n = batched_greedy_eval(full_next, tok, m.max_seq_len,
                                                            a.device, a.batch)
    t_exp = time.time() - t0
    print(f"[exported ] exact-match={em_exp:.4f} (n={n})  {t_exp:.0f}s", flush=True)

    # ---- 2. segmented engine ----------------------------------------------------
    shared = SharedParams(m)
    shared.load_state_dict(shared_state)
    shared = shared.to(a.device)
    loader = StrictSegmentLoader(m, s, store, a.device, tech=tech_from_code(RESIDENT_CODE))
    fwd = SegmentedForwardEngine(m, s, loader, shared, a.device).eval()
    loader.training = False
    gen = SegmentedGenerator(fwd, tok)

    def seg_next(ids, pad):
        with torch.no_grad():
            hidden = fwd.forward_hidden(ids, pad_token_id=pad)
            return gen._argmax_last(hidden[:, -1:, :]).view(-1)

    t0 = time.time()
    p_seg, ids_seg, _, em_seg, _ = batched_greedy_eval(seg_next, tok, m.max_seq_len,
                                                       a.device, a.batch)
    t_seg = time.time() - t0
    print(f"[segmented] exact-match={em_seg:.4f} (n={n})  {t_seg:.0f}s", flush=True)
    json.dump({"engine": "torch", "preds": p_seg, "pred_ids": ids_seg,
               "preds_exported": p_exp, "pred_ids_exported": ids_exp,
               "labels": labels},
              open(a.out_dir / "preds_torch.json", "w"))

    # ---- 3. agreement -----------------------------------------------------------
    agree = sum(int(x == y) for x, y in zip(p_exp, p_seg))
    print(f"[agreement] segmented == exported on {agree}/{n} examples "
          f"({agree / n:.6f})", flush=True)
    diffs = [{"i": i, "exported": x, "segmented": y, "label": l}
             for i, (x, y, l) in enumerate(zip(p_exp, p_seg, labels)) if x != y][:20]

    out = a.out_dir / "inference_accuracy_fresh.json"
    json.dump({"run": "quality_inference_accuracy_fresh",
               "checkpoint": str(a.checkpoint), "preset": ck["preset"],
               "segmented_mode": "resident, batched greedy (no cache)",
               "n_test": n,
               "exported_exact_match": em_exp, "segmented_exact_match": em_seg,
               "prediction_agreement": agree / n, "n_disagreements": n - agree,
               "first_disagreements": diffs,
               "eval_time_s": {"exported": t_exp, "segmented": t_seg}},
              open(out, "w"), indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
