"""
B1 — segmented REAL inference accuracy on the FULL test set. Loads the trained A1
checkpoint into the segmented engine and runs BATCHED segmented greedy generation
(one segment resident at a time; left-padded variable-length prompts via the pad mask),
reporting honest exact-match label accuracy + issue-only / level-only (like full_model B1,
0.934). Segmented greedy == reference greedy is already proven; this is the real-data number.

The A1 checkpoint predates the attn_out_proj segmentation, so its W_o lives in `shared`;
we load it into the attn_out_proj segment store for the current engine.
"""
from __future__ import annotations
import argparse, csv, json, re, sys, time
from pathlib import Path

import torch

SEG_PKG = Path(__file__).resolve().parent.parent / "segmentation_management"
sys.path.insert(0, str(SEG_PKG))
from trainer import SegmentedTrainer                 # noqa: E402
from optimizer import all_segment_keys               # noqa: E402
from inference import SegmentedGenerator             # noqa: E402
from data import LazyLogDataset, TEST_CSV            # noqa: E402
from config import PROMPT_TEMPLATE                    # noqa: E402


def parse_il(label):
    m = re.match(r"\s*[Ii]ssue:\s*(.*?),\s*level:\s*(.*)", label.strip())
    return (m.group(1).strip(), m.group(2).strip()) if m else (None, None)


def load_checkpoint(tr, ckpt, device):
    # current engine: every weight (incl attn_out_proj) is a segment in `params`; the
    # resident `shared` is only norms + mlp_out_bias + final_norm.
    for k in all_segment_keys(tr.m, tr.s):
        tr.param_store.put(k, ckpt["params"][f"{k.layer_id}|{k.kind}|{k.seg}"])
    tr.shared.load_state_dict({kk: v.to(device) for kk, v in ckpt["shared"].items()})
    tr.fwd.eval()


@torch.no_grad()
def generate_batch(gen, prompts, max_new_tokens, pad_id, eos_id, device):
    """Batched greedy over LEFT-padded variable-length prompts; one segment at a time."""
    maxlen = max(len(p) for p in prompts); B = len(prompts)
    ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, maxlen - len(p):] = torch.tensor(p, dtype=torch.long)
    ids = ids.to(device)
    outs = [[] for _ in prompts]; done = [False] * B
    for _ in range(max_new_tokens):
        nxt = gen._next_token(ids, pad_token_id=pad_id)        # [B,1]
        for i in range(B):
            if not done[i]:
                t = int(nxt[i, 0])
                if t == eos_id: done[i] = True
                else: outs[i].append(t)
        ids = torch.cat([ids, nxt], dim=1)
        if all(done): break
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-test", type=int, default=-1)        # -1 = full test set
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(a.checkpoint, map_location=a.device)
    preset = ckpt["preset"]
    tr = SegmentedTrainer(preset, a.device, a.out_dir / "_work")
    load_checkpoint(tr, ckpt, a.device)
    gen = SegmentedGenerator(tr.fwd, tr.tok)
    tok = tr.tok; pad_id = tok.pad_token_id; eos_id = tok.eos_token_id

    n_rows = None if a.num_test < 0 else a.num_test
    test = LazyLogDataset(TEST_CSV, max_rows=n_rows)
    ex = [test[i] for i in range(len(test))]
    prompts = [tok.encode(PROMPT_TEMPLATE.format(data=e["data"]), add_special_tokens=False) for e in ex]
    order = sorted(range(len(ex)), key=lambda i: len(prompts[i]))   # group by length -> less padding

    n = exact = issue_ok = level_ok = 0; rows = []
    t0 = time.perf_counter()
    for b0 in range(0, len(order), a.batch_size):
        bidx = order[b0:b0 + a.batch_size]
        outs = generate_batch(gen, [prompts[i] for i in bidx], a.max_new_tokens, pad_id, eos_id, a.device)
        for j, i in enumerate(bidx):
            pred = tok.decode(outs[j], skip_special_tokens=True).strip()
            true = ex[i]["label"].strip()
            ok = pred == true; pi, pl = parse_il(pred); ti, tl = parse_il(true)
            n += 1; exact += int(ok); issue_ok += int(pi is not None and pi == ti); level_ok += int(pl is not None and pl == tl)
            rows.append({"data": ex[i]["data"][:80], "true": true, "pred": pred, "correct": ok})
        print(f"  {n}/{len(ex)}  exact={exact/n:.4f} issue={issue_ok/n:.4f} level={level_ok/n:.4f}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    dt = time.perf_counter() - t0

    metrics = {"run": "segmented_infer_accuracy", "preset": preset, "device": a.device,
               "n_test": n, "batch_size": a.batch_size, "max_new_tokens": a.max_new_tokens,
               "exact_match": exact / n, "issue_acc": issue_ok / n, "level_acc": level_ok / n,
               "elapsed_s": round(dt, 1)}
    json.dump(metrics, open(a.out_dir / "infer_metrics.json", "w"), indent=2)
    with open(a.out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["data", "true", "pred", "correct"]); w.writeheader(); w.writerows(rows)
    print(f"[B1] {preset} on {a.device}  n={n}")
    print(f"  EXACT-MATCH={metrics['exact_match']:.4f}  issue={metrics['issue_acc']:.4f}  level={metrics['level_acc']:.4f}  ({dt:.0f}s)")
    print(f"  -> {a.out_dir/'infer_metrics.json'}")


if __name__ == "__main__":
    main()
