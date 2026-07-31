"""
Shared B1 real-inference logic (small model; CPU + GPU).

Loads the matching A1 checkpoint, runs BATCHED greedy generation over the full
test set, and reports the HONEST exact-match label accuracy (generated
`Issue: ..., level: ...` vs the true label), plus issue-only and level-only
accuracy. Device-matched: GPU checkpoint -> GPU, CPU checkpoint -> CPU.

Speed: accuracy is independent of batch size (greedy is deterministic), so we use
a LARGE batch (default 256) purely for throughput. Generation is batched with
RIGHT-padding + per-sequence position tracking (right-pad keeps position 0 a real
token, avoiding the fully-masked-row NaN).

Outputs (default outputs/infer_<dev>/):
  infer_metrics.json   exact_match / issue / level accuracy, timing, config
  predictions.csv      log_line, true_label, predicted_label, correct
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import torch

import _common as C


def parse_issue_level(label: str):
    m = re.match(r"\s*Issue:\s*(.*?),\s*level:\s*(.*)", label.strip())
    if not m:
        return None, None
    return m.group(1).strip(), m.group(2).strip()


@torch.no_grad()
def generate_batch(model, prompt_ids_list, pad_id, eos_id, max_new_tokens, device):
    """Batched greedy generation (right-padded, per-sequence position tracking)."""
    B = len(prompt_ids_list)
    prompt_lens = [len(p) for p in prompt_ids_list]
    Lmax = max(prompt_lens)
    total = Lmax + max_new_tokens
    buf = torch.full((B, total), pad_id, dtype=torch.long, device=device)
    for i, p in enumerate(prompt_ids_list):
        buf[i, :len(p)] = torch.tensor(p, dtype=torch.long, device=device)
    cur = torch.tensor(prompt_lens, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    ar = torch.arange(B, device=device)

    for _ in range(max_new_tokens):
        M = int(cur.max().item())
        logits, _ = model(buf[:, :M], pad_token_id=pad_id)
        next_logits = logits[ar, (cur - 1).clamp(min=0)]
        next_tok = next_logits.argmax(dim=-1)
        active = ~finished
        wpos = cur.clamp(max=total - 1)
        buf[ar[active], wpos[active]] = next_tok[active]
        cur = torch.where(active, cur + 1, cur)
        # a still-active sequence that just emitted EOS is now finished
        finished = finished | (active & (next_tok == eos_id))
        if bool(finished.all()):
            break

    out = []
    for i in range(B):
        gen = buf[i, prompt_lens[i]:cur[i]].tolist()
        if gen and gen[-1] == eos_id:
            gen = gen[:-1]
        out.append(gen)
    return out


def add_infer_args(p: argparse.ArgumentParser, default_device: str, default_out: Path,
                   default_ckpt: Path) -> None:
    p.add_argument("--device", default=default_device)
    p.add_argument("--checkpoint", type=Path, default=default_ckpt)
    p.add_argument("--batch-size", type=int, default=C.INFER_BATCH_SIZE)  # 256
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--num-examples", type=int, default=None, help="limit (smoke)")
    p.add_argument("--out-dir", type=Path, default=default_out)


def run_infer(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1] B1 real inference (small model) on {device}  bs={args.batch_size}  "
          f"max_new_tokens={args.max_new_tokens}")
    tokenizer = C.build_tokenizer()  # SMALL config -> BPE
    pad_id, eos_id = tokenizer.pad_token_id, tokenizer.eos_token_id
    model = C.build_model(tokenizer.vocab_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"    checkpoint={args.checkpoint} (epoch {ckpt.get('epoch','?')})  "
          f"params={C.count_params(model)/1e6:.2f}M")

    # read full test set
    rows = []
    with open(C.TEST_CSV, newline="") as f:
        for r in csv.DictReader(f):
            ll, lab = r["log_line"].strip(), r["label"].strip()
            if ll and lab:
                rows.append((ll, lab))
            if args.num_examples and len(rows) >= args.num_examples:
                break
    print(f"[2] test examples: {len(rows)}")

    # cap prompt length so prompt + generation never exceeds the model context
    prompt_max = C.MODEL_CONFIG["max_seq_len"] - args.max_new_tokens
    n_truncated = 0

    def encode_prompt(ll):
        nonlocal n_truncated
        ids = tokenizer.encode(C.PROMPT_TEMPLATE.format(data=ll), add_special_tokens=False)
        if len(ids) > prompt_max:
            ids = ids[-prompt_max:]   # keep the tail (includes "\nLabel:")
            n_truncated += 1
        return ids

    n = exact = issue_ok = level_ok = 0
    preds = []
    t0 = time.time()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        prompt_ids = [encode_prompt(ll) for ll, _ in batch]
        gen = generate_batch(model, prompt_ids, pad_id, eos_id, args.max_new_tokens, device)
        for (ll, true_lab), g in zip(batch, gen):
            pred = tokenizer.decode(g, skip_special_tokens=True).strip()
            n += 1
            ok = pred == true_lab
            exact += ok
            ti, tl = parse_issue_level(true_lab)
            pi, pl = parse_issue_level(pred)
            issue_ok += (pi == ti and ti is not None)
            level_ok += (pl == tl and tl is not None)
            preds.append((ll, true_lab, pred, int(ok)))
    elapsed = time.time() - t0

    metrics = {
        "run": "B1_real_inference",
        "device": str(device),
        "model": "small",
        "model_config": C.MODEL_CONFIG,
        "checkpoint": str(args.checkpoint),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "prompt_max_tokens": prompt_max,
        "prompts_truncated": n_truncated,
        "n_examples": n,
        "exact_match_accuracy": exact / n if n else float("nan"),
        "issue_accuracy": issue_ok / n if n else float("nan"),
        "level_accuracy": level_ok / n if n else float("nan"),
        "elapsed_s": elapsed,
        "examples_per_s": n / elapsed if elapsed else float("nan"),
    }
    with open(out_dir / "infer_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["log_line", "true_label", "predicted_label", "correct"])
        w.writerows(preds)

    print(f"[done] exact_match={metrics['exact_match_accuracy']:.4f}  "
          f"issue={metrics['issue_accuracy']:.4f}  level={metrics['level_accuracy']:.4f}  "
          f"| {n} ex in {elapsed:.1f}s = {metrics['examples_per_s']:.0f} ex/s  -> {out_dir}")
