#!/usr/bin/env python
"""Generate the B1 real-inference results table into full_model/tables/.

Minimal table: device x {exact-match, issue, level, teacher-forced token acc (A1),
throughput, time}. Writes Markdown + LaTeX.

Promoted from temp/test_b1_tables.py after review.
"""
import json
from pathlib import Path

FULL_MODEL = Path(__file__).resolve().parent.parent
OUT = FULL_MODEL / "outputs"
TAB = FULL_MODEL / "tables"
TAB.mkdir(parents=True, exist_ok=True)


def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


rows = []
for dev in ("GPU", "CPU"):
    m = load(OUT / f"infer_{dev.lower()}" / "infer_metrics.json")
    a1 = load(OUT / f"train_{dev.lower()}" / "metrics.json")
    if m is None:
        continue
    rows.append({
        "device": dev, "exact": m["exact_match_accuracy"], "issue": m["issue_accuracy"],
        "level": m["level_accuracy"], "tf_token": a1["test_acc"] if a1 else float("nan"),
        "eps": m["examples_per_s"], "elapsed": m["elapsed_s"],
        "trunc": m.get("prompts_truncated", 0), "n": m["n_examples"],
    })

md = ["### B1 — Real inference (small model, generation over full test set)\n",
      "Honest generation accuracy (whole-label exact match) vs the teacher-forced "
      "token accuracy from A1.\n",
      "| Device | Exact-match | Issue acc | Level acc | Teacher-forced token acc (A1) | Examples/s | Time (s) | Prompts truncated |",
      "|--------|------------:|----------:|----------:|------------------------------:|-----------:|---------:|------------------:|"]
for r in rows:
    md.append(f"| {r['device']} | {r['exact']:.4f} | {r['issue']:.4f} | {r['level']:.4f} | "
              f"{r['tf_token']:.4f} | {r['eps']:.0f} | {r['elapsed']:.1f} | {r['trunc']}/{r['n']} |")
(TAB / "b1_inference_table.md").write_text("\n".join(md) + "\n")

tex = [r"\begin{table}[t]\centering",
       r"\caption{B1 real inference (small model): generation exact-match accuracy vs.\ "
       r"teacher-forced token accuracy, with throughput.}",
       r"\begin{tabular}{lrrrrrr}", r"\hline",
       r"Device & Exact-match & Issue & Level & TF token acc & Examples/s & Time (s) \\", r"\hline"]
for r in rows:
    tex.append(f"{r['device']} & {r['exact']:.4f} & {r['issue']:.4f} & {r['level']:.4f} & "
               f"{r['tf_token']:.4f} & {r['eps']:.0f} & {r['elapsed']:.1f} \\\\")
tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
(TAB / "b1_inference_table.tex").write_text("\n".join(tex) + "\n")

print(f"wrote {TAB}/b1_inference_table.md and .tex")
