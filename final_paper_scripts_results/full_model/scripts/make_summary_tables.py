#!/usr/bin/env python
"""Generate the summary tables into full_model/tables/ (Markdown + LaTeX):

  config_table        TBL-1  small vs large model config
  dataset_table       TBL-2  train/val/test split sizes
  a1_accuracy_table   TBL-3  A1 accuracy training (loss + token acc, CPU/GPU)

Reads outputs/*/metrics.json (no torch). Run aggregate_results.py first or not —
this reads the per-run metrics directly.
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


def write(name, md, tex):
    (TAB / f"{name}.md").write_text(md)
    (TAB / f"{name}.tex").write_text(tex)
    print(f"wrote tables/{name}.md and .tex")


# ---- TBL-1 config (small vs large) ----
small = load(OUT / "train_gpu" / "metrics.json")          # has small model_config
large = load(OUT / "cost_profiler_gpu" / "cost_profiler_metrics.json")  # large model_config
sc = small["model_config"] if small else {}
lc = large["model_config"] if large else {}
rows = [("tokenizer / vocab", "BPE-2k / 2,000", "GPT-2 / 50,257"),
        ("max_seq_len", sc.get("max_seq_len"), lc.get("max_seq_len")),
        ("n_layers", sc.get("n_layers"), lc.get("n_layers")),
        ("d_model", sc.get("d_model"), lc.get("d_model")),
        ("n_heads", sc.get("n_heads"), lc.get("n_heads")),
        ("d_ff", sc.get("d_ff"), lc.get("d_ff")),
        ("dropout", sc.get("dropout"), lc.get("dropout")),
        ("params", f"{small['params_million']:.2f}M" if small else "~0.46M",
                   f"{large['params_million']:.0f}M" if large else "~838M"),
        ("used for", "A1 accuracy, B1 inference", "A2/B2/C cost")]
md = ["### TBL-1 — Model configuration (small vs large)\n",
      "| Parameter | Small (accuracy) | Large (cost) |", "|---|---|---|"]
for a, b, c in rows:
    md.append(f"| {a} | {b} | {c} |")
tex = [r"\begin{table}[t]\centering\caption{Model configurations.}",
       r"\begin{tabular}{lll}\hline", r"Parameter & Small (accuracy) & Large (cost) \\\hline"]
tex += [f"{a} & {b} & {c} \\\\" for a, b, c in rows]
tex += [r"\hline\end{tabular}\end{table}"]
write("config_table", "\n".join(md) + "\n", "\n".join(tex) + "\n")

# ---- TBL-2 dataset ----
splits = [("train", "train.csv", 32568), ("validation", "validation.csv", 11520),
          ("test", "test.csv", 11654)]
md = ["### TBL-2 — Dataset splits (log-line -> Issue/level label)\n",
      "| Split | File | Examples |", "|---|---|---:|"]
for a, b, c in splits:
    md.append(f"| {a} | {b} | {c:,} |")
md.append("\n154 distinct labels; train is class-balanced; rare issue/level combos excluded.")
tex = [r"\begin{table}[t]\centering\caption{Dataset splits.}",
       r"\begin{tabular}{llr}\hline", r"Split & File & Examples \\\hline"]
tex += [f"{a} & {b} & {c:,} \\\\" for a, b, c in splits]
tex += [r"\hline\end{tabular}\end{table}"]
write("dataset_table", "\n".join(md) + "\n", "\n".join(tex) + "\n")

# ---- TBL-3 A1 accuracy ----
rows = []
for dev in ("GPU", "CPU"):
    m = load(OUT / f"train_{dev.lower()}" / "metrics.json")
    if not m:
        continue
    ed = m["epochs_detail"][-1]
    rows.append((dev, m["epochs"], ed["train_loss"], ed["val_loss"],
                 m["best_val_loss"], m["test_loss"], m["test_acc"], m["test_acc_example"]))
md = ["### TBL-3 — A1 accuracy training (small model, teacher-forced token acc)\n",
      "| Device | Epochs | Final train loss | Final val loss | Best val loss | Test loss | Test acc (global) | Test acc (example) |",
      "|---|---:|---:|---:|---:|---:|---:|---:|"]
for d, e, tl, vl, bv, tsl, ta, tae in rows:
    md.append(f"| {d} | {e} | {tl:.4f} | {vl:.4f} | {bv:.4f} | {tsl:.4f} | {ta:.4f} | {tae:.4f} |")
md.append("\n(Generation exact-match is in the B1 table — the honest deployment metric.)")
tex = [r"\begin{table}[t]\centering\caption{A1 accuracy training (small model).}",
       r"\begin{tabular}{lrrrrrrr}\hline",
       r"Device & Epochs & Train loss & Val loss & Best val & Test loss & Test acc (glob) & Test acc (ex) \\\hline"]
for d, e, tl, vl, bv, tsl, ta, tae in rows:
    tex.append(f"{d} & {e} & {tl:.4f} & {vl:.4f} & {bv:.4f} & {tsl:.4f} & {ta:.4f} & {tae:.4f} \\\\")
tex += [r"\hline\end{tabular}\end{table}"]
write("a1_accuracy_table", "\n".join(md) + "\n", "\n".join(tex) + "\n")
