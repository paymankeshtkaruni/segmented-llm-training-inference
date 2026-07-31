"""Temp (for review): build the B1 real-inference results table.

Reads outputs/infer_{gpu,cpu}/infer_metrics.json (and A1 train metrics for the
teacher-forced token-accuracy comparison) and writes Markdown + LaTeX.

Once confirmed, graduates into scripts/ writing to tables/.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FULL_MODEL = HERE.parent
OUT = FULL_MODEL / "outputs"


def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


gpu = load(OUT / "infer_gpu" / "infer_metrics.json")
cpu = load(OUT / "infer_cpu" / "infer_metrics.json")
a1_gpu = load(OUT / "train_gpu" / "metrics.json")
a1_cpu = load(OUT / "train_cpu" / "metrics.json")

rows = []
for dev, m, a1 in [("GPU", gpu, a1_gpu), ("CPU", cpu, a1_cpu)]:
    if m is None:
        continue
    rows.append({
        "device": dev,
        "exact": m["exact_match_accuracy"],
        "issue": m["issue_accuracy"],
        "level": m["level_accuracy"],
        "tf_token": a1["test_acc"] if a1 else float("nan"),  # teacher-forced token acc (A1 test)
        "eps": m["examples_per_s"],
        "elapsed": m["elapsed_s"],
        "trunc": m.get("prompts_truncated", 0),
        "n": m["n_examples"],
    })

# ---- Markdown ----
md = []
md.append("### B1 — Real inference (small model, generation over full test set)\n")
md.append("Honest generation accuracy (whole-label exact match) vs the "
          "teacher-forced token accuracy from A1.\n")
md.append("| Device | Exact-match | Issue acc | Level acc | Teacher-forced token acc (A1) | Examples/s | Time (s) | Prompts truncated |")
md.append("|--------|------------:|----------:|----------:|------------------------------:|-----------:|---------:|------------------:|")
for r in rows:
    md.append(f"| {r['device']} | {r['exact']:.4f} | {r['issue']:.4f} | {r['level']:.4f} | "
              f"{r['tf_token']:.4f} | {r['eps']:.0f} | {r['elapsed']:.1f} | {r['trunc']}/{r['n']} |")
md_text = "\n".join(md) + "\n"

# ---- LaTeX ----
tex = []
tex.append(r"\begin{table}[t]\centering")
tex.append(r"\caption{B1 real inference (small model): generation exact-match accuracy vs.\ "
           r"teacher-forced token accuracy, with throughput.}")
tex.append(r"\begin{tabular}{lrrrrrr}")
tex.append(r"\hline")
tex.append(r"Device & Exact-match & Issue & Level & TF token acc & Examples/s & Time (s) \\")
tex.append(r"\hline")
for r in rows:
    tex.append(f"{r['device']} & {r['exact']:.4f} & {r['issue']:.4f} & {r['level']:.4f} & "
               f"{r['tf_token']:.4f} & {r['eps']:.0f} & {r['elapsed']:.1f} \\\\")
tex.append(r"\hline")
tex.append(r"\end{tabular}")
tex.append(r"\end{table}")
tex_text = "\n".join(tex) + "\n"

(HERE / "b1_inference_table.md").write_text(md_text)
(HERE / "b1_inference_table.tex").write_text(tex_text)
print(md_text)
print("wrote b1_inference_table.md and .tex")
