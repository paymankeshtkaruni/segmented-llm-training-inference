# Full-Model Baseline

The full (non-segmented) GPTTransformer baseline the paper compares the segmented
sequential approach against. Five experiments, each CPU + GPU.

## Two models (on purpose)
| | Small (accuracy) | Large (cost) |
|---|---|---|
| used by | A1 train, B1 inference | A2/B2/C cost |
| tokenizer | BPE-2k (`../../bpe_tokenizer/`) | GPT-2 (`../../gpt2_tokenizer/`) |
| size | ~0.46M | ~838M |

Small → the task has accuracy headroom (it's an easy 154-label task). Large → the
model's own memory dominates the fixed runtime overhead, so cost is measurable.

## Experiments
| ID | What | Model | sbatch |
|----|------|-------|--------|
| A1 | accuracy training (loss, val acc) | small | `slurm/train_{cpu,gpu}.sbatch` |
| A2 | training cost (memory + timing) | large | `slurm/cost.sbatch` |
| B1 | real inference (exact-match acc) | small | `slurm/infer_{cpu,gpu}.sbatch` |
| B2 | inference cost (torch) | large | `slurm/infer_cost.sbatch` |
| C  | inference cost (torch-free ONNX) | large | `slurm/export_onnx.sbatch` then `slurm/onnx_infer_cost.sbatch` |

## How to run

All jobs run on a **GPU node** (`grete:shared`, `--gres=gpu:A100:1`) — even the
CPU-compute variants (the A100 is only for the shared parallel filesystem access). Never the
login node except tiny checks. Interpreter = the poetry venv (torch 2.7); env:
`module load gcc/13.2.0 cuda/12.8.2`, `PYTHONPATH=src`, `HF_HUB_OFFLINE=1`.

```bash
cd <repo root>

# A1 — accuracy training (produces checkpoints used by B1)
sbatch final_paper_scripts_results/full_model/slurm/train_gpu.sbatch
sbatch final_paper_scripts_results/full_model/slurm/train_cpu.sbatch

# B1 — real inference (loads the matching A1 checkpoint)
sbatch final_paper_scripts_results/full_model/slurm/infer_gpu.sbatch
sbatch final_paper_scripts_results/full_model/slurm/infer_cpu.sbatch

# A2 — training cost (all 4 variants: profiler+light, GPU+CPU)
sbatch final_paper_scripts_results/full_model/slurm/cost.sbatch

# B2 — inference cost (torch, all 4 variants)
sbatch final_paper_scripts_results/full_model/slurm/infer_cost.sbatch

# C — ONNX inference cost: export ONCE (torch), then measure TORCH-FREE
sbatch final_paper_scripts_results/full_model/slurm/export_onnx.sbatch        # build large_model.onnx
sbatch final_paper_scripts_results/full_model/slurm/onnx_infer_cost.sbatch    # torch-free measure
```

Smoke versions (`slurm/smoke_*.sbatch`) are tiny debug runs — verify the sbatch
path before the real jobs.

## Regenerate figures & tables (login-node OK — small, read-only)
```bash
PY=<venv python>
$PY .../scripts/aggregate_results.py      # outputs/* -> jsons/full_model_summary.json
$PY .../scripts/plot_train_curves.py      # A1 learning curves
$PY .../scripts/plot_a2_cost.py           # A2 training-cost figures
$PY .../scripts/plot_b2_cost.py           # B2 inference-cost figures
$PY .../scripts/plot_c_onnx.py            # C ONNX-cost figures
$PY .../scripts/make_b1_table.py          # B1 accuracy table
$PY .../scripts/make_cost_table.py        # A2/B2/C cost comparison
$PY .../scripts/make_summary_tables.py    # config / dataset / A1 tables
```

## Layout
```
scripts/   _common.py (configs+builders), _train_lib, _cost_lib, _infer_lib,
           _onnx_cost_lib (torch-free), full_model_*.py entrypoints, plot_*/make_*
slurm/     one sbatch per run (+ smoke)
outputs/   per-run metrics.json / trace.csv / checkpoints / predictions / logs
figures/   train_curves_*, a2_*, b2_*, c_*  (.png)
tables/    config / dataset / a1_accuracy / b1_inference / cost_comparison (.md + .tex)
jsons/     full_model_summary.json
onnx/      large_model.onnx (+ weights)  — build artifact, gitignored
temp/      plot/table prototyping sandbox (not paper artifacts)
```

## Key results
- A1 test token-acc ~0.987; **B1 generation exact-match ~0.934** (the honest metric).
- A2 training cost (large): GPU VRAM ~25 GB, CPU host ~33 GB.
- B2 inference cost (torch): GPU VRAM ~5.3 GB. C (ONNX, torch-free): ~4.6 GB,
  half the host framework, 4× faster CPU — modestly leaner (weights dominate; the
  big reductions people cite come from quantization, which applies to torch too).
