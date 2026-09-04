#!/bin/bash
#SBATCH --job-name=exp1_compare
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
# (site note: pin to one homogeneous node pool at your site;
#  the original runs excluded non-matching nodes here)
#SBATCH --time=06:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp1_compare_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp1_compare_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY="${PYTHON:-python}"
OUT=results/exp1_compare
mkdir -p "$OUT"
echo "Node: $(hostname)  Start: $(date)"

# Experiment 1 — GPU comparison suite (pinned A100-40):
# full model (dropout 0.1 and 0.0) x3, segmented coarse/fine x {T2, T3} x3.
# ZeRO-2/3 cells kept from the verified Jul-31 runs (same node class).
T2=11110111000   # base + recompute, resident, after-full
T3=11111111100   # base + recompute + streaming(+output head), after-full

for rep in 1 2 3; do
  echo "=== rep $rep: full model, dropout 0.1 ==="
  "$PY" scale_cost.py --preset large_8x2x2x8 --device cuda --run full_train \
      --out-dir "$OUT/full_drop_rep$rep"
  echo "=== rep $rep: full model, dropout 0.0 ==="
  "$PY" scale_cost.py --preset large_8x2x2x8 --device cuda --run full_train \
      --dropout 0.0 --out-dir "$OUT/full_nodrop_rep$rep"
  for part in 2x1x1x2 16x4x4x16; do
    echo "=== rep $rep: segmented $part T2 ==="
    "$PY" measure_one.py --kind train --preset large_8x2x2x8 --device cuda \
        --tech-code $T2 --seg-override $part --prefix "seg_${part}_T2" \
        --out-dir "$OUT/seg_${part}_T2_rep$rep"
    echo "=== rep $rep: segmented $part T3 ==="
    "$PY" measure_one.py --kind train --preset large_8x2x2x8 --device cuda \
        --tech-code $T3 --seg-override $part --prefix "seg_${part}_T3" \
        --out-dir "$OUT/seg_${part}_T3_rep$rep"
  done
done
echo "Done: $(date)"
