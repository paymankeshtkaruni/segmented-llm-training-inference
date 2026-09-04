#!/bin/bash
#SBATCH --job-name=mrt_gpu_scale_fit
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=04:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_gpu_scale_fit_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_gpu_scale_fit_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
MRT=final_paper_scripts_results/segmented_model/memory_reduction_techniques
echo "Node: $(hostname)  Start: $(date)"
# Intermediate cost-model fit points (~1.6B and ~5.1B), GPU training only.
# 1.6B full_train is expected to sit right at the 40 GB edge (est. ~39 GB) — fits
# or OOMs, both informative; 5.1B full_train OOMs (states ~82 GB; recorded).
for PRESET in xl15b_8x2x2x8 xl5b_8x2x2x8; do
  for RUN in full_train seg_train; do
    echo "===== $PRESET $RUN  $(date) ====="
    "$PYTHON" "$MRT/scale_cost.py" --preset "$PRESET" --device cuda --run "$RUN" \
        --batch 4 --n-steps 3
  done
done
echo "Done: $(date)"
