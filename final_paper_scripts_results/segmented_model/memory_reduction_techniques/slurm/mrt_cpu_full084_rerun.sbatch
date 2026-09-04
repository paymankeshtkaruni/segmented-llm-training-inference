#!/bin/bash
#SBATCH --job-name=mrt_cpu_full084_rerun
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=02:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_cpu_full084_rerun_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_cpu_full084_rerun_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
# glibc heap tuning — IDENTICAL to mrt_cpu_scale_3b.sbatch (the 68 s / 59.4 GB cell)
# so the 0.84B CPU full-model cell becomes comparable down the tab:scale column.
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072
PYTHON="${PYTHON:-python}"
MRT=final_paper_scripts_results/segmented_model/memory_reduction_techniques
echo "Node: $(hostname)  Start: $(date)  OMP_NUM_THREADS=$OMP_NUM_THREADS"
# Re-measurement of the 0.84B CPU FULL-model training step (tab:scale row 1, CPU
# full column). The original 113.5 s came from full_model/slurm/cost.sbatch with
# --cpus-per-task=8 and OMP_NUM_THREADS unset (oversubscribed); this rerun uses
# the exact environment of the 3.09B/6.86B scale cells. Output goes to the new
# results/scale_cpu_large/ directory; nothing existing is overwritten.
"$PYTHON" "$MRT/scale_cost.py" --preset large_8x2x2x8 --device cpu --run full_train \
    --batch 4 --n-steps 3
echo "Done: $(date)"
