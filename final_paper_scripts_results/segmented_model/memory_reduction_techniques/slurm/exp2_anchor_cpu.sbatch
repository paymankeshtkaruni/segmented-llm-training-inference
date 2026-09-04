#!/bin/bash
#SBATCH --job-name=exp2_anchor_cpu
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=04:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp2_anchor_cpu_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp2_anchor_cpu_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
# corrected-CPU-protocol glibc heap tuning (identical to the Jul-31 campaign)
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_ARENA_MAX=2
PY="${PYTHON:-python}"
echo "Node: $(hostname)  Start: $(date)  MALLOC_ARENA_MAX=$MALLOC_ARENA_MAX"
# Experiment 2 — naive-segmented anchor on CPU (disk store), every technique OFF.
for rep in 1 2 3; do
  echo "=== rep $rep ==="
  "$PY" measure_one.py --kind train --preset large_8x2x2x8 --device cpu \
      --tech-code 00000000000 --prefix naive_anchor \
      --out-dir results/exp2_anchor/cpu_rep$rep
done
echo "Done: $(date)"
