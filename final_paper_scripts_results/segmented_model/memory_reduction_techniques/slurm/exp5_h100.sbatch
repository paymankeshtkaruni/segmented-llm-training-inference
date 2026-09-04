#!/bin/bash
#SBATCH --job-name=exp5_h100
#SBATCH --partition=grete-h100:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=04:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp5_h100_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp5_h100_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2
PY="${PYTHON:-python}"
echo "Node: $(hostname)  Start: $(date)"
echo "=== xl3b_8x2x2x8 full_train rep1 ==="
"$PY" scale_cost.py --preset xl3b_8x2x2x8 --device cuda --run full_train --out-dir "results/exp5_scale/h100_xl3b_full_train_rep1" || echo "[recorded failure/OOM]"
echo "=== xxl7b_8x2x2x8 full_train rep1 ==="
"$PY" scale_cost.py --preset xxl7b_8x2x2x8 --device cuda --run full_train --out-dir "results/exp5_scale/h100_xxl7b_full_train_rep1" || echo "[recorded failure/OOM]"
echo "Done: $(date)"
