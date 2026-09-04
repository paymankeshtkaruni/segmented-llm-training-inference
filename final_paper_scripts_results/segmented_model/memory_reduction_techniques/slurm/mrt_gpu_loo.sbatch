#!/bin/bash
#SBATCH --job-name=mrt_gpu_loo
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=02:30:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_gpu_loo_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_gpu_loo_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY="${PYTHON:-python}"
echo "Node: $(hostname)  Start: $(date)"
"$PY" final_paper_scripts_results/segmented_model/memory_reduction_techniques/loo_cost.py \
    --preset large_8x2x2x8 --device cuda --kinds train,infer --batch 4 --n-steps 3 --prompt-len 256 --gen-tokens 8
echo "Done: $(date)"
