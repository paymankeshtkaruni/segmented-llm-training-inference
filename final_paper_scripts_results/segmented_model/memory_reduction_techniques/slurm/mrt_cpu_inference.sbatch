#!/bin/bash
#SBATCH --job-name=mrt_cpu_infer
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=03:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_cpu_infer_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_cpu_infer_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_ARENA_MAX=2
PYTHON="${PYTHON:-python}"
echo "Node: $(hostname)  Start: $(date)  MALLOC_ARENA_MAX=$MALLOC_ARENA_MAX"
"$PYTHON" final_paper_scripts_results/segmented_model/memory_reduction_techniques/cpu_inference_cost.py \
    --preset large_8x2x2x8 --device cpu --prompt-len 256 --gen-tokens 8
echo "Done: $(date)"
