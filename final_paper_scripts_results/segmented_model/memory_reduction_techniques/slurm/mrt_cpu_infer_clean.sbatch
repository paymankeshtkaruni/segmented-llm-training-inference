#!/bin/bash
#SBATCH --job-name=mrt_cpu_infer_clean
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=01:30:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_cpu_infer_clean_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_cpu_infer_clean_%j.err
# Site-independent: export PROJECT_ROOT=<path to this repository> and, if the
# interpreter is not on PATH, PYTHON=<interpreter> before submitting. Submit from
# the repository root so the relative --output/--error log paths resolve.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2
PY="${PYTHON:-python}"
echo "Node: $(hostname)  Start: $(date)"
# clean segmented torch CPU inference (from_scratch -> no build spike), seg_c/B2 protocol
"$PY" final_paper_scripts_results/segmented_model/memory_reduction_techniques/seg_infer_cost_lib.py \
    --preset large_8x2x2x8 --device cpu --batch 4 --seq 512 --gen-tokens 4 --prompt-len 64 \
    --out-dir final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/seg_b2_cpu_clean --prefix seg_b2_clean
echo "Done: $(date)"
