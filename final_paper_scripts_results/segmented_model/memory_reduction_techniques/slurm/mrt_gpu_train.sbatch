#!/bin/bash
#SBATCH --job-name=mrt_gpu_train
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=02:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_gpu_train_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_gpu_train_%j.err
# Site-independent: export PROJECT_ROOT=<path to this repository> and, if the
# interpreter is not on PATH, PYTHON=<interpreter> before submitting. Submit from
# the repository root so the relative --output/--error log paths resolve.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
echo "Node: $(hostname)  Start: $(date)"
# rung r0 (all techniques OFF: all segments resident + full autograd graph) is the peak;
# 838M at bs=4 seq=512 needs the most VRAM there. Later rungs shrink it toward ~964 MB.
"$PYTHON" final_paper_scripts_results/segmented_model/memory_reduction_techniques/gpu_train_cost.py \
    --preset large_8x2x2x8 --device cuda --batch 4 --n-steps 3
echo "Done: $(date)"
