#!/bin/bash
#SBATCH --job-name=mrt_validate_cell
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
#SBATCH --time=02:30:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_validate_cell_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_validate_cell_%j.err
# Site-independent: export PROJECT_ROOT=<path to this repository> and, if the
# interpreter is not on PATH, PYTHON=<interpreter> before submitting. Submit from
# the repository root so the relative --output/--error log paths resolve.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
MRT=final_paper_scripts_results/segmented_model/memory_reduction_techniques
echo "Node: $(hostname)  Start: $(date)"
# GENERAL-FIT VALIDATION CELL — pre-registered prediction (results/general_fit.json,
# committed before this job ran): 3.09B @ 16x4x4x16 -> T = 466 s/step, M = 1162 MB.
# The cell was never used for fitting: new size AND new partition simultaneously.
# Pinned to the A100-40GB variant (the time fit's node class).
"$PYTHON" "$MRT/scale_cost.py" --preset xl3b_16x4x4x16 --device cuda --run seg_train \
    --batch 4 --n-steps 2 --out-dir "$MRT/results/scale_gpu_xl3b_16x4x4x16"
echo "Done: $(date)"
