#!/bin/bash
#SBATCH --job-name=seg_smoke_cost
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=00:45:00
#SBATCH --output=final_paper_scripts_results/segmented_model/outputs/logs/smoke_seg_cost_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/outputs/logs/smoke_seg_cost_%j.err
# Site-independent: export PROJECT_ROOT=<path to this repository> and, if the
# interpreter is not on PATH, PYTHON=<interpreter> before submitting. Submit from
# the repository root so the relative --output/--error log paths resolve.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
echo "Node: $(hostname)  Start: $(date)"
"$PYTHON" final_paper_scripts_results/segmented_model/scripts/seg_cost_lib.py --preset large_8x2x2x8 --device cuda     --batch 4 --n-steps 1 --out-dir final_paper_scripts_results/segmented_model/outputs/smoke_seg_cost --prefix seg_cost
echo "Done: $(date)"
