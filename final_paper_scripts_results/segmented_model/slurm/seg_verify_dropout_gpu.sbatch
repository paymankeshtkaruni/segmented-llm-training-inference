#!/bin/bash
#SBATCH --job-name=seg_verify_drop_gpu
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=00:15:00
#SBATCH --output=final_paper_scripts_results/segmented_model/outputs/logs/seg_verify_drop_gpu_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/outputs/logs/seg_verify_drop_gpu_%j.err
# Site-independent: export PROJECT_ROOT=<path to this repository> and, if the
# interpreter is not on PATH, PYTHON=<interpreter> before submitting. Submit from
# the repository root so the relative --output/--error log paths resolve.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
echo "=== GPU dropout-consistency check ===  Node: $(hostname)  Start: $(date)"
"$PYTHON" final_paper_scripts_results/segmented_model/scripts/seg_verify_dropout_consistency.py cuda
echo "Done: $(date)"
