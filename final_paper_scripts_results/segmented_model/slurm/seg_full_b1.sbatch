#!/bin/bash
#SBATCH --job-name=seg_full_b1
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=00:30:00
#SBATCH --output=final_paper_scripts_results/segmented_model/outputs/logs/seg_full_b1_SEED_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/outputs/logs/seg_full_b1_SEED_%j.err
# Site-independent: export PROJECT_ROOT=<path to this repository> and, if the
# interpreter is not on PATH, PYTHON=<interpreter> before submitting. Submit from
# the repository root so the relative --output/--error log paths resolve.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY="${PYTHON:-python}"
echo "Node: $(hostname) Start: $(date) seed=SEED"
"$PY" final_paper_scripts_results/segmented_model/scripts/seg_full_train_b1.py --preset small_8x2x2x8 --device cuda --epochs 5 --batch-size 64 --seed SEED
echo "Done: $(date)"
