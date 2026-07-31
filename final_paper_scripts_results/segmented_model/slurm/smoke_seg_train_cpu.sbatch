#!/bin/bash
#SBATCH --job-name=seg_smoke_train_cpu
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=00:20:00
#SBATCH --output=final_paper_scripts_results/segmented_model/outputs/logs/smoke_seg_train_cpu_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/outputs/logs/smoke_seg_train_cpu_%j.err
# Site-independent: export PROJECT_ROOT=<path to this repository> and, if the
# interpreter is not on PATH, PYTHON=<interpreter> before submitting. Submit from
# the repository root so the relative --output/--error log paths resolve.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
echo "=== seg smoke train cpu ===  Node: $(hostname)  Start: $(date)"
"$PYTHON" final_paper_scripts_results/segmented_model/scripts/seg_train_cpu.py --device cpu     --epochs 1 --batch-size 4 --max-train-steps 3 --max-val-steps 2 --max-rows 64     --warmup 1 --log-every 1 --out-dir final_paper_scripts_results/segmented_model/outputs/smoke_seg_train_cpu
echo "Done: $(date)"
