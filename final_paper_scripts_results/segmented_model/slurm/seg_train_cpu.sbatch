#!/bin/bash
#SBATCH --job-name=seg_train_cpu
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=2-00:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/outputs/logs/seg_train_cpu_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/outputs/logs/seg_train_cpu_%j.err
# Site-independent: export PROJECT_ROOT=<path to this repository> and, if the
# interpreter is not on PATH, PYTHON=<interpreter> before submitting. Submit from
# the repository root so the relative --output/--error log paths resolve.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
echo "=== seg train CPU (FULL) ===  Node: $(hostname)  Start: $(date)"
"$PYTHON" final_paper_scripts_results/segmented_model/scripts/seg_train_cpu.py --device cpu --epochs 5 --batch-size 64     --store-kind cpu_ram --workers 8 --log-every 50
echo "Done: $(date)"
