#!/bin/bash
#SBATCH --job-name=fm_train_cpu
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=1-00:00:00
#SBATCH --output=final_paper_scripts_results/full_model/outputs/logs/train_cpu_%j.out
#SBATCH --error=final_paper_scripts_results/full_model/outputs/logs/train_cpu_%j.err
# Site-independent: export PROJECT_ROOT=<path to this repository> and, if the
# interpreter is not on PATH, PYTHON=<interpreter> before submitting. Submit from
# the repository root so the relative --output/--error log paths resolve.

# A1 accuracy training — FULL run, CPU compute (A100 only for filesystem access).
set -euo pipefail
REPO="${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
cd "$REPO"
mkdir -p final_paper_scripts_results/full_model/outputs/logs
module load gcc/13.2.0 cuda/12.8.2
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON="${PYTHON:-python}"

echo "=== fm train cpu (FULL) ===  Job: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
"$PYTHON" final_paper_scripts_results/full_model/scripts/full_model_train_cpu.py \
    --device cpu --epochs 5 --batch-size 64 --workers 8 --log-every 50
echo "Done: $(date)"
