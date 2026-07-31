#!/bin/bash
#SBATCH --job-name=fm_smoke_train_cpu
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=00:20:00
#SBATCH --output=final_paper_scripts_results/full_model/outputs/logs/smoke_train_cpu_%j.out
#SBATCH --error=final_paper_scripts_results/full_model/outputs/logs/smoke_train_cpu_%j.err
# Site-independent: export PROJECT_ROOT=<path to this repository> and, if the
# interpreter is not on PATH, PYTHON=<interpreter> before submitting. Submit from
# the repository root so the relative --output/--error log paths resolve.

# A1 accuracy-training SMOKE (CPU compute). gpu:A100:1 is requested only for
# the shared parallel filesystem filesystem access; the workload runs on CPU (--device cpu).
# Purpose: verify the real sbatch path works before the full run.

set -euo pipefail

REPO="${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
cd "$REPO"
mkdir -p final_paper_scripts_results/full_model/outputs/logs

module load gcc/13.2.0 cuda/12.8.2

export PYTHONPATH=src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHON="${PYTHON:-python}"

echo "=== fm smoke train cpu ==="
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"

"$PYTHON" final_paper_scripts_results/full_model/scripts/full_model_train_cpu.py \
    --device cpu \
    --epochs 1 \
    --batch-size 4 \
    --max-train-steps 3 \
    --max-val-steps 2 \
    --max-train-rows 32 \
    --max-val-rows 16 \
    --max-test-rows 16 \
    --warmup-steps 1 \
    --log-every 1 \
    --out-dir final_paper_scripts_results/full_model/outputs/smoke_train_cpu

echo "Done: $(date)"
