#!/bin/bash
#SBATCH --job-name=fm_train_gpu
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=06:00:00
#SBATCH --output=final_paper_scripts_results/full_model/outputs/logs/train_gpu_%j.out
#SBATCH --error=final_paper_scripts_results/full_model/outputs/logs/train_gpu_%j.err
# Site-independent: export PROJECT_ROOT=<path to this repository> and, if the
# interpreter is not on PATH, PYTHON=<interpreter> before submitting. Submit from
# the repository root so the relative --output/--error log paths resolve.

# A1 accuracy training — FULL run, GPU compute.
set -euo pipefail
REPO="${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
cd "$REPO"
mkdir -p final_paper_scripts_results/full_model/outputs/logs
module load gcc/13.2.0 cuda/12.8.2
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON="${PYTHON:-python}"

echo "=== fm train gpu (FULL) ===  Job: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
"$PYTHON" -c "import torch; assert torch.cuda.is_available(); print('CUDA:', torch.cuda.get_device_name(0))"
"$PYTHON" final_paper_scripts_results/full_model/scripts/full_model_train_gpu.py \
    --device cuda --epochs 5 --batch-size 64 --workers 4 --log-every 50
echo "Done: $(date)"
