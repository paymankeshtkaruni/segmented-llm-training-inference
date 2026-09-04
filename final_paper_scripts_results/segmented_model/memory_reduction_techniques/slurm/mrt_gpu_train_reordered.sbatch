#!/bin/bash
#SBATCH --job-name=mrt_gpu_train_reord
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=02:00:00
#SBATCH --array=1-3
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_gpu_train_reord_%A_%a.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_gpu_train_reord_%A_%a.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
echo "Node: $(hostname)  Repeat: ${SLURM_ARRAY_TASK_ID}  Start: $(date)"
# Cheap-first reordered TRAIN_LADDER (techniques.py @ a348471): all free techniques,
# then stream_segments (r9), then free_device (r10). One full ladder per array task.
"$PYTHON" final_paper_scripts_results/segmented_model/memory_reduction_techniques/gpu_train_cost.py \
    --preset large_8x2x2x8 --device cuda --batch 4 --n-steps 3 \
    --out-dir final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/gpu_train_reordered_rep${SLURM_ARRAY_TASK_ID}
echo "Done: $(date)"
