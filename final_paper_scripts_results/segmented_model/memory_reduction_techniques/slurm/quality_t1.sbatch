#!/bin/bash
#SBATCH --job-name=quality_t1
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=02:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/quality_t1_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/quality_t1_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY="${PYTHON:-python}"
echo "Node: $(hostname)  Start: $(date)"
# Full quality training in the fastest protocol-compatible mode:
# T1 = resident, no recompute, dropout 0.1, after-full-backward + global clip.
# Protocol identical to the archived reference run: lr 3e-4, wd 0.1,
# betas (0.9,0.95), clip 1.0, warmup 200, cosine, 5 epochs, batch 64, seed 42.
"$PY" segmentation_management/trainer.py \
    --preset small_8x2x2x8 --device cuda \
    --tech-code 11100111000 --full-run \
    --epochs 5 --batch-size 64 --warmup 200 \
    --out-dir results/quality_T1_gpu
echo "Done: $(date)"
