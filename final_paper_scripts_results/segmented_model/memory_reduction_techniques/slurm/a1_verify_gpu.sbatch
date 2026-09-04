#!/bin/bash
#SBATCH --job-name=a1_verify_gpu
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=01:30:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/a1_verify_gpu_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/a1_verify_gpu_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY="${PYTHON:-python}"
echo "Node: $(hostname)  Start: $(date)"
# smoke first (small model, seconds); abort before the big run if identity is broken
"$PY" verify_modes.py --device cuda --preset small_8x2x2x8 --batch 2 --seq 24 --steps 2 \
    --out-dir results/verify_modes/smoke_gpu
# the real thing: paper model size, all three modes, 4 steps
"$PY" verify_modes.py --device cuda --preset large_8x2x2x8 --batch 2 --seq 512 --steps 4 \
    --out-dir results/verify_modes
echo "Done: $(date)"
