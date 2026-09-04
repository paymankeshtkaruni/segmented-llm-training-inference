#!/bin/bash
#SBATCH --job-name=x1_verify_cpu
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=06:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/x1_verify_cpu_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/x1_verify_cpu_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
PY="${PYTHON:-python}"
echo "Node: $(hostname)  Start: $(date)"
# smoke first (small model, disk store, minutes); abort before the big run on any failure
"$PY" verify_modes.py --device cpu --store disk --preset small_8x2x2x8 \
    --batch 2 --seq 24 --steps 2 --out-dir results/verify_modes/smoke_x1_cpu
# the real thing: 0.84B, all six training modes, paper-true store binding (CPU=disk)
"$PY" verify_modes.py --device cpu --store disk --preset large_8x2x2x8 \
    --batch 2 --seq 512 --steps 4 --out-dir results/verify_modes
echo "Done: $(date)"
