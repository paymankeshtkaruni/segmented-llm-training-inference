#!/bin/bash
#SBATCH --job-name=mrt_full_decode
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
#SBATCH --time=00:40:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_full_decode_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_full_decode_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
PYTHON="${PYTHON:-python}"
MRT=final_paper_scripts_results/segmented_model/memory_reduction_techniques
echo "Node: $(hostname)  Start: $(date)"
# FULL-MODEL DECODE TIMING — like-for-like counterpart of the segmented inference
# ladder protocol (batch 1, 256-token prompt, warmup 1, 8 timed greedy tokens),
# so the dumbbell figure can pair (memory, per-token time) on the inference rows.
"$PYTHON" "$MRT/full_decode_cost.py" --preset large_8x2x2x8 --device cuda
"$PYTHON" "$MRT/full_decode_cost.py" --preset large_8x2x2x8 --device cpu
echo "Done: $(date)"
