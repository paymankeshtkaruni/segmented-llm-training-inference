#!/bin/bash
#SBATCH --job-name=verify_scale_gpu
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
#SBATCH --time=08:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/verify_scale_gpu_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/verify_scale_gpu_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
# site-independent: override PROJECT_ROOT / PYTHON to run this elsewhere.
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT}}"
PY="${PYTHON:-"${PYTHON:-python}"}"
cd "$PROJECT_ROOT/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
echo "Node: $(hostname)  Start: $(date)"
# smoke first (small model, seconds); set -e aborts before the big runs on any failure
"$PY" verify_scale_forward.py --device cuda --store cpu_ram --preset small_8x2x2x8 \
    --batch 2 --seq 24 --prompt-len 8 --new-tokens 4 \
    --out-dir results/verify_modes/smoke_scale
# the real thing: forward-state identity at 0.84B, 3.1B and 6.9B. The reference and
# the segmented engine are never on the GPU at the same time (that is the point:
# 6.9B fp32 weights are 27.4 GB, so the pair does not fit a 40 GB A100).
for P in large_8x2x2x8 xl3b_8x2x2x8 xxl7b_8x2x2x8; do
  echo "===== $P ====="
  "$PY" verify_scale_forward.py --device cuda --store cpu_ram --preset "$P" \
      --batch 2 --seq 512 --out-dir results/verify_modes
done
echo "Done: $(date)"
