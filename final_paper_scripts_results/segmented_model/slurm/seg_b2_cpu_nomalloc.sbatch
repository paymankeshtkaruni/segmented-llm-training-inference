#!/bin/bash
#SBATCH --job-name=seg_b2_cpu_nm
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=01:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/outputs/logs/seg_b2_cpu_nm_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/outputs/logs/seg_b2_cpu_nm_%j.err
# Site-independent: export PROJECT_ROOT=<path to this repository> and, if the
# interpreter is not on PATH, PYTHON=<interpreter> before submitting. Submit from
# the repository root so the relative --output/--error log paths resolve.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
# ABLATION: malloc env vars intentionally NOT set
PYTHON="${PYTHON:-python}"
echo "Node: $(hostname)  Start: $(date)  MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-unset}"
"$PYTHON" final_paper_scripts_results/segmented_model/scripts/seg_infer_cost_lib.py --preset large_8x2x2x8 --device cpu     --batch 4 --seq 512 --gen-tokens 4 --prompt-len 64 --out-dir final_paper_scripts_results/segmented_model/outputs/seg_b2_cpu_nomalloc --prefix seg_b2
echo "Done: $(date)"
