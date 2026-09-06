#!/bin/bash
#SBATCH --job-name=sampler_off_rerun
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
#SBATCH --time=01:30:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/sampler_off_rerun_%j.out
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2
PY="${PYTHON:-python}"
# C2 rerun + paired C1 on the SAME node for a clean within-node comparison
"$PY" exp8_long_run.py --preset large_8x2x2x8 --device cpu --tech-code 11100111000 --update-style immediate --n-steps 25 --prefix c1_on --out-dir results/measurement_checks/sampler_on_paired
"$PY" exp8_long_run.py --preset large_8x2x2x8 --device cpu --tech-code 11100111000 --update-style immediate --n-steps 25 --no-mem-sampler --prefix c2_off --out-dir results/measurement_checks/sampler_off
echo "Node: $(hostname)  Done: $(date)"
