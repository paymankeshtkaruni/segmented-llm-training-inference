#!/bin/bash
#SBATCH --job-name=store_bandwidth
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
#SBATCH --time=01:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/store_bandwidth_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/store_bandwidth_%j.err
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
# GPU cell binding (paper): inactive segments park in host RAM.
echo "=== GPU: cuda / cpu_ram store ==="
"$PY" store_bandwidth.py --device cuda --store cpu_ram --preset large_8x2x2x8 \
    --passes 3 --out-dir results/measurement_checks
# CPU cell binding (paper): inactive segments park on disk. Same glibc allocator and
# thread pinning the CPU cost cells run under, so the measured path is their path.
export OMP_NUM_THREADS=16
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_ARENA_MAX=2
echo "=== CPU: cpu / disk store ==="
"$PY" store_bandwidth.py --device cpu --store disk --preset large_8x2x2x8 \
    --passes 3 --out-dir results/measurement_checks
echo "Done: $(date)"
