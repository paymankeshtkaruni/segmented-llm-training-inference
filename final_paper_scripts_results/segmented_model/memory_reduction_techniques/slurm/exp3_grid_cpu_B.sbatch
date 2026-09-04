#!/bin/bash
#SBATCH --job-name=exp3_grid_cpuB
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=12:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp3_grid_cpuB_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp3_grid_cpuB_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_ARENA_MAX=2
PY="${PYTHON:-python}"
echo "Node: $(hostname)  Start: $(date)"
MODES="
T7  11100111000 immediate  -
T8  11110111000 immediate  -
T9  11111111100 immediate  -
T10 11100111000 immediate  0.0
T11 11110111000 immediate  0.0
T12 11111111100 immediate  0.0
"
for rep in 1 2 3; do
  echo "$MODES" | while read -r name code style drop; do
    [ -z "$name" ] && continue
    DARG=""; [ "$drop" != "-" ] && DARG="--dropout $drop"
    echo "=== rep $rep: $name ($code $style drop=$drop) ==="
    "$PY" measure_one.py --kind train --preset large_8x2x2x8 --device cpu \
        --tech-code "$code" --update-style "$style" $DARG \
        --prefix "$name" --out-dir "results/exp3_grid/cpu_${name}_rep$rep"
  done
done
echo "Done: $(date)"
