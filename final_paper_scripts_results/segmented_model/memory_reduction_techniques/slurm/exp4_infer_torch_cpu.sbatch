#!/bin/bash
#SBATCH --job-name=exp4_torch_cpu
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:A100:1

#SBATCH --time=08:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp4_torch_cpu\_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp4_torch_cpu\_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2
PY="${PYTHON:-python}"
echo "Node: $(hostname)  Start: $(date)"
MODES="
I1 11100111000
I2 11100111001
I3 11101111100
I4 11101111101
"
for rep in 1 2 3; do
  echo "$MODES" | while read -r name code; do
    [ -z "$name" ] && continue
    echo "=== rep $rep: $name ==="
    "$PY" measure_one.py --kind infer --preset large_8x2x2x8 --device cpu \
        --tech-code "$code" --prompt-len 256 --gen-tokens 32 \
        --prefix "$name" --out-dir "results/exp4_infer/cpu_${name}_rep$rep"
  done
done
echo "Done: $(date)"
