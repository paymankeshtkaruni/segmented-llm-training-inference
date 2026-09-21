#!/bin/bash
#SBATCH --job-name=gran_std_decode_gpu
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
# (site note: pin to one homogeneous node pool at your site;
#  the original runs excluded non-matching nodes here)
#SBATCH --time=01:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/gran_std_decode_gpu_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/gran_std_decode_gpu_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2
PY="${PYTHON:-python}"
echo "Node: $(hostname)  Start: $(date)"
# Standard partition (8x2x2x8) decode cells under the granularity protocol
# (prompt 256, 8 generated tokens), so that the granularity table compares
# coarse / standard / fine decode at one protocol.
for rep in 1 2 3; do
  echo "=== gran_8x2x2x8 I1 rep$rep ==="
  "$PY" measure_one.py --kind infer --preset large_8x2x2x8 --device cuda --tech-code 11100111000 --prompt-len 256 --gen-tokens 8 --seg-override 8x2x2x8 --prefix I1 --out-dir "results/exp6_fast/gran_8x2x2x8_I1_rep$rep"
done
for rep in 1 2 3; do
  echo "=== gran_8x2x2x8 I4 rep$rep ==="
  "$PY" measure_one.py --kind infer --preset large_8x2x2x8 --device cuda --tech-code 11101111101 --prompt-len 256 --gen-tokens 8 --seg-override 8x2x2x8 --prefix I4 --out-dir "results/exp6_fast/gran_8x2x2x8_I4_rep$rep"
done
echo "Done: $(date)"
