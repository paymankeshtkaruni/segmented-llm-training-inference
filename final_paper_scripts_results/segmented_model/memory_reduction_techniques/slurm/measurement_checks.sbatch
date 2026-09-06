#!/bin/bash
#SBATCH --job-name=measurement_checks
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
#SBATCH --time=04:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/measurement_checks_%j.out
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2
PY="${PYTHON:-python}"
# CHECK A: inference profiler bridge — profiled (cost protocol) vs unprofiled
# (long-serve runner) back-to-back on THIS node, fast mode (I1) + slow mode (I4).
echo "=== A1: I1 profiled (cost protocol) ==="
"$PY" measure_one.py --kind infer --preset large_8x2x2x8 --device cuda --tech-code 11100111000 --prompt-len 256 --gen-tokens 32 --prefix i1_profiled --out-dir results/measurement_checks/i1_profiled
echo "=== A2: I1 unprofiled (long-serve runner, 50 requests) ==="
"$PY" exp8_long_serve.py --engine torch --preset large_8x2x2x8 --device cuda --tech-code 11100111000 --n-requests 50 --prompt-len 256 --gen-tokens 32 --prefix i1_unprofiled --out-dir results/measurement_checks/i1_unprofiled
echo "=== A3: I4 profiled ==="
"$PY" measure_one.py --kind infer --preset large_8x2x2x8 --device cuda --tech-code 11101111101 --prompt-len 256 --gen-tokens 8 --prefix i4_profiled --out-dir results/measurement_checks/i4_profiled
echo "=== A4: I4 unprofiled (20 requests) ==="
"$PY" exp8_long_serve.py --engine torch --preset large_8x2x2x8 --device cuda --tech-code 11101111101 --n-requests 20 --prompt-len 256 --gen-tokens 8 --prefix i4_unprofiled --out-dir results/measurement_checks/i4_unprofiled
# CHECK C: RSS-sampler interference — CPU fastest mode, 25 steps with sampler,
# 25 without, same node back-to-back.
echo "=== C1: CPU 25 steps WITH sampler ==="
"$PY" exp8_long_run.py --preset large_8x2x2x8 --device cpu --tech-code 11100111000 --update-style immediate --n-steps 25 --prefix sampler_on --out-dir results/measurement_checks/sampler_on
echo "=== C2: CPU 25 steps WITHOUT sampler ==="
"$PY" exp8_long_run.py --preset large_8x2x2x8 --device cpu --tech-code 11100111000 --update-style immediate --n-steps 25 --no-mem-sampler --prefix sampler_off --out-dir results/measurement_checks/sampler_off
echo "Node: $(hostname)  Done: $(date)"
