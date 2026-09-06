#!/bin/bash
#SBATCH --job-name=exp8_ic_084b_onnx_disk
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
# (site note: pin to one homogeneous node pool at your site;
#  the original runs excluded non-matching nodes here)
#SBATCH --time=48:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp8_ic_084b_onnx_disk_%j.out
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"  # repo root: the ONNX manifest stores repo-relative paths
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2
PY="${PYTHON:-python}"
"$PY" final_paper_scripts_results/segmented_model/memory_reduction_techniques/exp8_long_serve.py --engine onnx --onnx-dir final_paper_scripts_results/segmented_model/onnx/c_seg_large --device cpu --n-requests 500 --prompt-len 256 --gen-tokens 8 --prefix ic --max-hours 46 --out-dir "final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/exp8_sustained/ic_084b_onnx_disk"
echo "Done: $(date)"
