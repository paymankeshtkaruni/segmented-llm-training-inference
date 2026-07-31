#!/bin/bash
#SBATCH --job-name=fm_onnx_cost
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=00:30:00
#SBATCH --output=final_paper_scripts_results/full_model/outputs/logs/onnx_cost_%j.out
#SBATCH --error=final_paper_scripts_results/full_model/outputs/logs/onnx_cost_%j.err
# Site-independent: export PROJECT_ROOT=<path to this repository> and, if the
# interpreter is not on PATH, PYTHON=<interpreter> before submitting. Submit from
# the repository root so the relative --output/--error log paths resolve.

# C — TORCH-FREE ONNX inference cost (LARGE). Note: NO PYTHONPATH=src here — the C
# scripts import only _onnx_cost_lib (onnxruntime/numpy/psutil), never torch/_common.
set -euo pipefail
cd "${PROJECT_ROOT}/final_paper_scripts_results/full_model/scripts"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
PYTHON="${PYTHON:-python}"
echo "=== C onnx cost (torch-free) ===  Node: $(hostname)  Start: $(date)"
echo "--- GPU profiler ---"; "$PYTHON" full_model_onnx_infer_cost_profiler_gpu.py
echo "--- GPU light ---";    "$PYTHON" full_model_onnx_infer_cost_gpu.py
echo "--- CPU profiler ---"; "$PYTHON" full_model_onnx_infer_cost_profiler_cpu.py
echo "--- CPU light ---";    "$PYTHON" full_model_onnx_infer_cost_cpu.py
echo "Done: $(date)"
