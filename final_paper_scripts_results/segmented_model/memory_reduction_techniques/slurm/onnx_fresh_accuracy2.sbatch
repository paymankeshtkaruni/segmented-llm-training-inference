#!/bin/bash
#SBATCH --job-name=onnx_fresh_acc2
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=04:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/onnx_fresh_acc2_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/onnx_fresh_acc2_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
BASE="${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
MRT="$BASE/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
cd "$MRT"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
PY="${PYTHON:-python}"
EXP="$BASE/final_paper_scripts_results/segmented_model/onnx/c_seg_quality_fresh"
echo "Node: $(hostname)  Start: $(date)"
for prov in cuda cpu; do
  echo "=== ONNX $prov stream ==="
  "$PY" eval_onnx_accuracy.py --onnx-dir "$EXP" --provider $prov
  echo "=== ONNX $prov preload ==="
  "$PY" eval_onnx_accuracy.py --onnx-dir "$EXP" --provider $prov --preload-weights
done
echo "Done: $(date)"
