#!/bin/bash
#SBATCH --job-name=a2_exact_gpu
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=04:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/a2_exact_gpu_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/a2_exact_gpu_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY="${PYTHON:-python}"
MRT=final_paper_scripts_results/segmented_model/memory_reduction_techniques
OUT=$MRT/results/verify_modes
mkdir -p "$OUT"
echo "Node: $(hostname)  Start: $(date)"

echo "=== Part 1: torch inference modes (4) ==="
"$PY" "$MRT/verify_infer_modes.py" --device cuda

echo "=== Part 2: ONNX streaming (30 test examples) ==="
"$PY" scripts/infer_onnx.py --onnx-dir final_paper_scripts_results/segmented_model/onnx/c_seg_large \
    --provider cuda --evaluate-test --num-test-examples 30 \
    --state-dir "$TMPDIR/onnx_state_stream" 2>&1 | tee "$OUT/onnx_gpu_stream.log"

echo "=== Part 2: ONNX preload-weights (30 test examples) ==="
"$PY" scripts/infer_onnx.py --onnx-dir final_paper_scripts_results/segmented_model/onnx/c_seg_large \
    --provider cuda --evaluate-test --num-test-examples 30 --preload-weights \
    --state-dir "$TMPDIR/onnx_state_preload" 2>&1 | tee "$OUT/onnx_gpu_preload.log"

echo "=== Part 2: torch segmented, same 30 examples ==="
PYTHONPATH=src "$PY" scripts/infer_segmented.py \
    --checkpoint experiments/outputs_paper/checkpoints/segmented_gpu --checkpoint-name best \
    --evaluate-test --num-test-examples 30 --device cuda 2>&1 | tee "$OUT/torch_gpu_30.log"
echo "Done: $(date)"
