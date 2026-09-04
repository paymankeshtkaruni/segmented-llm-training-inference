#!/bin/bash
#SBATCH --job-name=a2_onnx_only
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=12:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/a2_onnx_only_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/a2_onnx_only_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
PY="${PYTHON:-python}"
MRT=final_paper_scripts_results/segmented_model/memory_reduction_techniques
OUT=$MRT/results/verify_modes
ONNX=final_paper_scripts_results/segmented_model/onnx/c_seg_large
mkdir -p "$OUT"
echo "Node: $(hostname)  Start: $(date)"

for prov in cuda cpu; do
  echo "=== ONNX $prov streaming (30 examples) ==="
  "$PY" scripts/infer_onnx.py --onnx-dir "$ONNX" --provider $prov \
      --evaluate-test --num-test-examples 30 \
      --state-dir "$TMPDIR/onnx_${prov}_stream" 2>&1 | tee "$OUT/onnx_${prov}_stream.log"
  echo "=== ONNX $prov preload-weights (30 examples) ==="
  "$PY" scripts/infer_onnx.py --onnx-dir "$ONNX" --provider $prov \
      --evaluate-test --num-test-examples 30 --preload-weights \
      --state-dir "$TMPDIR/onnx_${prov}_preload" 2>&1 | tee "$OUT/onnx_${prov}_preload.log"
done

for dev in cuda cpu; do
  echo "=== torch segmented $dev (30 examples) ==="
  PYTHONPATH=src "$PY" scripts/infer_segmented.py \
      --checkpoint experiments/outputs_paper/checkpoints/segmented_gpu --checkpoint-name best \
      --evaluate-test --num-test-examples 30 --device $dev 2>&1 | tee "$OUT/torch_${dev}_30.log"
done
echo "Done: $(date)"
