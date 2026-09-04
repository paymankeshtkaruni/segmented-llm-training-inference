#!/bin/bash
#SBATCH --job-name=onnx_fresh_acc
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=06:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/onnx_fresh_acc_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/onnx_fresh_acc_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
BASE="${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
MRT="$BASE/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
cd "$BASE"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
PY="${PYTHON:-python}"
CKPT="$MRT/results/quality_fresh_pair/T1/checkpoints/best.pt"
EXP="$BASE/final_paper_scripts_results/segmented_model/onnx/c_seg_quality_fresh"
OUT="$MRT/results/quality_fresh_pair"
echo "Node: $(hostname)  Start: $(date)"

echo "=== 1/5 export fresh trained weights to segmented ONNX ==="
"$PY" final_paper_scripts_results/segmented_model/scripts/seg_c_export_segmented.py \
    --preset small_8x2x2x8 --checkpoint "$CKPT" --out-dir "$EXP"

for prov in cuda cpu; do
  for mode in stream preload; do
    FLAG=""; [ "$mode" = "preload" ] && FLAG="--preload-weights"
    echo "=== ONNX $prov $mode: full test set ==="
    "$PY" scripts/infer_onnx.py --onnx-dir "$EXP" --provider $prov $FLAG \
        --evaluate-test --num-test-examples 11654 --eval-batch-size 32 \
        --state-dir "$TMPDIR/onnx_${prov}_${mode}" \
        2>&1 | tee "$OUT/onnx_fresh_${prov}_${mode}.log" | tail -5
  done
done
echo "Done: $(date)"
