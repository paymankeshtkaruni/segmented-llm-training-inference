#!/bin/bash
#SBATCH --job-name=onnx_full_acc
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=02:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/onnx_full_accuracy_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/onnx_full_accuracy_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
# Full-session (O1) ONNX row of the fresh inference-accuracy table: export the
# quality checkpoint as ONE baked ONNX session, run the same full-test-set greedy
# exact-match as the segmented ONNX modes, then check its predictions against the
# torch dump. Paper: Sec. VI-D six-pipeline agreement / Fig. 6 caption.
# site-independent: override PROJECT_ROOT / PYTHON to run this elsewhere.
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT}}"
PY="${PYTHON:-"${PYTHON:-python}"}"
SM="$PROJECT_ROOT/final_paper_scripts_results/segmented_model"
MRT="$SM/memory_reduction_techniques"
cd "$MRT"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
# The ORT CUDA provider needs cuDNN/cuBLAS visible, and this process imports no
# torch (which would otherwise pull them in) — derive the venv wheel lib dirs from
# the interpreter itself so the job is not tied to one site's venv path.
ORT_LIBS="$("$PY" -c 'import os, nvidia; b = os.path.dirname(nvidia.__file__); print(":".join(os.path.join(b, p, "lib") for p in ("cudnn", "cublas", "cuda_runtime", "cufft", "curand", "nvjitlink")))')"
export LD_LIBRARY_PATH="$ORT_LIBS:${LD_LIBRARY_PATH:-}"
EXP="$SM/onnx/c_full_quality_fresh"
OUT="$MRT/results/quality_fresh_pair"
CK="$OUT/T1/checkpoints/best.pt"
echo "Node: $(hostname)  Start: $(date)"
# one-time export of the TRAINED weights as a single baked session (skipped if present)
[ -f "$EXP/full_model.onnx" ] || \
  "$PY" ../scripts/seg_c_export_baked.py --what full --preset small_8x2x2x8 \
      --checkpoint "$CK" --out-dir "$EXP"
for prov in cuda cpu; do
  echo "=== ONNX $prov full-session accuracy (11,654 test rows) ==="
  "$PY" eval_onnx_accuracy.py --mode full --onnx-dir "$EXP" --provider "$prov"
done
for prov in cpu cuda; do
  echo "=== I-A1 agreement: torch vs ONNX full $prov ==="
  "$PY" ia1_agreement.py \
      --torch-preds "$OUT/preds_torch.json" \
      --onnx-preds "$OUT/preds_onnx_${prov}_full.json" \
      --checkpoint "$CK" --device cuda \
      --out "ia1_agreement_full_${prov}.json"
done
echo "Done: $(date)"
