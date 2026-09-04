#!/bin/bash
#SBATCH --job-name=exp4_res_cpu
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:A100:1

#SBATCH --time=04:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp4_res_cpu\_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp4_res_cpu\_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2
PY="${PYTHON:-python}"
ONNX_BASE="${PROJECT_ROOT}/final_paper_scripts_results/segmented_model/onnx"
echo "Node: $(hostname)  Start: $(date)"
# one-time exports (skipped if already present)
[ -f "$ONNX_BASE/c_full_large/full_model.onnx" ] || \
  "$PY" ../scripts/seg_c_export_baked.py --what full  --out-dir "$ONNX_BASE/c_full_large"
[ -f "$ONNX_BASE/c_baked_large/manifest.json" ] || \
  "$PY" ../scripts/seg_c_export_baked.py --what baked --out-dir "$ONNX_BASE/c_baked_large"
for rep in 1 2 3; do
  echo "=== rep $rep: O1 full ==="
  "$PY" onnx_resident_cost.py --mode full  --onnx-dir "$ONNX_BASE/c_full_large" \
      --provider cpu --prompt-len 256 --gen-tokens 32 --prefix O1_full \
      --out-dir "results/exp4_infer/onnx_cpu_O1_full_rep$rep"
  echo "=== rep $rep: O2 baked ==="
  "$PY" onnx_resident_cost.py --mode baked --onnx-dir "$ONNX_BASE/c_baked_large" \
      --provider cpu --prompt-len 256 --gen-tokens 32 --prefix O2_baked \
      --out-dir "results/exp4_infer/onnx_cpu_O2_baked_rep$rep"
done
echo "Done: $(date)"
