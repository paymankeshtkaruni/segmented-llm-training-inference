#!/bin/bash
#SBATCH --job-name=mrt_cpu_scale_7b
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=16:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_cpu_scale_7b_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_cpu_scale_7b_%j.err
# Site-independent: export PROJECT_ROOT=<path to this repository> and, if the
# interpreter is not on PATH, PYTHON=<interpreter> before submitting. Submit from
# the repository root so the relative --output/--error log paths resolve.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
# glibc heap tuning — SAME settings as the 0.84B CPU ladder runs (comparability).
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072
PYTHON="${PYTHON:-python}"
MRT=final_paper_scripts_results/segmented_model/memory_reduction_techniques
echo "Node: $(hostname)  Start: $(date)"
# Scale experiment on CPU, ~6.9B params. FULL model: states ~110 GB + graph
# activations -> expect ~180-200 GB RSS (fits --mem=400G) -> MEASURED
# full-vs-segmented ratio at 7B. seg_*: disk store (~110 GB scratch in _work).
# n-steps 2 to bound the segmented step time (~45-60 min/step expected).
for RUN in full_train full_infer seg_train seg_infer; do
  echo "===== $RUN  $(date) ====="
  "$PYTHON" "$MRT/scale_cost.py" --preset xxl7b_8x2x2x8 --device cpu --run "$RUN" \
      --batch 4 --n-steps 2
done
# the disk segment/grad/Adam stores are regenerable scratch (~110 GB) — free them
rm -rf "$MRT/results/scale_cpu_xxl7b/_work"
echo "Done: $(date)"
