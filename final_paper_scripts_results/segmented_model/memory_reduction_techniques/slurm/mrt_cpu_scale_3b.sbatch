#!/bin/bash
#SBATCH --job-name=mrt_cpu_scale_3b
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=10:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_cpu_scale_3b_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_cpu_scale_3b_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
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
# Scale experiment on CPU, ~3.1B params. Host RAM is large enough that the FULL
# model still runs (states ~48 GB + graph activations -> expect ~100 GB RSS), so
# this yields a MEASURED full-vs-segmented ratio at 3B. seg_* uses the disk store.
for RUN in full_train full_infer seg_train seg_infer; do
  echo "===== $RUN  $(date) ====="
  "$PYTHON" "$MRT/scale_cost.py" --preset xl3b_8x2x2x8 --device cpu --run "$RUN" \
      --batch 4 --n-steps 3
done
# the disk segment/grad/Adam stores are regenerable scratch (~50 GB) — free them
rm -rf "$MRT/results/scale_cpu_xl3b/_work"
echo "Done: $(date)"
