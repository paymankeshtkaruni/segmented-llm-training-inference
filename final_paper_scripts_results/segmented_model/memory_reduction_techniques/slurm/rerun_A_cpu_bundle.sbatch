#!/bin/bash
#SBATCH --job-name=rerun_cpu_bundle084
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=06:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/rerun_cpu_bundle084_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/rerun_cpu_bundle084_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
# EXACT env of the 3.09B/6.86B scale cells (mrt_cpu_scale_*.sbatch):
# 16 pinned threads, mmap+trim malloc tuning, NO MALLOC_ARENA_MAX.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072
PYTHON="${PYTHON:-python}"
MRT=final_paper_scripts_results/segmented_model/memory_reduction_techniques
echo "Node: $(hostname)  Start: $(date)  OMP_NUM_THREADS=$OMP_NUM_THREADS"
# Config-consistency reruns of the remaining 0.84B CPU cells (audit SEV-1 #2 and
# SEV-2 #4): full inference, segmented train (no ARENA cap, unlike the ladder),
# and segmented inference. Outputs join full_train_metrics.json (job 15115043)
# in results/scale_cpu_large/. Nothing pre-existing is overwritten.
for RUN in full_infer seg_train seg_infer; do
  echo "===== $RUN  $(date) ====="
  "$PYTHON" "$MRT/scale_cost.py" --preset large_8x2x2x8 --device cpu --run "$RUN" \
      --batch 4 --n-steps 3
done
# disk segment/grad/Adam stores are regenerable scratch — free them
rm -rf "$MRT/results/scale_cpu_large/_work"
echo "Done: $(date)"
