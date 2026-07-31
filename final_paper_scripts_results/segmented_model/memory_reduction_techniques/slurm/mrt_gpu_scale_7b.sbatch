#!/bin/bash
#SBATCH --job-name=mrt_gpu_scale_7b
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=08:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_gpu_scale_7b_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_gpu_scale_7b_%j.err
# Site-independent: export PROJECT_ROOT=<path to this repository> and, if the
# interpreter is not on PATH, PYTHON=<interpreter> before submitting. Submit from
# the repository root so the relative --output/--error log paths resolve.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
MRT=final_paper_scripts_results/segmented_model/memory_reduction_techniques
echo "Node: $(hostname)  Start: $(date)"
# Scale experiment, ~6.9B params, partition fixed 8x2x2x8 (see scale_cost.py).
# full_train: fp32 AdamW states alone = ~110 GB -> OOM on any A100; recorded.
# full_infer: ~27.4 GB weights + activations -> near the 40 GB edge; measured
# either way (a caught OOM is also a result). seg_*: one segment resident
# (max ~268 MB); host-RAM store needs ~110 GB -> --mem=240G. n-steps 2 for time.
for RUN in full_train full_infer seg_train seg_infer; do
  echo "===== $RUN  $(date) ====="
  "$PYTHON" "$MRT/scale_cost.py" --preset xxl7b_8x2x2x8 --device cuda --run "$RUN" \
      --batch 4 --n-steps 2
done
echo "Done: $(date)"
