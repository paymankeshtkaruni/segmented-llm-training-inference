#!/bin/bash
#SBATCH --job-name=mrt_gpu_scale_3b
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=04:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_gpu_scale_3b_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_gpu_scale_3b_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
MRT=final_paper_scripts_results/segmented_model/memory_reduction_techniques
echo "Node: $(hostname)  Start: $(date)"
# Scale experiment, ~3.1B params, partition fixed 8x2x2x8 (see scale_cost.py).
# full_train: fp32 AdamW states alone = ~48 GB > 40 GB A100 -> the OOM is the
# baseline data point (caught + recorded, exit 0). full_infer: ~12.4 GB weights,
# expected to fit. seg_*: one segment resident (max ~105 MB), techniques all ON;
# host-RAM store needs ~52 GB (params+grads+Adam) -> --mem=140G.
for RUN in full_train full_infer seg_train seg_infer; do
  echo "===== $RUN  $(date) ====="
  "$PYTHON" "$MRT/scale_cost.py" --preset xl3b_8x2x2x8 --device cuda --run "$RUN" \
      --batch 4 --n-steps 3
done
echo "Done: $(date)"
