#!/bin/bash
#SBATCH --job-name=rerun_oldenv_ctrl
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=02:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/rerun_oldenv_ctrl_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/rerun_oldenv_ctrl_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
# Audit D-7 control: same script and workload as job 15115043 but under the OLD
# environment of full_model/slurm/cost.sbatch — 8 CPUs, OMP_NUM_THREADS unset,
# no MALLOC tuning. If this reproduces ~114 s / ~33 GB, the 33.0 s / 26.9 GB
# correction is proven to be environment-caused, not script-caused.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
MRT=final_paper_scripts_results/segmented_model/memory_reduction_techniques
echo "Node: $(hostname)  Start: $(date)  OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}"
"$PYTHON" "$MRT/scale_cost.py" --preset large_8x2x2x8 --device cpu --run full_train \
    --batch 4 --n-steps 3 \
    --out-dir "$MRT/results/scale_cpu_large_oldenv"
echo "Done: $(date)"
