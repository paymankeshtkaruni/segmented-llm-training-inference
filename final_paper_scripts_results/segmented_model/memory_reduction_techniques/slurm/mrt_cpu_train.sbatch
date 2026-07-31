#!/bin/bash
#SBATCH --job-name=mrt_cpu_train
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=10:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_cpu_train_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_cpu_train_%j.err
# Site-independent: export PROJECT_ROOT=<path to this repository> and, if the
# interpreter is not on PATH, PYTHON=<interpreter> before submitting. Submit from
# the repository root so the relative --output/--error log paths resolve.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
# glibc heap tuning (constant across all rungs): cap arenas + return freed pages to the OS
# so disk-store churn does not inflate RSS via fragmentation (the free_device TECHNIQUE is
# still toggled per-rung on top of this).
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_ARENA_MAX=2
PYTHON="${PYTHON:-python}"
echo "Node: $(hostname)  Start: $(date)  MALLOC_ARENA_MAX=$MALLOC_ARENA_MAX"
# NOTE: rung r0 (all-resident + full autograd graph, 838M) is RAM-heavy and the streamed
# rungs (disk store + recompute) are the slow ones; hence the large --mem and --time.
"$PYTHON" final_paper_scripts_results/segmented_model/memory_reduction_techniques/cpu_train_cost.py \
    --preset large_8x2x2x8 --device cpu --batch 4 --n-steps 3
echo "Done: $(date)"
