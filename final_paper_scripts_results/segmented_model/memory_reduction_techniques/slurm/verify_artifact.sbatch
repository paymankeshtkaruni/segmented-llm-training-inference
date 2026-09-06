#!/bin/bash
#SBATCH --job-name=verify_artifact
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=01:30:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/verify_artifact_%j.out
set -uo pipefail
PUB=${PROJECT_PARENT}/segmented-llm-training-inference
PY="${PYTHON:-python}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
CL=$TMPDIR/artifact_clone
echo "=== 1. fresh clone ==="
git clone -q --depth 1 "file://$PUB" "$CL" && echo clone-ok || exit 1
cd "$CL"
export PROJECT_ROOT="$CL" PYTHON="$PY"
echo "=== 2. test suite ==="
PYTHONPATH=src "$PY" -m pytest tests/ -q 2>&1 | tail -2
echo "=== 3. regenerate summaries + cost model + sustained report ==="
cd final_paper_scripts_results/segmented_model/memory_reduction_techniques
"$PY" build_exp_summaries.py && "$PY" eval_cost_laws.py && "$PY" sustained_report.py
echo "=== 4. regenerate figures ==="
cd ../paper_jsa && "$PY" paper_v4_figures.py
echo "=== 5. diff verdict (JSON outputs must be byte-identical; PDFs excluded) ==="
cd "$CL"
CHANGED=$(git status --porcelain | grep -v '\.pdf$' || true)
if [ -z "$CHANGED" ]; then echo "VERDICT: PASS — all regenerated JSONs byte-identical"; else echo "VERDICT: DIFFS FOUND:"; echo "$CHANGED"; git diff --stat | tail -5; fi
echo "=== 6. smoke train (tiny, cpu) ==="
PYTHONPATH=src timeout 900 "$PY" scripts/train_segmented.py --epochs 1 --batch-size 2 --n-layers 2 --d-model 128 --n-heads 4 --d-ff 256 --attention-segments 2 --mlp-chunks 2 --checkpoint-dir "$TMPDIR/smoke" --device cpu --max-train-steps 3 --max-val-steps 3 2>&1 | tail -2
echo "Done: $(date)"
