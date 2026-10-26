#!/bin/sh
# Stage-1 ECR matrix: temporal windows + two continuous scoring methods.
#
# Result matrix (32 runs):
#   Direct full rubric:  5D/12D x fewshot/no-fewshot x continuous/ordinal-prob = 8
#   Divided + weighted Python: same 8 dimension sets = 8
#   Divided + strict Python:   same 8 dimension sets = 8
#   Divided + LLM:             same 8 dimension sets = 8
#
# Re-running this script resumes successful samples. No model fine-tuning occurs.

set -eu
cd "$(dirname "$0")"

OSS_WEIGHTS=${OSS_WEIGHTS:-/root/workspace/cvuaggk7v38s73dgjft0/videollama2weights}
PY=${PY:-/root/anaconda3/envs/videollama2/bin/python}
JSON=${JSON:-test_500.json}
MODAL=${MODAL:-av}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p results

run_direct() {
  rubric="$1"
  fewshot_name="$2"
  fewshot_flag="$3"
  score_source="$4"
  short_source="$5"
  output="results/stage1_${rubric}d_${fewshot_name}_direct_${short_source}.csv"
  echo "===== DIRECT: ${rubric}D ${fewshot_name} ${score_source} ====="
  "$PY" infer_ecr_stage1.py \
    --model-path "$OSS_WEIGHTS" --modal-type "$MODAL" --json-file "$JSON" --resume \
    --rubric-size "$rubric" --flow direct --score-source "$score_source" \
    $fewshot_flag --save-csv "$output"
}

run_divide_python() {
  rubric="$1"
  fewshot_name="$2"
  fewshot_flag="$3"
  score_source="$4"
  short_source="$5"
  output="results/stage1_${rubric}d_${fewshot_name}_divide_${short_source}_python_strict.csv"
  echo "===== DIVIDE+PYTHON: ${rubric}D ${fewshot_name} ${score_source} ====="
  "$PY" infer_ecr_stage1.py \
    --model-path "$OSS_WEIGHTS" --modal-type "$MODAL" --json-file "$JSON" --resume \
    --rubric-size "$rubric" --flow divide --score-source "$score_source" \
    --aggregate python --python-formula strict $fewshot_flag --save-csv "$output"
}

run_divide_python_weighted_reuse() {
  rubric="$1"
  fewshot_name="$2"
  fewshot_flag="$3"
  score_source="$4"
  short_source="$5"
  source="results/stage1_${rubric}d_${fewshot_name}_divide_${short_source}_python_strict_detail.csv"
  output="results/stage1_${rubric}d_${fewshot_name}_divide_${short_source}_python_weighted.csv"
  echo "===== DIVIDE+PYTHON WEIGHTED: ${rubric}D ${fewshot_name} ${score_source} ====="
  "$PY" infer_ecr_stage1.py \
    --model-path "$OSS_WEIGHTS" --modal-type "$MODAL" --json-file "$JSON" --resume \
    --rubric-size "$rubric" --flow divide --score-source "$score_source" \
    --aggregate python --python-formula weighted --dimensions-from "$source" \
    $fewshot_flag --save-csv "$output"
}

run_divide_llm_reuse() {
  rubric="$1"
  fewshot_name="$2"
  fewshot_flag="$3"
  score_source="$4"
  short_source="$5"
  source="results/stage1_${rubric}d_${fewshot_name}_divide_${short_source}_python_strict_detail.csv"
  output="results/stage1_${rubric}d_${fewshot_name}_divide_${short_source}_llm.csv"
  echo "===== DIVIDE+LLM: ${rubric}D ${fewshot_name} ${score_source} ====="
  "$PY" infer_ecr_stage1.py \
    --model-path "$OSS_WEIGHTS" --modal-type "$MODAL" --json-file "$JSON" --resume \
    --rubric-size "$rubric" --flow divide --score-source "$score_source" \
    --aggregate llm --dimensions-from "$source" $fewshot_flag --save-csv "$output"
}

# ---------------------------------------------------------------------------
# 1. Full rubric in one multimodal call. The input is the first five seconds.
# ---------------------------------------------------------------------------
for rubric in 5 12; do
  run_direct "$rubric" fewshot "" continuous continuous
  run_direct "$rubric" nofewshot "--no-fewshot" continuous continuous
  run_direct "$rubric" fewshot "" ordinal-prob ordinalprob
  run_direct "$rubric" nofewshot "--no-fewshot" ordinal-prob ordinalprob
done

# ---------------------------------------------------------------------------
# 2. Per-dimension temporal windows + deterministic strict Python aggregation.
#    These detail CSVs become the reusable dimension-feature source.
# ---------------------------------------------------------------------------
for rubric in 5 12; do
  run_divide_python "$rubric" fewshot "" continuous continuous
  run_divide_python "$rubric" nofewshot "--no-fewshot" continuous continuous
  run_divide_python "$rubric" fewshot "" ordinal-prob ordinalprob
  run_divide_python "$rubric" nofewshot "--no-fewshot" ordinal-prob ordinalprob
done

# ---------------------------------------------------------------------------
# 3. Reuse the same dimensions for a weighted-only Python baseline.
#    No model dimension inference is repeated here.
# ---------------------------------------------------------------------------
for rubric in 5 12; do
  run_divide_python_weighted_reuse "$rubric" fewshot "" continuous continuous
  run_divide_python_weighted_reuse "$rubric" nofewshot "--no-fewshot" continuous continuous
  run_divide_python_weighted_reuse "$rubric" fewshot "" ordinal-prob ordinalprob
  run_divide_python_weighted_reuse "$rubric" nofewshot "--no-fewshot" ordinal-prob ordinalprob
done

# ---------------------------------------------------------------------------
# 4. Reuse exactly the same dimensions and add one final LLM aggregation call.
#    No dimension inference is repeated here.
# ---------------------------------------------------------------------------
for rubric in 5 12; do
  run_divide_llm_reuse "$rubric" fewshot "" continuous continuous
  run_divide_llm_reuse "$rubric" nofewshot "--no-fewshot" continuous continuous
  run_divide_llm_reuse "$rubric" fewshot "" ordinal-prob ordinalprob
  run_divide_llm_reuse "$rubric" nofewshot "--no-fewshot" ordinal-prob ordinalprob
done

"$PY" compare_stage1.py --output results/comparison_stage1.csv

echo "Stage-1 matrix completed: results/comparison_stage1.csv"
