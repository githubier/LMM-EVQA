#!/bin/sh
# Experiment 2 ablation: original discrete 0/0.5/1 scores WITH temporal windows.
# This isolates temporal cropping without either soft-score method.

set -eu
cd "$(dirname "$0")"
OSS_WEIGHTS=${OSS_WEIGHTS:-/root/workspace/cvuaggk7v38s73dgjft0/videollama2weights}
PY=${PY:-/root/anaconda3/envs/videollama2/bin/python}
JSON=${JSON:-test_500.json}
MODAL=${MODAL:-av}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results

run_case() {
  rubric="$1"; fs="$2"; fsflag="$3"
  prefix="results/experiment2_discrete_temporal_${rubric}d_${fs}"

  # Complete rubric in one call, using the first five seconds.
  "$PY" infer_ecr_stage1.py --model-path "$OSS_WEIGHTS" --modal-type "$MODAL" \
    --json-file "$JSON" --resume --temporal-mode windowed --rubric-size "$rubric" \
    --flow direct --score-source discrete $fsflag \
    --save-csv "${prefix}_direct.csv"

  # Produce discrete dimensions once and apply the strict Python formula.
  "$PY" infer_ecr_stage1.py --model-path "$OSS_WEIGHTS" --modal-type "$MODAL" \
    --json-file "$JSON" --resume --temporal-mode windowed --rubric-size "$rubric" \
    --flow divide --score-source discrete --aggregate python --python-formula strict \
    $fsflag --save-csv "${prefix}_divide_python_strict.csv"
  source="${prefix}_divide_python_strict_detail.csv"

  # Reuse identical dimensions for weighted Python and both LLM aggregation prompts.
  "$PY" infer_ecr_stage1.py --model-path "$OSS_WEIGHTS" --modal-type "$MODAL" \
    --json-file "$JSON" --resume --temporal-mode windowed --rubric-size "$rubric" \
    --flow divide --score-source discrete --aggregate python --python-formula weighted \
    --dimensions-from "$source" $fsflag \
    --save-csv "${prefix}_divide_python_weighted.csv"
  "$PY" infer_ecr_stage1.py --model-path "$OSS_WEIGHTS" --modal-type "$MODAL" \
    --json-file "$JSON" --resume --temporal-mode windowed --rubric-size "$rubric" \
    --flow divide --score-source discrete --aggregate llm \
    --dimensions-from "$source" $fsflag \
    --save-csv "${prefix}_divide_llm_nodims.csv"
  "$PY" infer_ecr_stage1.py --model-path "$OSS_WEIGHTS" --modal-type "$MODAL" \
    --json-file "$JSON" --resume --temporal-mode windowed --rubric-size "$rubric" \
    --flow divide --score-source discrete --aggregate llm --aggregate-with-dimensions \
    --dimensions-from "$source" $fsflag \
    --save-csv "${prefix}_divide_llm_dims.csv"
}

run_case 5 fewshot ""
run_case 5 nofewshot "--no-fewshot"
run_case 12 fewshot ""
run_case 12 nofewshot "--no-fewshot"

"$PY" compare_stage1.py --pattern 'results/experiment2_discrete_temporal_*_detail.csv' \
  --output results/comparison_experiment2_discrete_temporal.csv
echo "Experiment 2 completed: results/comparison_experiment2_discrete_temporal.csv"
