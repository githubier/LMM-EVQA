#!/bin/sh
# Requirement 1: five-dimension divide-and-conquer experiments a/b/c/d.
# Each video uses five independent dimension calls plus one final LLM call.

set -eu
cd "$(dirname "$0")"

OSS_WEIGHTS=${OSS_WEIGHTS:-/root/workspace/cvuaggk7v38s73dgjft0/videollama2weights}
PY=${PY:-/root/anaconda3/envs/videollama2/bin/python}
JSON=${JSON:-test_500.json}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p results

# a: dimensions with few-shot; aggregation gets scores + formula + few-shot.
$PY infer_ecr_prompt.py \
  --model-path "$OSS_WEIGHTS" --modal-type av --json-file "$JSON" \
  --score-mode rubric5-divide --resume \
  --save-csv results/results_500_rubric5_divide_a_fewshot.csv

# b: no few-shot in dimension or aggregation calls.
$PY infer_ecr_prompt.py \
  --model-path "$OSS_WEIGHTS" --modal-type av --json-file "$JSON" \
  --score-mode rubric5-divide --no-fewshot --resume \
  --save-csv results/results_500_rubric5_divide_b_nofewshot.csv

# c: with few-shot; aggregation also repeats all five dimension definitions.
$PY infer_ecr_prompt.py \
  --model-path "$OSS_WEIGHTS" --modal-type av --json-file "$JSON" \
  --score-mode rubric5-divide --aggregate-with-dimensions --resume \
  --save-csv results/results_500_rubric5_divide_c_fewshot_dims.csv

# d: no few-shot; aggregation also repeats all five dimension definitions.
$PY infer_ecr_prompt.py \
  --model-path "$OSS_WEIGHTS" --modal-type av --json-file "$JSON" \
  --score-mode rubric5-divide --no-fewshot --aggregate-with-dimensions --resume \
  --save-csv results/results_500_rubric5_divide_d_nofewshot_dims.csv

$PY compare_results_500.py \
  --output results/comparison_requirement1_5d_divide.csv \
  --runs \
  "5D divide a: fewshot, scores+formula=results/results_500_rubric5_divide_a_fewshot_detail.csv" \
  "5D divide b: no fewshot, scores+formula=results/results_500_rubric5_divide_b_nofewshot_detail.csv" \
  "5D divide c: fewshot, dimensions+scores+formula=results/results_500_rubric5_divide_c_fewshot_dims_detail.csv" \
  "5D divide d: no fewshot, dimensions+scores+formula=results/results_500_rubric5_divide_d_nofewshot_dims_detail.csv"

echo "Requirement 1 completed: results/comparison_requirement1_5d_divide.csv"
