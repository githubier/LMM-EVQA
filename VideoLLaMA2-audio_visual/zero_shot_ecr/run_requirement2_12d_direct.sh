#!/bin/sh
# Requirement 2: direct ECR generation with the complete twelve-dimension rubric.
# Runs both the supplied few-shot prompt and its no-few-shot ablation.

set -eu
cd "$(dirname "$0")"

OSS_WEIGHTS=${OSS_WEIGHTS:-/root/workspace/cvuaggk7v38s73dgjft0/videollama2weights}
PY=${PY:-/root/anaconda3/envs/videollama2/bin/python}
JSON=${JSON:-test_500.json}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p results

# Complete 12D rubric with few-shot anchors.
$PY infer_ecr_prompt.py \
  --model-path "$OSS_WEIGHTS" --modal-type av --json-file "$JSON" \
  --score-mode rubric12 --resume \
  --save-csv results/results_500_rubric12_fewshot.csv

# Complete 12D rubric without few-shot anchors.
$PY infer_ecr_prompt.py \
  --model-path "$OSS_WEIGHTS" --modal-type av --json-file "$JSON" \
  --score-mode rubric12 --no-fewshot --resume \
  --save-csv results/results_500_rubric12_nofewshot.csv

$PY compare_results_500.py \
  --output results/comparison_requirement2_12d_direct.csv \
  --runs \
  "12D direct: with fewshot=results/results_500_rubric12_fewshot_detail.csv" \
  "12D direct: no fewshot=results/results_500_rubric12_nofewshot_detail.csv"

echo "Requirement 2 completed: results/comparison_requirement2_12d_direct.csv"
