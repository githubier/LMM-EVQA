#!/bin/sh
# New ECR experiments requested after the original zero-shot runs:
#   - 5D divide-and-conquer: cases a/b/c/d (5 dimension calls + 1 aggregate)
#   - 12D direct: with/without few-shot (1 call)
#   - 12D divide-and-conquer: cases a/b/c/d (12 calls + 1 aggregate)
#
# Run from any directory:
#   sh LMM-EVQA/VideoLLaMA2-audio_visual/zero_shot_ecr/run_divide_and_12d.sh
#
# Every run writes a checkpoint after each video. Re-running this script keeps
# successful rows and retries unfinished/failed rows because --resume is set.

set -e
cd "$(dirname "$0")"

OSS_WEIGHTS=${OSS_WEIGHTS:-/root/workspace/cvuaggk7v38s73dgjft0/videollama2weights}
PY=${PY:-/root/anaconda3/envs/videollama2/bin/python}
JSON=${JSON:-test_500.json}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p results

COMMON="--model-path $OSS_WEIGHTS --modal-type av --json-file $JSON --resume"

# 5D case a: dimension calls WITH few-shot; aggregate WITH few-shot, no definitions.
$PY infer_ecr_prompt.py $COMMON --score-mode rubric5-divide \
  --save-csv results/results_500_rubric5_divide_a_fewshot.csv

# 5D case b: no few-shot in either stage; aggregate has no definitions.
$PY infer_ecr_prompt.py $COMMON --score-mode rubric5-divide --no-fewshot \
  --save-csv results/results_500_rubric5_divide_b_nofewshot.csv

# 5D case c: WITH few-shot; aggregate repeats all five definitions.
$PY infer_ecr_prompt.py $COMMON --score-mode rubric5-divide --aggregate-with-dimensions \
  --save-csv results/results_500_rubric5_divide_c_fewshot_dims.csv

# 5D case d: no few-shot; aggregate repeats all five definitions.
$PY infer_ecr_prompt.py $COMMON --score-mode rubric5-divide --no-fewshot --aggregate-with-dimensions \
  --save-csv results/results_500_rubric5_divide_d_nofewshot_dims.csv

# 12D direct calls: original structure, once with and once without few-shot.
$PY infer_ecr_prompt.py $COMMON --score-mode rubric12 \
  --save-csv results/results_500_rubric12_fewshot.csv
$PY infer_ecr_prompt.py $COMMON --score-mode rubric12 --no-fewshot \
  --save-csv results/results_500_rubric12_nofewshot.csv

# 12D cases a-d, following the same switches as the 5D experiments.
$PY infer_ecr_prompt.py $COMMON --score-mode rubric12-divide \
  --save-csv results/results_500_rubric12_divide_a_fewshot.csv
$PY infer_ecr_prompt.py $COMMON --score-mode rubric12-divide --no-fewshot \
  --save-csv results/results_500_rubric12_divide_b_nofewshot.csv
$PY infer_ecr_prompt.py $COMMON --score-mode rubric12-divide --aggregate-with-dimensions \
  --save-csv results/results_500_rubric12_divide_c_fewshot_dims.csv
$PY infer_ecr_prompt.py $COMMON --score-mode rubric12-divide --no-fewshot --aggregate-with-dimensions \
  --save-csv results/results_500_rubric12_divide_d_nofewshot_dims.csv

$PY compare_results_500.py
