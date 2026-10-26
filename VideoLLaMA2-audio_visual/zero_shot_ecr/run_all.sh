#!/bin/bash
# Zero-shot ECR prediction with the ORIGINAL open-source VideoLLaMA2.1-7B-AV.
# No fine-tuning, no regression head -- everything is driven by the prompt.
#
# Run from inside zero_shot_ecr/:
#     bash run_all.sh
#
# HF_HUB_OFFLINE avoids re-validating the cached SigLIP vision tower, which
# fails behind the proxy.

set -e
cd "$(dirname "$0")"

OSS_WEIGHTS=/root/workspace/cvuaggk7v38s73dgjft0/videollama2weights
FT_WEIGHTS=/workspace/LMM-EVQA/SnapUGC/videollama2_EVQA_weights_mse
PY=/root/anaconda3/envs/videollama2/bin/python
JSON=test_500.json
DISTILLED=rubric_distilled/04_reusable_prompt.md

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p results

# 0. build the 500-video test json (Id/Title/Description + ECR ground truth)
$PY prepare_dataset_500.py

# --- the Chinese rubric prompt -----------------------------------------------
# 1a. WITHOUT few-shot anchors  (best of the rubric variants)
$PY infer_ecr_prompt.py --model-path $OSS_WEIGHTS --modal-type av --json-file $JSON \
    --score-mode rubric --no-fewshot \
    --save-csv results/results_500_rubric_nofewshot.csv

# 1b. without anchors + distilled rubric prepended
$PY infer_ecr_prompt.py --model-path $OSS_WEIGHTS --modal-type av --json-file $JSON \
    --score-mode rubric --no-fewshot --rubric-file $DISTILLED \
    --save-csv results/results_500_rubric_nofewshot_rubricfile.csv

# 1c. WITH few-shot anchors (the prompt exactly as supplied)
$PY infer_ecr_prompt.py --model-path $OSS_WEIGHTS --modal-type av --json-file $JSON \
    --score-mode rubric \
    --save-csv results/results_500_rubric.csv

# 1d. with anchors + distilled rubric
$PY infer_ecr_prompt.py --model-path $OSS_WEIGHTS --modal-type av --json-file $JSON \
    --score-mode rubric --rubric-file $DISTILLED \
    --save-csv results/results_500_rubric_rubricfile.csv

# --- baselines ---------------------------------------------------------------
# 2. continuous score read off the rating-token probabilities
$PY infer_ecr_prompt.py --model-path $OSS_WEIGHTS --modal-type av --json-file $JSON \
    --score-mode soft --save-csv results/results_500_soft.csv

$PY infer_ecr_prompt.py --model-path $OSS_WEIGHTS --modal-type av --json-file $JSON \
    --score-mode soft --rubric-file $DISTILLED \
    --save-csv results/results_500_soft_rubricfile.csv

# 3. model writes a bare number
$PY infer_ecr_prompt.py --model-path $OSS_WEIGHTS --modal-type av --json-file $JSON \
    --score-mode generate --save-csv results/results_500_generate.csv

$PY infer_ecr_prompt.py --model-path $OSS_WEIGHTS --modal-type av --json-file $JSON \
    --score-mode generate --rubric-file $DISTILLED \
    --save-csv results/results_500_gen_rubricfile.csv

# --- upper-bound reference: the fine-tuned checkpoint ------------------------
# (uses the ORIGINAL repo script + regression head, not a prompt)
cd ..
$PY videollama2/test.py --model-path $FT_WEIGHTS --modal-type av \
    --json_file zero_shot_ecr/$JSON
cd zero_shot_ecr
$PY - <<'PY'
import pandas as pd, json
p = pd.read_csv('../submission_new.csv')
gt = {d['id']: d['ECR'] for d in json.load(open('test_500.json'))}
p['ECR_gt'] = p['Id'].map(gt)
p['raw_answer'] = '<regression head>'
p.dropna(subset=['ECR_gt']).to_csv('results/results_500_finetuned_detail.csv', index=False)
PY

# --- SRCC / PLCC / final_score table -----------------------------------------
$PY compare_results_500.py

# --- optional: regenerate the distilled rubric -------------------------------
# cd .. && $PY videollama2/distill_ecr_rubric.py \
#     --model-path $FT_WEIGHTS --json-file zero_shot_ecr/test_500.json \
#     --modal-type av --top-k 30 --bottom-k 30 \
#     --out-dir zero_shot_ecr/rubric_distilled
