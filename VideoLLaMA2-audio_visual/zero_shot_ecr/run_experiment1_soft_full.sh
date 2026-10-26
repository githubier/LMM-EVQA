#!/bin/sh
# Experiment 1 ablation: soft dimension scores WITHOUT temporal cropping.
# Every call sees the original full video and full audio.
# Covers continuous generation and low/medium/high probability expectation.

set -eu
cd "$(dirname "$0")"
OSS_WEIGHTS=${OSS_WEIGHTS:-/root/workspace/cvuaggk7v38s73dgjft0/videollama2weights}
PY=${PY:-/root/anaconda3/envs/videollama2/bin/python}
JSON=${JSON:-test_500.json}
MODAL=${MODAL:-av}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results

run_direct() {
  rubric="$1"; fs="$2"; fsflag="$3"; source="$4"; short="$5"
  "$PY" infer_ecr_stage1.py --model-path "$OSS_WEIGHTS" --modal-type "$MODAL" \
    --json-file "$JSON" --resume --temporal-mode full --rubric-size "$rubric" \
    --flow direct --score-source "$source" $fsflag \
    --save-csv "results/experiment1_soft_full_${rubric}d_${fs}_direct_${short}.csv"
}

run_dimensions() {
  rubric="$1"; fs="$2"; fsflag="$3"; source="$4"; short="$5"
  "$PY" infer_ecr_stage1.py --model-path "$OSS_WEIGHTS" --modal-type "$MODAL" \
    --json-file "$JSON" --resume --temporal-mode full --rubric-size "$rubric" \
    --flow divide --score-source "$source" --aggregate python --python-formula strict \
    $fsflag --save-csv "results/experiment1_soft_full_${rubric}d_${fs}_divide_${short}_python_strict.csv"
}

run_reuse() {
  rubric="$1"; fs="$2"; fsflag="$3"; source="$4"; short="$5"
  base="results/experiment1_soft_full_${rubric}d_${fs}_divide_${short}_python_strict"
  "$PY" infer_ecr_stage1.py --model-path "$OSS_WEIGHTS" --modal-type "$MODAL" \
    --json-file "$JSON" --resume --temporal-mode full --rubric-size "$rubric" \
    --flow divide --score-source "$source" --aggregate python --python-formula weighted \
    --dimensions-from "${base}_detail.csv" $fsflag \
    --save-csv "results/experiment1_soft_full_${rubric}d_${fs}_divide_${short}_python_weighted.csv"
  "$PY" infer_ecr_stage1.py --model-path "$OSS_WEIGHTS" --modal-type "$MODAL" \
    --json-file "$JSON" --resume --temporal-mode full --rubric-size "$rubric" \
    --flow divide --score-source "$source" --aggregate llm \
    --dimensions-from "${base}_detail.csv" $fsflag \
    --save-csv "results/experiment1_soft_full_${rubric}d_${fs}_divide_${short}_llm_nodims.csv"
  "$PY" infer_ecr_stage1.py --model-path "$OSS_WEIGHTS" --modal-type "$MODAL" \
    --json-file "$JSON" --resume --temporal-mode full --rubric-size "$rubric" \
    --flow divide --score-source "$source" --aggregate llm --aggregate-with-dimensions \
    --dimensions-from "${base}_detail.csv" $fsflag \
    --save-csv "results/experiment1_soft_full_${rubric}d_${fs}_divide_${short}_llm_dims.csv"
}

for rubric in 5 12; do
  for spec in "fewshot||continuous|continuous" "nofewshot|--no-fewshot|continuous|continuous" \
              "fewshot||ordinal-prob|ordinalprob" "nofewshot|--no-fewshot|ordinal-prob|ordinalprob"; do
    oldifs=$IFS; IFS='|'; set -- $spec; IFS=$oldifs
    fs="$1"; fsflag="$2"; source="$3"; short="$4"
    run_direct "$rubric" "$fs" "$fsflag" "$source" "$short"
    run_dimensions "$rubric" "$fs" "$fsflag" "$source" "$short"
    run_reuse "$rubric" "$fs" "$fsflag" "$source" "$short"
  done
done

"$PY" compare_stage1.py --pattern 'results/experiment1_soft_full_*_detail.csv' \
  --output results/comparison_experiment1_soft_full.csv
echo "Experiment 1 completed: results/comparison_experiment1_soft_full.csv"
