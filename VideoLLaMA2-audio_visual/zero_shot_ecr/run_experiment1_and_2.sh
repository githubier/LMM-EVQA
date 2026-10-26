#!/bin/sh
# Run experiment 1 (soft scores, full video) and experiment 2
# (discrete scores, temporal windows) sequentially.
# Both child scripts use --resume, so this wrapper is safe to rerun.

set -eu
cd "$(dirname "$0")"

PY=${PY:-/root/anaconda3/envs/videollama2/bin/python}

sh run_experiment1_soft_full.sh
sh run_experiment2_discrete_temporal.sh

"$PY" compare_stage1.py \
  --pattern 'results/experiment[12]_*_detail.csv' \
  --output results/comparison_experiment1_and_2.csv

echo "Experiments 1 and 2 completed."
echo "Combined comparison: results/comparison_experiment1_and_2.csv"
