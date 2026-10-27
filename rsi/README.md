# Interactive RSI for ECR Prediction

This directory restores the interactive Recursive Self-Improvement workflow described by the recovered RSI design and `RSI_reference/` implementations.

## Responsibility boundary

- **RSI Advisor (this conversation):** reads the previous round's rubric, strategy, detailed predictions and evaluation report; proposes one testable change and edits the next round's code/configuration.
- **Qwen3.8-Max via Mario Agent:** distills or revises the dynamic ECR rubric from a bounded number of labeled training videos.
- **VideoLLaMA2-7B locally:** default predictor. It executes dynamic call groups, windows and prompts.
- **Qwen2.5-VL locally:** optional predictor using the same rubric/strategy protocol.
- **Post-trained `F+D`:** a later experiment branch; the post-trained model remains prediction-only and is never modified by RSI.
- **You:** run the supplied heavy commands on the development machine, then make the artifacts available for the next analysis.
- **Python controller:** deterministic split, schema validation, checkpointing, prediction details, metrics and acceptance gate.

The first version uses **Correct only**. Shuffled and Constant controls are deliberately not fabricated; they can be added when a candidate improvement warrants causal verification.

## Leakage and budget policy

1. Training data is only used by dimension distillation. `groups * videos-per-group` cannot exceed `--max-training-videos`.
2. Validation is split once with a fixed seed into Feedback and Selection sets.
3. Feedback guides revisions; Selection gates them.
4. Test is not read by the RSI controller. Use it only after finalists have been selected, first on a fixed 500/1000 subset and then, if justified, at full scale.
5. Mario credentials must be environment variables. Do not write keys into source, configs, shell history or result artifacts.

## Environment

Run commands from the repository root. The local model command must use the existing VideoLLaMA2 environment (historically `/root/anaconda3/envs/videollama2/bin/python`; verify it on the current machine).

```bash
export MARIO_API_KEY='...'
export MARIO_AGENT_ID='...'
```

## Step 0: prepare fixed validation subsets once

Use validation data, not test data:

```bash
python -m rsi.prepare_splits \
  --data-csv SnapUGC/val_data.csv \
  --truth-csv SnapUGC/val_truth.csv \
  --video-dir SnapUGC/val_videos \
  --output-dir rsi_runs/splits \
  --feedback-size 500 \
  --selection-size 500 \
  --seed 20261026
```

The repository currently did not contain `SnapUGC/val_videos` when this code was restored. Point `--video-dir` to the actual extracted validation-video directory.

## Step 1: distill or revise a rubric

Round 1 example (10 Map calls × 8 uploads = 80 video uses):

```bash
python -m rsi.distill \
  --caller YOUR_USERNAME \
  --env prt-idc \
  --input-csv SnapUGC/train_data.csv \
  --video-dir /path/to/extracted/train/videos \
  --output-dir rsi_runs/round_001/distill \
  --round 1 \
  --groups 10 \
  --videos-per-group 8 \
  --max-training-videos 1000
```

For later rounds, the Advisor writes a short instruction file and normally reuses the previous rubric:

```bash
python -m rsi.distill \
  ... \
  --round 2 \
  --output-dir rsi_runs/round_002/distill \
  --previous-rubric rsi_runs/round_001/distill/final_rubric.json \
  --advisor-instructions rsi_runs/round_002/advisor_instructions.md
```

Use `--resume` after an interrupted run. Completed Map artifacts are reused. Mario receives local video uploads through the same OpenAPI flow as `RSI_reference/mario_agent.py`.

## Step 2: create or revise the prediction strategy

The distilled rubric contains default groups and windows. A separate optional JSON strategy can change them without paying for another distillation call:

```json
{
  "schema_version": "rsi-strategy-v1",
  "predictor": "videollama2",
  "modal_type": "av",
  "call_groups": [["dimension_key_1", "dimension_key_2"]],
  "window_overrides": {"dimension_key_1": 3.0},
  "prompt_prefix": "",
  "prompt_suffix": "",
  "aggregation": {"method": "weighted_sum"}
}
```

Every dimension key must occur exactly once across `call_groups`. Free-form formulas returned by Mario are retained as research metadata but never executed as Python. The v1 executable aggregation is a normalized weighted sum.

## Step 3: run local predictions

Run Feedback first:

```bash
/root/anaconda3/envs/videollama2/bin/python -m rsi.predict_videollama2 \
  --model-path /path/to/original/VideoLLaMA2.1-7B-AV \
  --rubric rsi_runs/round_001/distill/final_rubric.json \
  --strategy rsi_runs/round_001/strategy.json \
  --json-file rsi_runs/splits/validation_feedback.json \
  --output-detail rsi_runs/round_001/predictions/feedback_detail.csv \
  --limit 500 \
  --resume
```

Then run Selection with the same rubric, strategy and predictor:

```bash
/root/anaconda3/envs/videollama2/bin/python -m rsi.predict_videollama2 \
  --model-path /path/to/original/VideoLLaMA2.1-7B-AV \
  --rubric rsi_runs/round_001/distill/final_rubric.json \
  --strategy rsi_runs/round_001/strategy.json \
  --json-file rsi_runs/splits/validation_selection.json \
  --output-detail rsi_runs/round_001/predictions/selection_detail.csv \
  --limit 500 \
  --resume
```

Omit `--strategy` to use the groups and windows from the rubric. The script writes per-dimension score/reason columns and raw group answers so the Advisor can diagnose failures.

Optional Qwen2.5-VL uses the same arguments and artifacts. Set `"predictor": "qwen2.5-vl"` in the strategy and run:

```bash
/root/anaconda3/envs/qwenvl/bin/python -m rsi.predict_qwen25vl \
  --model-path /path/to/Qwen2.5-VL \
  --rubric rsi_runs/round_001/distill/final_rubric.json \
  --strategy rsi_runs/round_001/strategy_qwen.json \
  --json-file rsi_runs/splits/validation_feedback.json \
  --output-detail rsi_runs/round_001/predictions/qwen_feedback_detail.csv \
  --limit 500 \
  --resume
```

The exact Qwen environment path must be checked on the development machine.

## Step 4: evaluate and gate

The first round establishes a baseline and therefore cannot auto-accept itself. From round 2 onward:

```bash
python -m rsi.evaluate \
  --candidate-feedback rsi_runs/round_002/predictions/feedback_detail.csv \
  --candidate-selection rsi_runs/round_002/predictions/selection_detail.csv \
  --baseline-feedback rsi_runs/round_001/predictions/feedback_detail.csv \
  --baseline-selection rsi_runs/round_001/predictions/selection_detail.csv \
  --output rsi_runs/round_002/evaluation.json
```

Metrics include raw Pearson and challenge PLCC after the five-parameter logistic fit. The acceptance gate defaults to:

- Feedback Final Score strictly improves;
- Selection Final Score does not decline;
- failure rate is at most 5%;
- predictions do not collapse to a single value.

An accepted candidate becomes the next baseline. A rejected candidate is retained for diagnosis, while the prior rubric and strategy remain active.

## Step 5: return artifacts for Advisor analysis

After each command, provide or retain these files:

- `distill/final_rubric.json`, `run_manifest.json`, `provenance.json`;
- `strategy.json` when present;
- `predictions/*_detail.csv`;
- `evaluation.json`.

The next conversational round follows: **observe → diagnose → propose one hypothesis → edit rubric/strategy → run → gate → learn**.

## Verification without models

```bash
python -m unittest rsi.test_rsi
python -m compileall -q rsi
```

These tests do not call Mario, load videos or initialize model weights.
