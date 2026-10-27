#!/usr/bin/env python3
"""Run a dynamic RSI rubric with local VideoLLaMA2 and emit auditable details."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
ZERO_SHOT = ROOT / "VideoLLaMA2-audio_visual" / "zero_shot_ecr"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "VideoLLaMA2-audio_visual"))
sys.path.insert(0, str(ZERO_SHOT))

from videollama2 import mm_infer, model_init  # noqa: E402
from infer_ecr_prompt import (  # noqa: E402
    _clean, compute_metrics, language_modeling_head, load_batch_samples,
)
from rsi.rubric import (  # noqa: E402
    build_group_prompt, load_rubric, parse_group_answer, weighted_score,
)
from rsi.strategy import load_strategy  # noqa: E402


def checkpoint(records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output)
    submission = frame[["Id", "ECR"]]
    prediction_path = output.with_name(output.stem.replace("_detail", "") + ".csv")
    temporary = prediction_path.with_suffix(prediction_path.suffix + ".tmp")
    submission.to_csv(temporary, index=False)
    temporary.replace(prediction_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--rubric", required=True)
    parser.add_argument("--strategy")
    parser.add_argument("--json-file", required=True)
    parser.add_argument("--output-detail", required=True)
    parser.add_argument("--modal-type", choices=["v", "av"], default="av")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rubric = load_rubric(args.rubric)
    strategy = load_strategy(args.strategy, rubric)
    if strategy["predictor"] != "videollama2":
        parser.error("this entry point requires strategy.predictor=videollama2")
    if args.modal_type != strategy["modal_type"]:
        parser.error("--modal-type must match strategy.modal_type")
    samples = load_batch_samples(args.json_file)
    if args.limit:
        samples = samples[:args.limit]
    output = Path(args.output_detail)
    records: list[dict] = []
    completed = set()
    if args.resume and output.exists():
        previous = pd.read_csv(output, dtype={"Id": str})
        records = previous.to_dict("records")
        completed = set(previous.loc[previous["ECR"].notna(), "Id"].astype(str))
        print(f"resume: {len(completed)} successful samples")

    print(f"loading local predictor: {args.model_path}")
    model, processor, tokenizer = model_init(args.model_path)
    if args.modal_type == "v":
        model.model.audio_tower = None
    model.eval()
    preprocess = processor["video"]

    with language_modeling_head(model), torch.no_grad():
        for sample in tqdm(samples):
            if str(sample["id"]) in completed:
                continue
            answers = {}
            dimension_scores = {}
            error = ""
            try:
                title = _clean(sample["title"])
                description = _clean(sample["description"])
                tensor_cache = {}
                for group_index, group in enumerate(strategy["call_groups"], 1):
                    group_dimensions = [
                        item for item in rubric["dimensions"] if item["key"] in group
                    ]
                    seconds = max(
                        strategy["window_overrides"].get(item["key"], item["window_seconds"])
                        for item in group_dimensions
                    )
                    if seconds not in tensor_cache:
                        tensor_cache[seconds] = preprocess(
                            sample["video"], s=0.0, e=float(seconds),
                            va=(args.modal_type == "av"),
                        )
                    prompt = (
                        strategy["prompt_prefix"]
                        + build_group_prompt(rubric, group, title, description)
                        + strategy["prompt_suffix"]
                    )
                    answer = mm_infer(
                        tensor_cache[seconds], prompt, model=model, tokenizer=tokenizer,
                        modal="video", do_sample=False, temperature=0.0,
                        max_new_tokens=args.max_new_tokens,
                    )
                    answers[f"group_{group_index}"] = answer
                    dimension_scores.update(parse_group_answer(answer, group))
                prediction = weighted_score(rubric, dimension_scores) * 100.0
            except Exception as exc:  # keep failed samples inspectable and resumable
                prediction = None
                error = f"{type(exc).__name__}: {exc}"

            record = {
                "Id": sample["id"], "ECR": prediction, "ECR_gt": sample.get("ecr"),
                "error": error, "rubric_path": str(Path(args.rubric).resolve()),
                "strategy_path": str(Path(args.strategy).resolve()) if args.strategy else "<rubric-default>",
                "predictor": strategy["predictor"], "modal_type": args.modal_type,
                "group_answers_json": json.dumps(answers, ensure_ascii=False),
            }
            for item in rubric["dimensions"]:
                key = item["key"]
                result = dimension_scores.get(key, {})
                record[f"{key}_score"] = result.get("score")
                record[f"{key}_reason"] = result.get("reason", "")
            records.append(record)
            checkpoint(records, output)

    frame = pd.DataFrame(records)
    valid = frame.dropna(subset=["ECR", "ECR_gt"])
    print(f"valid={len(valid)}/{len(frame)}, failed={frame['ECR'].isna().sum()}")
    if len(valid) > 1:
        srcc, krcc, plcc, rmse = compute_metrics(
            valid["ECR"].to_numpy(float), valid["ECR_gt"].to_numpy(float)
        )
        print(
            f"SRCC={srcc:.6f} PLCC={plcc:.6f} Final={0.6*srcc+0.4*plcc:.6f} "
            f"KRCC={krcc:.6f} RMSE={rmse:.6f}"
        )


if __name__ == "__main__":
    main()
