#!/usr/bin/env python3
"""Run a dynamic RSI rubric with local Qwen2.5-VL and auditable details."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Qwen2.5-VL"))
from vision_process import process_vision_info  # noqa: E402
from rsi.rubric import build_group_prompt, load_rubric, parse_group_answer, weighted_score  # noqa: E402
from rsi.strategy import load_strategy  # noqa: E402


def checkpoint(records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output)
    prediction = output.with_name(output.stem.replace("_detail", "") + ".csv")
    frame[["Id", "ECR"]].to_csv(prediction, index=False)


def infer(model, processor, video_path: str, end: float, prompt: str, max_tokens: int) -> str:
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video", "video": f"file://{Path(video_path).resolve()}",
                "video_start": 0.0, "video_end": float(end), "fps": 2.0,
            },
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)
    generated = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--rubric", required=True)
    parser.add_argument("--strategy")
    parser.add_argument("--json-file", required=True)
    parser.add_argument("--output-detail", required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rubric = load_rubric(args.rubric)
    strategy = load_strategy(args.strategy, rubric)
    if strategy["predictor"] != "qwen2.5-vl":
        parser.error("this entry point requires strategy.predictor=qwen2.5-vl")
    samples = json.loads(Path(args.json_file).read_text(encoding="utf-8"))
    if args.limit:
        samples = samples[:args.limit]

    output = Path(args.output_detail)
    records: list[dict] = []
    completed = set()
    if args.resume and output.exists():
        previous = pd.read_csv(output, dtype={"Id": str})
        records = previous.to_dict("records")
        completed = set(previous.loc[previous["ECR"].notna(), "Id"].astype(str))

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    for sample in tqdm(samples):
        if str(sample["id"]) in completed:
            continue
        scores = {}
        answers = {}
        error = ""
        try:
            for group_index, group in enumerate(strategy["call_groups"], 1):
                dimensions = [item for item in rubric["dimensions"] if item["key"] in group]
                seconds = max(
                    strategy["window_overrides"].get(item["key"], item["window_seconds"])
                    for item in dimensions
                )
                prompt = (
                    strategy["prompt_prefix"]
                    + build_group_prompt(
                        rubric, group, str(sample.get("title", "")),
                        str(sample.get("description", "")),
                    )
                    + strategy["prompt_suffix"]
                )
                answer = infer(
                    model, processor, sample["video"], seconds, prompt,
                    args.max_new_tokens,
                )
                answers[f"group_{group_index}"] = answer
                scores.update(parse_group_answer(answer, group))
            prediction = weighted_score(rubric, scores) * 100
        except Exception as exc:
            prediction = None
            error = f"{type(exc).__name__}: {exc}"
        record = {
            "Id": sample["id"], "ECR": prediction, "ECR_gt": sample.get("ecr"),
            "error": error, "rubric_path": str(Path(args.rubric).resolve()),
            "strategy_path": str(Path(args.strategy).resolve()) if args.strategy else "<rubric-default>",
            "predictor": "qwen2.5-vl", "group_answers_json": json.dumps(answers, ensure_ascii=False),
        }
        for item in rubric["dimensions"]:
            result = scores.get(item["key"], {})
            record[f"{item['key']}_score"] = result.get("score")
            record[f"{item['key']}_reason"] = result.get("reason", "")
        records.append(record)
        checkpoint(records, output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
