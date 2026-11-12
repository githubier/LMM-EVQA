#!/usr/bin/env python3
"""Run the trained VideoLLaMA2 regression head on an RSI manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "VideoLLaMA2-audio_visual"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODEL_ROOT))

from videollama2 import model_init  # noqa: E402
from videollama2.mm_utils import tokenizer_multimodal_token  # noqa: E402
from rsi.sample_manifest import load_rsi_samples  # noqa: E402


def build_prompt(title: object, description: object) -> str:
    title = "None" if title is None else str(title)
    description = "None" if description is None else str(description)
    return (
        "<video>\nHow would you judge the engagement continuation rate of the "
        "given content, where engagement continuation rate represents the "
        "probability of watch time exceeding 5 seconds. The title of the video "
        f"is {title}, and the description of the video is {description}"
    )


def regression_input(prompt: str, tokenizer) -> dict[str, torch.Tensor | list]:
    message = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "The engagement continuation rate of the video."},
    ]
    rendered = tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=False
    )
    ids = tokenizer_multimodal_token(
        rendered, tokenizer, "<video>", return_tensors="pt"
    )
    ids = ids.unsqueeze(0)
    return {
        "input_ids": ids,
        "attention_mask": ids.ne(tokenizer.pad_token_id),
    }


def parse_regression_output(output: object) -> float:
    value = output[0] if isinstance(output, (tuple, list)) else output
    if hasattr(value, "detach"):
        value = value.detach().float().reshape(-1)[0].item()
    value = float(value)
    # Training labels in this repository are ECR * 100. Keep a guard for a
    # checkpoint trained on [0, 1] labels, but never permit invalid output.
    if -1.0 <= value <= 1.0:
        value *= 100.0
    return float(min(max(value, 0.0), 100.0))


def checkpoint(records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output)
    submission = frame[["Id", "ECR"]]
    prediction = output.with_name(output.stem.replace("_detail", "") + ".csv")
    temporary = prediction.with_suffix(prediction.suffix + ".tmp")
    submission.to_csv(temporary, index=False)
    temporary.replace(prediction)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--json-file", required=True)
    parser.add_argument("--output-detail", required=True)
    parser.add_argument("--modal-type", choices=["v", "av"], default="av")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    samples = load_rsi_samples(args.json_file)
    if args.limit:
        samples = samples[: args.limit]
    output = Path(args.output_detail)
    records: list[dict] = []
    completed: set[str] = set()
    if args.resume and output.exists():
        previous = pd.read_csv(output, dtype={"Id": str})
        successful = previous.loc[previous["ECR"].notna()]
        records = successful.to_dict("records")
        completed = set(successful["Id"].astype(str))
        print(f"resume: keeping {len(completed)} successful rows")

    print(f"loading trained regression predictor: {args.model_path}", flush=True)
    model, processor, tokenizer = model_init(args.model_path)
    model.get_tokenizer(tokenizer)
    if args.modal_type == "v":
        model.model.audio_tower = None
    model.eval()
    preprocess = processor["video"]
    device = next(model.parameters()).device

    with torch.no_grad():
        for sample in tqdm(samples):
            sample_id = str(sample["id"])
            if sample_id in completed:
                continue
            prediction = None
            error = ""
            try:
                tensor = preprocess(
                    sample["video"], va=(args.modal_type == "av")
                )
                inputs = regression_input(
                    build_prompt(sample.get("title"), sample.get("description")),
                    tokenizer,
                )
                inputs["input_ids"] = inputs["input_ids"].to(device)
                inputs["attention_mask"] = inputs["attention_mask"].to(device)
                images = {
                    key: value.half().to(device) for key, value in tensor.items()
                }
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    images=[(images, "video")],
                    return_dict=False,
                )
                prediction = parse_regression_output(outputs)
            except Exception as exc:  # retain failed rows for --resume
                error = f"{type(exc).__name__}: {exc}"

            record = {
                "Id": sample["id"],
                "ECR": prediction,
                "ECR_gt": sample.get("ecr"),
                "error": error,
                "predictor": "videollama2_finetuned_regression",
                "modal_type": args.modal_type,
                "raw_answer": "<regression head>" if prediction is not None else "",
            }
            records.append(record)
            checkpoint(records, output)

    frame = pd.DataFrame(records)
    valid = frame.dropna(subset=["ECR"])
    print(f"valid={len(valid)}/{len(frame)}, failed={frame['ECR'].isna().sum()}")


if __name__ == "__main__":
    main()
