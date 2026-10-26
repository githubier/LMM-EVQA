"""
Zero-shot / prompt-based ECR prediction with the *original open-source*
VideoLLaMA2.1-7B-AV checkpoint.

Why this file exists
--------------------
This repo's `Videollama2Qwen2ForCausalLM.forward` is hard-wired to a regression
path:

    hidden_states -> mean-pool -> lm_head_reg (MLP) -> scalar

That path only makes sense for the *fine-tuned* checkpoint
(`videollama2_EVQA_weights_mse`), which actually contains `lm_head_reg.*`
weights. The open-source checkpoint (`DAMO-NLP-SG/VideoLLaMA2.1-7B-AV`, what
`download_model_weight.py` fetches) does **not** contain `lm_head_reg`, so:

  * `lm_head_reg` gets randomly initialised -> its scalar output is noise,
  * `.generate()` also breaks, because generation calls `forward` and would
    receive a scalar instead of vocabulary logits.

So here we temporarily patch `forward` back to the standard CausalLM path
(hidden -> `lm_head` -> vocab logits) and drive the model purely by prompt.

Scoring modes
-------------
`--score-mode soft`
    One forward pass over five rating-word probabilities.

`--score-mode generate`
    The model directly generates a numeric answer.

`--score-mode rubric`
    One direct call with the original five-dimension Chinese rubric.

`--score-mode rubric5-divide`
    Five independent dimension calls followed by one LLM aggregation call.

`--score-mode rubric12`
    One direct call with the twelve-dimension Chinese rubric.

`--score-mode rubric12-divide`
    Twelve independent dimension calls followed by one LLM aggregation call.

Usage
-----
Single video:
    python videollama2/infer_ecr_prompt.py \
        --model-path /root/workspace/cvuaggk7v38s73dgjft0/videollama2weights \
        --modal-type av \
        --video-path ./assets/00000368.mp4 \
        --title "..." --description "..."

Batch (reuses the same json produced by prepare_dataset.py):
    python videollama2/infer_ecr_prompt.py \
        --model-path /root/workspace/cvuaggk7v38s73dgjft0/videollama2weights \
        --modal-type av \
        --json-file ./test.json \
        --save-csv ./submission_prompt.csv

Optional: inject a distilled rubric as extra guidance
        --rubric-file ./rubric_out/04_reusable_prompt.md
"""

import argparse
import contextlib
import json
import os
import re
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.append("./")
# this script now lives in zero_shot_ecr/, so the repo root is one level up
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from videollama2 import model_init, mm_infer
from videollama2.mm_utils import tokenizer_multimodal_token
from ecr_rubric_prompt import (
    DIMENSIONS as RUBRIC5_DIMENSIONS,
    build_aggregate_prompt as build_rubric5_aggregate_prompt,
    build_dimension_prompt as build_rubric5_dimension_prompt,
    build_rubric_prompt,
)
from ecr_rubric_12_prompt import (
    DIMENSIONS as RUBRIC12_DIMENSIONS,
    build_aggregate_prompt as build_rubric12_aggregate_prompt,
    build_dimension_prompt as build_rubric12_dimension_prompt,
    build_rubric_prompt as build_rubric12_prompt,
)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

NUMERIC_PROMPT = (
    "You are an expert short-video analyst. Estimate the Engagement "
    "Continuation Rate (ECR) of this video.\n"
    "ECR is defined as the percentage of viewers whose watch time exceeds "
    "5 seconds. It ranges from 0 to 100.\n"
    "Video title: {title}\n"
    "Video description: {description}\n"
    "Consider the first-second hook, subject clarity, visual novelty, pace, "
    "audio hook, aesthetic quality and curiosity gap.\n"
    "Answer with a single number between 0 and 100 and nothing else. "
    "Do not explain."
)

COT_PROMPT = (
    "You are an expert short-video analyst. Estimate the Engagement "
    "Continuation Rate (ECR) of this video.\n"
    "ECR is defined as the percentage of viewers whose watch time exceeds "
    "5 seconds. It ranges from 0 to 100.\n"
    "Video title: {title}\n"
    "Video description: {description}\n"
    "First reason briefly (at most 3 short sentences) about the opening hook, "
    "subject clarity, pace and audio. Then output the final answer on its own "
    "last line in exactly this format:\n"
    "ECR: <number>"
)

# Used by --score-mode soft. We do NOT generate; we read the probability the
# model assigns to each rating word at the single next-token position, then take
# a weighted average. This produces a CONTINUOUS score, which is what SROCC /
# PLCC actually need — free-form generation collapses onto round numbers like
# 50 / 75 and destroys the ranking.
SOFT_PROMPT = (
    "You are an expert short-video analyst. Judge the engagement continuation "
    "rate of this video, i.e. how likely a viewer keeps watching past "
    "5 seconds.\n"
    "Video title: {title}\n"
    "Video description: {description}\n"
    "How would you rate the engagement of this video?"
)

# rating word -> weight in [0, 1]; mirrors the wa5() scheme already used in this repo
SOFT_TOKENS = [
    (" excellent", 1.00),
    (" good", 0.75),
    (" fair", 0.50),
    (" poor", 0.25),
    (" bad", 0.00),
]


def build_prompt(title, description, mode: str, use_cot: bool, rubric: str = None,
                 use_fewshot: bool = True) -> str:
    title = _clean(title)
    description = _clean(description)
    if mode == "rubric":
        # full Chinese 5-dimension rubric; already embeds metadata + output format
        prompt = build_rubric_prompt(title, description, use_fewshot=use_fewshot)
        if rubric:
            prompt = f"{rubric}\n\n---\n\n{prompt}"
        return prompt
    if mode == "rubric12":
        # full Chinese 12-dimension rubric in one direct generation call
        prompt = build_rubric12_prompt(title, description, use_fewshot=use_fewshot)
        if rubric:
            prompt = f"{rubric}\n\n---\n\n{prompt}"
        return prompt
    if mode == "soft":
        template = SOFT_PROMPT
    else:
        template = COT_PROMPT if use_cot else NUMERIC_PROMPT
    prompt = template.format(title=title, description=description)
    if rubric:
        prompt = f"{rubric}\n\n---\n\n{prompt}"
    return prompt


def _clean(value):
    if value is None:
        return "None"
    if isinstance(value, float) and np.isnan(value):
        return "None"
    return str(value)


# --------------------------------------------------------------------------- #
# Make the model generate text again
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def language_modeling_head(model):
    """Temporarily restore the plain CausalLM forward path.

    The checkpoint keeps the original `lm_head`, so we can simply route the
    hidden states through it instead of the (randomly initialised, useless)
    `lm_head_reg` regression head.
    """
    from transformers.modeling_outputs import CausalLMOutputWithPast

    backbone = model.get_model() if hasattr(model, "get_model") else model.model
    original_forward = model.forward

    def lm_forward(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        logits = model.lm_head(hidden_states).float()
        # keep logits on the same device generate() expects
        ref = input_ids if input_ids is not None else inputs_embeds
        if ref is not None:
            logits = logits.to(ref.device)
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    model.forward = lm_forward
    try:
        yield model
    finally:
        model.forward = original_forward


# --------------------------------------------------------------------------- #
# Soft scoring: continuous score from next-token probabilities
# --------------------------------------------------------------------------- #

def resolve_soft_token_ids(tokenizer):
    """Map each rating word to a single token id."""
    ids, weights, words = [], [], []
    for word, weight in SOFT_TOKENS:
        token_ids = tokenizer(word, add_special_tokens=False).input_ids
        if not token_ids:
            continue
        ids.append(token_ids[0])
        weights.append(weight)
        words.append(word.strip())
    return ids, np.asarray(weights, dtype=np.float64), words


@torch.no_grad()
def soft_score(model, tokenizer, tensor, prompt, modal, token_ids, weights):
    """One forward pass; weighted average over the rating-word probabilities.

    Returns (score_0_100, per_word_probability_dict).
    """
    modal_token = "<audio>" if modal == "audio" else "<video>"

    message = [{"role": "user", "content": f"{modal_token}\n{prompt}"}]
    text = tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    # nudge the model into emitting the rating word at the very next position
    text = text + "The engagement of this video is"

    device = model.device
    input_ids = (
        tokenizer_multimodal_token(text, tokenizer, modal_token, return_tensors="pt")
        .unsqueeze(0)
        .long()
        .to(device)
    )
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().to(device)

    if isinstance(tensor, dict):
        images = [({k: v.half().to(device) for k, v in tensor.items()}, "video")]
    else:
        images = [(tensor.half().to(device), modal)]

    # `model.forward` is patched to the plain CausalLM path by the caller, but the
    # multimodal embedding merge lives in the subclass, so do it explicitly here.
    _, attention_mask, _, inputs_embeds, _ = model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=None,
        images=images,
    )

    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        return_dict=True,
    )
    next_token_logits = outputs.logits[0, -1, :].float()

    selected = next_token_logits[token_ids]
    probs = torch.softmax(selected, dim=-1).cpu().numpy().astype(np.float64)

    score = float(np.dot(probs, weights) * 100.0)
    return score, probs


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def parse_ecr(text: str, prefer_first: bool = False):
    """Pull a 0-100 ECR value out of a free-form model answer.

    `prefer_first` is used for rubric mode, where the answer looks like
    "ECR: 0.62 ; reason: ..." — the trailing reason often contains other
    numbers (dimension scores), so falling back to the *last* number would be
    wrong.
    """
    if not text:
        return None

    # 1. explicit "ECR: 42.5" / "ECR = 0.62"
    m = re.search(r"ECR\s*[:=]\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    if m:
        return _clip(float(m.group(1)))

    # 2. otherwise fall back to a bare number
    numbers = re.findall(r"[0-9]*\.?[0-9]+", text)
    if not numbers:
        return None
    value = float(numbers[0] if prefer_first else numbers[-1])
    return _clip(value)


def _clip(value: float) -> float:
    # a model may answer in [0, 1] instead of [0, 100]
    if 0.0 <= value <= 1.0:
        value *= 100.0
    return float(min(max(value, 0.0), 100.0))


def parse_dimension_answer(text: str):
    """Parse one dimension answer and return (score, reason).

    Dimension rubrics use the discrete scale {0, 0.5, 1}. A numeric answer is
    snapped to the nearest legal value so the downstream aggregation prompt
    always receives a valid rubric score.
    """
    if not text:
        return None, ""

    match = re.search(r"score\s*[:=]\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    if match:
        value = float(match.group(1))
    else:
        numbers = re.findall(r"(?<![\w.])(?:0(?:\.5)?|1(?:\.0)?)(?![\w.])", text)
        if not numbers:
            return None, text.strip()
        value = float(numbers[0])

    value = min((0.0, 0.5, 1.0), key=lambda candidate: abs(candidate - value))
    reason_match = re.search(r"reason\s*[:：]\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else text.strip()
    return value, reason


def _prepend_optional_rubric(prompt: str, rubric: str = None) -> str:
    return f"{rubric}\n\n---\n\n{prompt}" if rubric else prompt


@torch.no_grad()
def run_divided_inference(
    model,
    tokenizer,
    tensor,
    modal,
    title,
    description,
    dimensions,
    dimension_prompt_builder,
    aggregate_prompt_builder,
    use_fewshot,
    include_dimensions,
    rubric,
    generation_kwargs,
):
    """Run N independent dimension calls followed by one aggregation call."""
    dimension_results = {}
    for index, (key, name, weight, _) in enumerate(dimensions):
        prompt = dimension_prompt_builder(
            title, description, index, use_fewshot=use_fewshot
        )
        prompt = _prepend_optional_rubric(prompt, rubric)
        raw_answer = mm_infer(
            tensor, prompt, model=model, tokenizer=tokenizer, modal=modal,
            **generation_kwargs,
        )
        score, reason = parse_dimension_answer(raw_answer)
        if score is None:
            raise ValueError(
                f"cannot parse dimension {key!r} from answer: {raw_answer!r}"
            )
        dimension_results[key] = {
            "name": name,
            "weight": weight,
            "score": score,
            "reason": reason,
            "raw_answer": raw_answer,
        }

    aggregate_prompt = aggregate_prompt_builder(
        title,
        description,
        dimension_results,
        use_fewshot=use_fewshot,
        include_dimensions=include_dimensions,
    )
    aggregate_prompt = _prepend_optional_rubric(aggregate_prompt, rubric)
    final_answer = mm_infer(
        tensor, aggregate_prompt, model=model, tokenizer=tokenizer, modal=modal,
        **generation_kwargs,
    )
    prediction = parse_ecr(final_answer, prefer_first=True)
    weighted_base = sum(
        result["score"] * result["weight"]
        for result in dimension_results.values()
    )
    return prediction, final_answer, dimension_results, weighted_base


# --------------------------------------------------------------------------- #
# Metrics (same definition as videollama2/test.py)
# --------------------------------------------------------------------------- #

def compute_metrics(y_pred, y_true):
    import scipy.stats
    from scipy.optimize import curve_fit
    from sklearn.metrics import mean_squared_error

    def logistic_func(X, b1, b2, b3, b4, b5):
        logistic_part = 1 + np.exp(np.multiply(b2, X - b3))
        return b1 * (0.5 - np.divide(1, logistic_part)) + np.multiply(b4, X) + b5

    srcc = scipy.stats.spearmanr(y_true, y_pred)[0]
    try:
        krcc = scipy.stats.kendalltau(y_true, y_pred)[0]
    except Exception:
        krcc = scipy.stats.kendalltau(y_true, y_pred, method="asymptotic")[0]

    try:
        beta_init = [10, 0, np.mean(y_pred), 0.1, 0.1]
        popt, _ = curve_fit(logistic_func, y_pred, y_true, p0=beta_init, maxfev=int(1e8))
        y_fit = logistic_func(y_pred, *popt)
        plcc = scipy.stats.pearsonr(y_true, y_fit)[0]
        rmse = float(np.sqrt(mean_squared_error(y_true, y_fit)))
    except Exception:
        plcc = scipy.stats.pearsonr(y_true, y_pred)[0]
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    return srcc, krcc, plcc, rmse


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #

TITLE_MARK = "The title of the video is"
DESC_MARK = "the description of the video is"


def split_title_description(human_turn: str):
    """Recover (title, description) from the prompt built by prepare_dataset.py."""
    title, description = None, None
    if TITLE_MARK in human_turn:
        tail = human_turn.split(TITLE_MARK, 1)[1]
        title = tail.split(f", and {DESC_MARK}", 1)[0].strip()
    if DESC_MARK in human_turn:
        description = human_turn.split(DESC_MARK, 1)[1].strip()
    return title, description


def load_batch_samples(json_file: str):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        human_turn = item["conversations"][0]["value"]
        title, description = split_title_description(human_turn)
        samples.append(
            {
                "id": item["id"],
                "video": item["video"],
                "title": title,
                "description": description,
                "ecr": float(item["ECR"]) if "ECR" in item else None,
            }
        )
    return samples


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run(args):
    rubric = None
    if args.rubric_file:
        rubric = open(args.rubric_file, "r", encoding="utf-8").read().strip()

    print(f"[1/3] loading model from {args.model_path} ...")
    model, processor, tokenizer = model_init(args.model_path)

    if args.modal_type == "a":
        model.model.vision_tower = None
    elif args.modal_type == "v":
        model.model.audio_tower = None
    model.eval()

    preprocess = processor["audio" if args.modal_type == "a" else "video"]

    if args.video_path:
        samples = [
            {
                "id": os.path.basename(args.video_path),
                "video": args.video_path,
                "title": args.title,
                "description": args.description,
                "ecr": None,
            }
        ]
    else:
        samples = load_batch_samples(args.json_file)
        if args.limit:
            samples = samples[: args.limit]

    fewshot_note = ""
    if args.score_mode.startswith("rubric"):
        fewshot_note = f", fewshot={'off' if args.no_fewshot else 'on'}"
    if args.score_mode.endswith("-divide"):
        fewshot_note += ", aggregate_dimensions={}".format(
            "on" if args.aggregate_with_dimensions else "off"
        )
    print(f"[2/3] running prompt-based inference on {len(samples)} video(s) "
          f"[score-mode={args.score_mode}{fewshot_note}] ...")

    modal = "audio" if args.modal_type == "a" else "video"
    token_ids, weights, words = resolve_soft_token_ids(tokenizer)

    records = []
    completed_ids = set()
    detail_path = None
    if args.save_csv:
        detail_path = os.path.splitext(args.save_csv)[0] + "_detail.csv"
        if args.resume and os.path.exists(detail_path):
            previous = pd.read_csv(detail_path, dtype={"Id": str})
            previous = previous[previous["ECR"].notna()]
            records = previous.to_dict("records")
            completed_ids = set(previous["Id"].astype(str))
            print(f"resume: keeping {len(records)} successful rows from {detail_path}")

    with language_modeling_head(model):
        for sample in tqdm(samples):
            if str(sample["id"]) in completed_ids:
                continue
            answer, pred = None, None
            dimension_results = None
            weighted_base = None
            try:
                if args.modal_type == "a":
                    tensor = preprocess(sample["video"])
                else:
                    tensor = preprocess(sample["video"], va=(args.modal_type == "av"))

                if args.score_mode not in ("rubric5-divide", "rubric12-divide"):
                    prompt = build_prompt(
                        sample["title"], sample["description"],
                        args.score_mode, args.cot, rubric,
                        use_fewshot=not args.no_fewshot,
                    )

                if args.score_mode == "soft":
                    pred, probs = soft_score(
                        model, tokenizer, tensor, prompt, modal, token_ids, weights
                    )
                    answer = " ".join(
                        f"{w}={p:.4f}" for w, p in zip(words, probs)
                    )
                elif args.score_mode in ("rubric5-divide", "rubric12-divide"):
                    if args.score_mode == "rubric5-divide":
                        dimensions = RUBRIC5_DIMENSIONS
                        dimension_builder = build_rubric5_dimension_prompt
                        aggregate_builder = build_rubric5_aggregate_prompt
                    else:
                        dimensions = RUBRIC12_DIMENSIONS
                        dimension_builder = build_rubric12_dimension_prompt
                        aggregate_builder = build_rubric12_aggregate_prompt
                    pred, answer, dimension_results, weighted_base = run_divided_inference(
                        model=model,
                        tokenizer=tokenizer,
                        tensor=tensor,
                        modal=modal,
                        title=_clean(sample["title"]),
                        description=_clean(sample["description"]),
                        dimensions=dimensions,
                        dimension_prompt_builder=dimension_builder,
                        aggregate_prompt_builder=aggregate_builder,
                        use_fewshot=not args.no_fewshot,
                        include_dimensions=args.aggregate_with_dimensions,
                        rubric=rubric,
                        generation_kwargs={
                            "do_sample": args.do_sample,
                            "temperature": args.temperature,
                            "max_new_tokens": args.max_new_tokens,
                        },
                    )
                else:
                    answer = mm_infer(
                        tensor,
                        prompt,
                        model=model,
                        tokenizer=tokenizer,
                        modal=modal,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                    )
                    pred = parse_ecr(answer, prefer_first=args.score_mode.startswith("rubric"))
            except Exception as exc:  # noqa: BLE001
                print(f"  ! failed on {sample['id']}: {exc}")
                answer, pred = f"<error> {exc}", None

            record = {
                "Id": sample["id"],
                "ECR": pred,
                "raw_answer": answer,
                "ECR_gt": sample["ecr"],
            }
            if dimension_results is not None:
                record["weighted_base_before_penalty"] = weighted_base * 100.0
                for key, result in dimension_results.items():
                    record[f"{key}_score"] = result["score"]
                    record[f"{key}_reason"] = result["reason"]
                    record[f"{key}_raw_answer"] = result["raw_answer"]
            records.append(record)

            # Multi-call experiments are expensive. Persist after every video so
            # --resume can continue after interruption without recomputing it.
            if args.save_csv:
                checkpoint = pd.DataFrame(records)
                checkpoint[["Id", "ECR"]].to_csv(args.save_csv, index=False)
                checkpoint.to_csv(detail_path, index=False)

            if args.video_path or args.verbose:
                print(f"\n--- {sample['id']} ---")
                print(f"raw answer : {answer}")
                print(f"parsed ECR : {pred}")

    print("[3/3] summarising ...")
    df = pd.DataFrame(records)
    n_failed = int(df["ECR"].isna().sum())
    print(f"parsed {len(df) - n_failed}/{len(df)} answers ({n_failed} failures)")

    valid = df.dropna(subset=["ECR"])
    if len(valid) > 1 and valid["ECR_gt"].notna().all():
        srcc, krcc, plcc, rmse = compute_metrics(
            valid["ECR"].to_numpy(dtype=float), valid["ECR_gt"].to_numpy(dtype=float)
        )
        print(
            "ECR Results: SRCC={:.4f}, KRCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(
                srcc, krcc, plcc, rmse
            )
        )
        print("Challenge score (0.6*SROCC + 0.4*PLCC) = {:.4f}".format(0.6 * srcc + 0.4 * plcc))

    if args.save_csv:
        df[["Id", "ECR"]].to_csv(args.save_csv, index=False)
        df.to_csv(detail_path, index=False)
        print(f"predictions -> {args.save_csv}")
        print(f"raw answers -> {detail_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Prompt-based (zero-shot) ECR prediction with open-source VideoLLaMA2"
    )
    parser.add_argument(
        "--model-path",
        default="/root/workspace/cvuaggk7v38s73dgjft0/videollama2weights",
        help="Path to the ORIGINAL open-source VideoLLaMA2.1-7B-AV weights",
    )
    parser.add_argument("--modal-type", default="av", choices=["a", "v", "av"])

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video-path", help="Single video mode")
    group.add_argument("--json-file", help="Batch mode: json from prepare_dataset.py")

    parser.add_argument("--title", default=None)
    parser.add_argument("--description", default=None)

    parser.add_argument(
        "--score-mode",
        default="soft",
        choices=["soft", "generate", "rubric", "rubric5-divide", "rubric12", "rubric12-divide"],
        help="soft/generate = existing baselines; rubric = direct 5-dimension prompt; "
             "rubric5-divide = five dimension calls plus one aggregation call; "
             "rubric12 = direct 12-dimension prompt; rubric12-divide = twelve "
             "dimension calls plus one aggregation call",
    )

    parser.add_argument(
        "--no-fewshot",
        action="store_true",
        help="All rubric modes: remove few-shot anchors from both dimension and "
             "aggregation calls.",
    )

    parser.add_argument(
        "--aggregate-with-dimensions",
        action="store_true",
        help="Divide modes only: repeat all dimension definitions in the final "
             "aggregation call (cases c/d). Without this flag, the final call "
             "receives scores and formula only (cases a/b).",
    )

    parser.add_argument("--rubric-file", default=None, help="Optional rubric markdown to prepend")
    parser.add_argument("--cot", action="store_true", help="Ask for brief reasoning before the number (generate mode only)")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--save-csv", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="With --save-csv, retain successful rows from the existing detail "
             "CSV and process only unfinished/failed samples.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N samples")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.cot and args.max_new_tokens < 128:
        args.max_new_tokens = 192
    if args.score_mode.startswith("rubric") and args.max_new_tokens < 128:
        # dimension calls need a short reason; aggregate calls emit ECR + reason
        args.max_new_tokens = 256

    if args.video_path and not os.path.exists(args.video_path):
        print(f"Error: video file {args.video_path} does not exist!")
        sys.exit(1)

    with torch.no_grad():
        run(args)


if __name__ == "__main__":
    main()
