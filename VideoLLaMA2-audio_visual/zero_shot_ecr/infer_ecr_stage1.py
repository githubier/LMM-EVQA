"""Stage-1 zero-shot ECR experiments: temporal windows + continuous scores.

This is intentionally separate from infer_ecr_prompt.py so historical runs are
fully reproducible. It supports:
  * 5D or 12D rubrics;
  * with/without few-shot anchors;
  * direct full-rubric scoring or per-dimension divide-and-conquer;
  * generated continuous values or low/medium/high next-token probabilities;
  * LLM or deterministic Python aggregation for divided runs.
"""

import argparse
import json
import os
import re
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from videollama2 import mm_infer, model_init
from videollama2.mm_utils import tokenizer_multimodal_token
from ecr_stage1_prompts import (
    build_dimension_continuous_prompt,
    build_dimension_discrete_prompt,
    build_dimension_ordinal_prompt,
    build_direct_continuous_prompt,
    build_direct_discrete_prompt,
    build_direct_ordinal_prompt,
    build_llm_aggregate_prompt,
    get_config,
)
from infer_ecr_prompt import (
    _clean,
    compute_metrics,
    language_modeling_head,
    load_batch_samples,
    parse_dimension_answer,
    parse_ecr,
)

ORDINAL_LABELS = [(" low", 0.0), (" medium", 0.5), (" high", 1.0)]


def parse_continuous_dimension(text):
    """Parse a generated score without snapping it to discrete rubric levels."""
    if not text:
        return None, ""
    match = re.search(r"score\s*[:=]\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    if not match:
        return None, text.strip()
    value = float(match.group(1))
    if value > 1.0 and value <= 100.0:
        value /= 100.0
    value = float(np.clip(value, 0.0, 1.0))
    reason_match = re.search(r"reason\s*[:：]\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else text.strip()
    return value, reason


def resolve_ordinal_tokens(tokenizer):
    """Resolve low/medium/high and require each candidate to be one token."""
    ids, weights, labels = [], [], []
    for label, weight in ORDINAL_LABELS:
        token_ids = tokenizer(label, add_special_tokens=False).input_ids
        if len(token_ids) != 1:
            raise ValueError(
                f"ordinal label {label!r} must be one token, got ids={token_ids}"
            )
        ids.append(token_ids[0])
        weights.append(weight)
        labels.append(label.strip())
    if len(set(ids)) != len(ids):
        raise ValueError(f"ordinal labels map to duplicate token ids: {ids}")
    return ids, np.asarray(weights, dtype=np.float64), labels


@torch.no_grad()
def ordinal_probability_score(model, tokenizer, tensor, prompt, modal, token_ids, weights):
    """Return E[label weight] from next-token low/medium/high probabilities."""
    modal_token = "<audio>" if modal == "audio" else "<video>"
    message = [{"role": "user", "content": f"{modal_token}\n{prompt}"}]
    text = tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    device = model.device
    input_ids = tokenizer_multimodal_token(
        text, tokenizer, modal_token, return_tensors="pt"
    ).unsqueeze(0).long().to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().to(device)
    if isinstance(tensor, dict):
        images = [({k: v.half().to(device) for k, v in tensor.items()}, "video")]
    else:
        images = [(tensor.half().to(device), modal)]
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
    logits = outputs.logits[0, -1, token_ids].float()
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
    return float(np.dot(probabilities, weights)), probabilities


def python_aggregate(rubric_size, results):
    """Apply the deterministic rubric formula to continuous dimension scores.

    Linear weighting always uses the continuous score. For ordinal-prob scores,
    discrete penalty/bonus conditions use the highest-probability hard label,
    because an expectation of strictly positive probabilities is almost never
    exactly zero.
    """
    config = get_config(rubric_size)
    score = sum(
        results[key]["score"] * weight
        for key, _, weight, _ in config["dimensions"]
    )
    details = {"weighted_base": score}
    hard_values = [
        results[key].get("hard_score", results[key]["score"])
        for key, _, _, _ in config["dimensions"]
    ]
    if rubric_size == 5:
        penalty = 0.15 if min(hard_values) == 0.0 else 0.0
        hook_hard = results["Hook_Strength"].get("hard_score", results["Hook_Strength"]["score"])
        narrative_hard = results["Narrative_Progression"].get("hard_score", results["Narrative_Progression"]["score"])
        bonus = 0.05 if hook_hard >= 0.8 and narrative_hard >= 0.8 else 0.0
        score = np.clip(score - penalty + bonus, 0.0, 1.0)
        details.update({"penalty": penalty, "bonus": bonus})
    else:
        # Supplied weights sum to 0.9996 because of decimal rounding.
        score /= sum(weight for _, _, weight, _ in config["dimensions"])
        flags = [
            results["Audio_Visual_Sync_Impact"].get("hard_score", results["Audio_Visual_Sync_Impact"]["score"]) == 0.0,
            results["Narrative_Arc_Completion"].get("hard_score", results["Narrative_Arc_Completion"]["score"]) == 0.0,
            results["Production_Quality_Technical_Clarity"].get("hard_score", results["Production_Quality_Technical_Clarity"]["score"]) == 0.0,
        ]
        penalty = min(sum(flags) * 0.05, 0.15)
        score = np.clip(score - penalty, 0.0, 1.0)
        details.update({"penalty": penalty, "bonus": 0.0})
    return float(score), details


def preprocess_window(preprocess, video_path, modal_type, seconds):
    """Process matching [0, seconds] video and audio ranges."""
    if modal_type == "a":
        # Audio-only processor has no temporal API. Stage-1 is intended for AV;
        # reject rather than silently use full audio with a cropped-video rubric.
        raise ValueError("stage-1 temporal experiments require --modal-type v or av")
    return preprocess(video_path, s=0.0, e=float(seconds), va=(modal_type == "av"))


def preprocess_input(preprocess, video_path, modal_type, seconds, temporal_mode):
    """Preprocess either the full AV stream or matching [0, seconds] ranges."""
    if temporal_mode == "full":
        return preprocess(video_path, va=(modal_type == "av"))
    return preprocess_window(preprocess, video_path, modal_type, seconds)


def checkpoint(records, save_csv):
    if not save_csv:
        return
    detail = os.path.splitext(save_csv)[0] + "_detail.csv"
    frame = pd.DataFrame(records)
    frame[["Id", "ECR"]].to_csv(save_csv, index=False)
    frame.to_csv(detail, index=False)


def load_samples(args):
    if args.video_path:
        return [{
            "id": os.path.basename(args.video_path),
            "video": args.video_path,
            "title": args.title,
            "description": args.description,
            "ecr": None,
        }]
    samples = load_batch_samples(args.json_file)
    return samples[:args.limit] if args.limit else samples


def run(args):
    config = get_config(args.rubric_size)
    use_fewshot = not args.no_fewshot
    dimension_source = None
    if args.dimensions_from:
        source_frame = pd.read_csv(args.dimensions_from, dtype={"Id": str})
        dimension_source = source_frame.set_index("Id", drop=False)
    print(f"[1/3] loading model from {args.model_path} ...")
    model, processor, tokenizer = model_init(args.model_path)
    if args.modal_type == "v":
        model.model.audio_tower = None
    model.eval()
    preprocess = processor["video"]
    modal = "video"
    token_ids, ordinal_weights, ordinal_labels = resolve_ordinal_tokens(tokenizer)
    samples = load_samples(args)

    detail_path = os.path.splitext(args.save_csv)[0] + "_detail.csv" if args.save_csv else None
    records, completed = [], set()
    if args.resume and detail_path and os.path.exists(detail_path):
        previous = pd.read_csv(detail_path, dtype={"Id": str})
        previous = previous[previous["ECR"].notna()]
        records = previous.to_dict("records")
        completed = set(previous["Id"].astype(str))
        print(f"resume: keeping {len(records)} successful rows")

    label = f"{args.rubric_size}D/{args.flow}/{args.score_source}"
    if args.flow == "divide":
        label += f"/{args.aggregate}"
        if args.aggregate == "llm":
            label += "/dimensions={}".format(
                "on" if args.aggregate_with_dimensions else "off"
            )
    print(f"[2/3] {label}, fewshot={'off' if args.no_fewshot else 'on'}, "
          f"temporal={args.temporal_mode}")

    with language_modeling_head(model), torch.no_grad():
        for sample in tqdm(samples):
            if str(sample["id"]) in completed:
                continue
            pred, raw_answer = None, None
            dimension_results = None
            aggregation_details = {}
            try:
                title = _clean(sample["title"])
                description = _clean(sample["description"])
                tensor_cache = {}

                def tensor_for(seconds):
                    seconds = float(seconds)
                    cache_key = "full" if args.temporal_mode == "full" else seconds
                    if cache_key not in tensor_cache:
                        tensor_cache[cache_key] = preprocess_input(
                            preprocess, sample["video"], args.modal_type, seconds,
                            args.temporal_mode,
                        )
                    return tensor_cache[cache_key]

                input_scope = (
                    "完整视频" if args.temporal_mode == "full" else "视频开头5秒"
                )

                if args.flow == "direct":
                    tensor = tensor_for(5.0)
                    if args.score_source == "continuous":
                        prompt = build_direct_continuous_prompt(
                            args.rubric_size, title, description, use_fewshot,
                            input_scope=input_scope,
                        )
                        raw_answer = mm_infer(
                            tensor, prompt, model=model, tokenizer=tokenizer,
                            modal=modal, do_sample=False, temperature=0.0,
                            max_new_tokens=args.max_new_tokens,
                        )
                        pred = parse_ecr(raw_answer, prefer_first=True)
                    elif args.score_source == "discrete":
                        prompt = build_direct_discrete_prompt(
                            args.rubric_size, title, description, use_fewshot,
                            input_scope=input_scope,
                        )
                        raw_answer = mm_infer(
                            tensor, prompt, model=model, tokenizer=tokenizer,
                            modal=modal, do_sample=False, temperature=0.0,
                            max_new_tokens=args.max_new_tokens,
                        )
                        pred = parse_ecr(raw_answer, prefer_first=True)
                    else:
                        prompt = build_direct_ordinal_prompt(
                            args.rubric_size, title, description, use_fewshot,
                            input_scope=input_scope,
                        )
                        score, probabilities = ordinal_probability_score(
                            model, tokenizer, tensor, prompt, modal,
                            token_ids, ordinal_weights,
                        )
                        pred = score * 100.0
                        raw_answer = json.dumps(
                            dict(zip(ordinal_labels, probabilities.tolist())),
                            ensure_ascii=False,
                        )
                else:
                    dimension_results = {}
                    source_row = None
                    if dimension_source is not None:
                        sample_id = str(sample["id"])
                        if sample_id not in dimension_source.index:
                            raise ValueError(f"sample {sample_id} missing from --dimensions-from")
                        source_row = dimension_source.loc[sample_id]
                        if isinstance(source_row, pd.DataFrame):
                            source_row = source_row.iloc[0]
                    for index, (key, name, weight, _) in enumerate(config["dimensions"]):
                        window = config["windows"][key]
                        if source_row is not None:
                            score = float(source_row[f"{key}_score"])
                            raw = source_row.get(f"{key}_raw_answer", "<reused>")
                            item = {
                                "score": score,
                                "raw_answer": raw,
                                "reason": source_row.get(f"{key}_reason", "reused from detail CSV"),
                            }
                            probability_cols = {
                                label_name: source_row.get(f"{key}_p_{label_name}")
                                for label_name in ordinal_labels
                            }
                            if all(pd.notna(value) for value in probability_cols.values()):
                                item["probabilities"] = probability_cols
                                hard_label = max(probability_cols, key=probability_cols.get)
                                item["hard_label"] = hard_label
                                item["hard_score"] = dict(ORDINAL_LABELS)[f" {hard_label}"]
                        else:
                            tensor = tensor_for(window)
                            if args.score_source == "continuous":
                                prompt = build_dimension_continuous_prompt(
                                    args.rubric_size, title, description, index, use_fewshot,
                                    input_scope=(
                                        "完整视频" if args.temporal_mode == "full"
                                        else f"视频开头 {window:g} 秒"
                                    ),
                                )
                                answer = mm_infer(
                                    tensor, prompt, model=model, tokenizer=tokenizer,
                                    modal=modal, do_sample=False, temperature=0.0,
                                    max_new_tokens=args.max_new_tokens,
                                )
                                score, reason = parse_continuous_dimension(answer)
                                if score is None:
                                    raise ValueError(f"cannot parse {key}: {answer!r}")
                                item = {"score": score, "reason": reason, "raw_answer": answer}
                            elif args.score_source == "discrete":
                                window_text = (
                                    f"视频开头 {window:g} 秒"
                                    if args.temporal_mode == "windowed"
                                    else "完整视频"
                                )
                                prompt = build_dimension_discrete_prompt(
                                    args.rubric_size, title, description, index,
                                    use_fewshot, window_text,
                                )
                                answer = mm_infer(
                                    tensor, prompt, model=model, tokenizer=tokenizer,
                                    modal=modal, do_sample=False, temperature=0.0,
                                    max_new_tokens=args.max_new_tokens,
                                )
                                score, reason = parse_dimension_answer(answer)
                                if score is None:
                                    raise ValueError(f"cannot parse {key}: {answer!r}")
                                item = {
                                    "score": score, "hard_score": score,
                                    "reason": reason, "raw_answer": answer,
                                }
                            else:
                                prompt = build_dimension_ordinal_prompt(
                                    args.rubric_size, title, description, index, use_fewshot,
                                    input_scope=(
                                        "完整视频" if args.temporal_mode == "full"
                                        else f"视频开头 {window:g} 秒"
                                    ),
                                )
                                score, probabilities = ordinal_probability_score(
                                    model, tokenizer, tensor, prompt, modal,
                                    token_ids, ordinal_weights,
                                )
                                probs = dict(zip(ordinal_labels, probabilities.tolist()))
                                hard_index = int(np.argmax(probabilities))
                                item = {
                                    "score": score,
                                    "probabilities": probs,
                                    "hard_label": ordinal_labels[hard_index],
                                    "hard_score": float(ordinal_weights[hard_index]),
                                    "raw_answer": json.dumps(probs, ensure_ascii=False),
                                }
                        item.update({
                            "name": name,
                            "weight": weight,
                            "window_seconds": (
                                window if args.temporal_mode == "windowed" else None
                            ),
                        })
                        dimension_results[key] = item

                    python_score, aggregation_details = python_aggregate(
                        args.rubric_size, dimension_results
                    )
                    if args.aggregate == "python":
                        selected_score = (
                            aggregation_details["weighted_base"]
                            if args.python_formula == "weighted"
                            else python_score
                        )
                        pred = selected_score * 100.0
                        raw_answer = json.dumps(
                            {**aggregation_details, "selected_formula": args.python_formula},
                            ensure_ascii=False,
                        )
                    else:
                        aggregate_prompt = build_llm_aggregate_prompt(
                            args.rubric_size, title, description,
                            dimension_results, use_fewshot,
                            include_dimensions=args.aggregate_with_dimensions,
                        )
                        raw_answer = mm_infer(
                            tensor_for(5.0), aggregate_prompt,
                            model=model, tokenizer=tokenizer, modal=modal,
                            do_sample=False, temperature=0.0,
                            max_new_tokens=args.max_new_tokens,
                        )
                        pred = parse_ecr(raw_answer, prefer_first=True)
            except Exception as exc:  # noqa: BLE001
                raw_answer = f"<error> {exc}"
                pred = None
                print(f"  ! failed on {sample['id']}: {exc}")

            record = {
                "Id": sample["id"], "ECR": pred,
                "raw_answer": raw_answer, "ECR_gt": sample["ecr"],
                "rubric_size": args.rubric_size,
                "flow": args.flow, "score_source": args.score_source,
                "aggregate": args.aggregate if args.flow == "divide" else "none",
                "python_formula": (
                    args.python_formula if args.aggregate == "python" else "none"
                ),
                "fewshot": use_fewshot,
                "temporal_mode": args.temporal_mode,
                "temporal_windows": args.temporal_mode == "windowed",
                "aggregate_with_dimensions": args.aggregate_with_dimensions,
            }
            for key, value in aggregation_details.items():
                record[f"python_{key}"] = value * 100.0
            if dimension_results:
                for key, item in dimension_results.items():
                    record[f"{key}_score"] = item["score"]
                    record[f"{key}_window_seconds"] = item["window_seconds"]
                    record[f"{key}_raw_answer"] = item["raw_answer"]
                    if "hard_score" in item:
                        record[f"{key}_hard_score"] = item["hard_score"]
                        record[f"{key}_hard_label"] = item["hard_label"]
                    if "reason" in item:
                        record[f"{key}_reason"] = item["reason"]
                    if "probabilities" in item:
                        for label_name, probability in item["probabilities"].items():
                            record[f"{key}_p_{label_name}"] = probability
            records.append(record)
            checkpoint(records, args.save_csv)
            if args.verbose or args.video_path:
                print(f"{sample['id']}: ECR={pred}; {raw_answer}")

    print("[3/3] summarising ...")
    frame = pd.DataFrame(records)
    valid = frame.dropna(subset=["ECR"])
    print(f"parsed {len(valid)}/{len(frame)}; unique={valid['ECR'].nunique()}")
    if len(valid) > 1 and valid["ECR_gt"].notna().all():
        srcc, krcc, plcc, rmse = compute_metrics(
            valid["ECR"].to_numpy(float), valid["ECR_gt"].to_numpy(float)
        )
        print(f"SRCC={srcc:.4f}, KRCC={krcc:.4f}, PLCC={plcc:.4f}, "
              f"RMSE={rmse:.4f}, final={0.6*srcc+0.4*plcc:.4f}")
    checkpoint(records, args.save_csv)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/root/workspace/cvuaggk7v38s73dgjft0/videollama2weights")
    parser.add_argument("--modal-type", choices=["v", "av"], default="av")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video-path")
    source.add_argument("--json-file")
    parser.add_argument("--title")
    parser.add_argument("--description")
    parser.add_argument("--rubric-size", type=int, choices=[5, 12], required=True)
    parser.add_argument("--flow", choices=["direct", "divide"], required=True)
    parser.add_argument("--score-source", choices=["discrete", "continuous", "ordinal-prob"], required=True)
    parser.add_argument(
        "--temporal-mode", choices=["full", "windowed"], default="windowed",
        help="full = original full video/audio; windowed = dimension-specific opening range",
    )
    parser.add_argument("--aggregate", choices=["llm", "python"], default=None)
    parser.add_argument(
        "--python-formula", choices=["weighted", "strict"], default="strict",
        help="Python aggregation: weighted sum only, or full rubric rules.",
    )
    parser.add_argument(
        "--dimensions-from",
        help="Divide mode: reuse dimension scores from an existing detail CSV; "
             "typically used to compare LLM aggregation without rerunning dimensions.",
    )
    parser.add_argument(
        "--aggregate-with-dimensions", action="store_true",
        help="LLM aggregation only: repeat all complete dimension definitions.",
    )
    parser.add_argument("--no-fewshot", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--save-csv", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.flow == "divide" and args.aggregate is None:
        parser.error("--flow divide requires --aggregate llm|python")
    if args.flow == "direct" and args.aggregate is not None:
        parser.error("--aggregate applies only to --flow divide")
    if args.dimensions_from and args.flow != "divide":
        parser.error("--dimensions-from applies only to --flow divide")
    if args.dimensions_from and not os.path.exists(args.dimensions_from):
        parser.error(f"dimension detail CSV does not exist: {args.dimensions_from}")
    if args.python_formula != "strict" and args.aggregate != "python":
        parser.error("--python-formula applies only to --aggregate python")
    if args.aggregate_with_dimensions and args.aggregate != "llm":
        parser.error("--aggregate-with-dimensions applies only to --aggregate llm")
    if args.video_path and not os.path.exists(args.video_path):
        parser.error(f"video does not exist: {args.video_path}")
    with torch.no_grad():
        run(args)


if __name__ == "__main__":
    main()
