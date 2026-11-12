#!/usr/bin/env python3
"""Generate controlled divide-and-conquer strategy variants for one rubric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsi.prepare_strategy import bounded_groups
from rsi.rubric import load_rubric, validate_predictor_safe_rubric
from rsi.strategy import normalize_strategy


def _nonempty(groups: list[list[str]]) -> list[list[str]]:
    return [group for group in groups if group]


def modality3(rubric: dict) -> list[list[str]]:
    groups = {"visual": [], "text_visual": [], "audio_visual": []}
    for item in rubric["dimensions"]:
        modality = item["modality"]
        text = f"{item['key']} {item['name']}".lower()
        if modality == "audio" or "audio" in text or "sound" in text or "speech" in text:
            target = "audio_visual"
        elif modality == "text" or any(token in text for token in ("text", "caption", "title")):
            target = "text_visual"
        else:
            target = "visual"
        groups[target].append(item["key"])
    return _nonempty([groups["visual"], groups["text_visual"], groups["audio_visual"]])


def temporal3(rubric: dict) -> list[list[str]]:
    early, middle, longer = [], [], []
    for item in rubric["dimensions"]:
        seconds = float(item["window_seconds"])
        (early if seconds <= 3 else middle if seconds <= 8 else longer).append(item["key"])
    return _nonempty([early, middle, longer])


def mechanism_groups(rubric: dict) -> list[list[str]]:
    groups = {"contract": [], "action": [], "progression": [], "context": []}
    for item in rubric["dimensions"]:
        text = " ".join((item["key"], item["name"], item["what_to_look_for"])).lower()
        if any(token in text for token in (
            "caption", "text", "question", "promise", "contract", "title",
            "字幕", "文字", "问题", "承诺",
        )):
            target = "contract"
        elif any(token in text for token in (
            "action", "motion", "subject", "event", "动作", "运动", "主体", "事件",
        )):
            target = "action"
        elif any(token in text for token in (
            "progress", "payoff", "outcome", "change", "new information",
            "推进", "兑现", "结果", "变化", "新信息",
        )):
            target = "progression"
        else:
            target = "context"
        groups[target].append(item["key"])
    return _nonempty(list(groups.values()))


def strategy_payload(rubric: dict, name: str, groups: list[list[str]], rationale: str) -> dict:
    payload = {
        "schema_version": "rsi-strategy-v1",
        "predictor": "videollama2",
        "modal_type": "av",
        "call_groups": groups,
        "window_overrides": {},
        "prompt_prefix": (
            f"Round 3 controlled divide-and-conquer variant: {name}. "
            "Judge only direct visible/audible evidence. Never claim OCR, ASR, "
            "optical-flow, frame-difference, RMS, spectrum, Sobel, coordinates, "
            "confidence values, or other external measurements.\n\n"
        ),
        "prompt_suffix": "",
        "aggregation": {"method": "weighted_sum"},
        "experiment": {"variant": name, "rationale": rationale},
    }
    return normalize_strategy(payload, rubric)


def build_variants(rubric: dict) -> dict[str, dict]:
    recommended = rubric["recommended_call_groups"]
    variants = {
        "modality3": strategy_payload(
            rubric, "modality3", modality3(rubric),
            "Historical non-RSI SOTA control: visual, text+visual, audio+visual.",
        ),
        "mechanism": strategy_payload(
            rubric, "mechanism", mechanism_groups(rubric),
            "Group by observable contract, action, progression, and context mechanism.",
        ),
        "temporal3": strategy_payload(
            rubric, "temporal3", temporal3(rubric),
            "Group by <=3s, 3-8s, and >8s observation windows.",
        ),
        "small_group": strategy_payload(
            rubric, "small_group", bounded_groups(recommended, maximum=2),
            "Round 2 control: preserve recommended semantics with at most two dimensions per call.",
        ),
    }
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rubric", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rubric = load_rubric(args.rubric)
    validate_predictor_safe_rubric(rubric)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    variants = build_variants(rubric)
    manifest = {}
    for name, strategy in variants.items():
        path = output / f"strategy_{name}.json"
        path.write_text(json.dumps(strategy, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest[name] = {
            "path": str(path.resolve()),
            "groups": strategy["call_groups"],
            "calls_per_video": len(strategy["call_groups"]),
        }
    (output / "strategy_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
