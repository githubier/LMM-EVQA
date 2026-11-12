#!/usr/bin/env python3
"""Create a predictor-safe VideoLLaMA2 strategy from a distilled rubric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsi.rubric import load_rubric, validate_predictor_safe_rubric
from rsi.strategy import normalize_strategy


def bounded_groups(groups: list[list[str]], maximum: int = 2) -> list[list[str]]:
    """Keep semantic groups but split oversized groups deterministically."""
    result = []
    for group in groups:
        result.extend(group[start:start + maximum] for start in range(0, len(group), maximum))
    return result


def build_strategy(rubric: dict) -> dict:
    if not 6 <= len(rubric["dimensions"]) <= 12:
        raise ValueError("Round 2 strategy requires 6 to 12 dimensions")
    validate_predictor_safe_rubric(rubric)
    raw = {
        "schema_version": "rsi-strategy-v1",
        "predictor": "videollama2",
        "modal_type": "av",
        "call_groups": bounded_groups(rubric["recommended_call_groups"], maximum=2),
        "window_overrides": {},
        "prompt_prefix": (
            "Round 2 predictor constraints: judge only direct visible/audible evidence. "
            "Never claim OCR, ASR, optical-flow, frame-difference, RMS, spectrum, "
            "Sobel, coordinates, confidence values, or other external measurements. "
            "Evidence absence must follow the dimension anchors and is not automatically 0.5.\n\n"
        ),
        "prompt_suffix": "",
        "aggregation": {"method": "weighted_sum"},
    }
    return normalize_strategy(raw, rubric)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rubric", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rubric = load_rubric(args.rubric)
    strategy = build_strategy(rubric)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(strategy, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
