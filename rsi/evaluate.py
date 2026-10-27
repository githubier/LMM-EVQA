#!/usr/bin/env python3
"""Evaluate candidate predictions and optionally apply the RSI acceptance gate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "VideoLLaMA2-audio_visual" / "zero_shot_ecr"))
from compare_results_500 import evaluate  # noqa: E402


def metrics(path: str) -> dict:
    frame = pd.read_csv(path, dtype={"Id": str})
    valid = frame.dropna(subset=["ECR", "ECR_gt"])
    if len(valid) < 2:
        raise ValueError(f"not enough valid rows in {path}")
    result = evaluate(valid["ECR"].to_numpy(float), valid["ECR_gt"].to_numpy(float))
    result.update({
        "path": str(Path(path).resolve()), "n": len(valid),
        "n_failed": int(frame["ECR"].isna().sum()),
        "n_unique": int(valid["ECR"].nunique()),
        "pred_std": float(valid["ECR"].std()),
        "raw_Pearson": float(np.corrcoef(valid["ECR"], valid["ECR_gt"])[0, 1]),
    })
    return {key: (None if isinstance(value, float) and not math.isfinite(value) else value)
            for key, value in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-feedback", required=True)
    parser.add_argument("--candidate-selection", required=True)
    parser.add_argument("--baseline-feedback")
    parser.add_argument("--baseline-selection")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-feedback-gain", type=float, default=0.0)
    parser.add_argument("--max-selection-drop", type=float, default=0.0)
    parser.add_argument("--max-failure-rate", type=float, default=0.05)
    args = parser.parse_args()

    report = {
        "candidate": {
            "feedback": metrics(args.candidate_feedback),
            "selection": metrics(args.candidate_selection),
        },
        "controls": {"mode": "correct_only", "shuffled": None, "constant": None},
    }
    checks = []
    for split in ("feedback", "selection"):
        item = report["candidate"][split]
        failure_rate = item["n_failed"] / (item["n"] + item["n_failed"])
        checks.append({
            "name": f"{split}_failure_rate", "value": failure_rate,
            "passed": failure_rate <= args.max_failure_rate,
        })
        checks.append({
            "name": f"{split}_noncollapsed", "value": item["n_unique"],
            "passed": item["n_unique"] >= 2,
        })

    if args.baseline_feedback and args.baseline_selection:
        report["baseline"] = {
            "feedback": metrics(args.baseline_feedback),
            "selection": metrics(args.baseline_selection),
        }
        feedback_gain = (
            report["candidate"]["feedback"]["final_score"]
            - report["baseline"]["feedback"]["final_score"]
        )
        selection_gain = (
            report["candidate"]["selection"]["final_score"]
            - report["baseline"]["selection"]["final_score"]
        )
        checks.extend([
            {"name": "feedback_gain", "value": feedback_gain,
             "passed": feedback_gain > args.min_feedback_gain},
            {"name": "selection_gain", "value": selection_gain,
             "passed": selection_gain >= -args.max_selection_drop},
        ])
    else:
        report["baseline"] = None
        checks.append({
            "name": "baseline_available", "value": False, "passed": False,
            "note": "First candidate is measurable but cannot be auto-accepted without a baseline.",
        })

    report["gate"] = {
        "accepted": all(item["passed"] for item in checks), "checks": checks,
        "policy": {
            "feedback": "strict improvement",
            "selection": "no drop beyond tolerance",
            "controls": "Correct only in v1; Shuffled/Constant deferred",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["gate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
