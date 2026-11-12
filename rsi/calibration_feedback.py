#!/usr/bin/env python3
"""Build small-model executability feedback cards from training calibration predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _retry_dimensions(raw: object) -> set[str]:
    try:
        answers = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()
    result = set()
    for key in answers if isinstance(answers, dict) else []:
        if "_retry_" in key:
            result.add(key.split("_retry_", 1)[1])
    return result


def _reasoning_risk(dimension: dict) -> str:
    text = " ".join(str(dimension.get(field, "")) for field in (
        "name", "what_to_look_for", "detection_heuristic"
    )).lower()
    high = (
        "overall", "entire", "narrative", "emotion", "empathy", "intent",
        "background knowledge", "social", "最终", "整体", "叙事", "共情",
        "意图", "背景知识", "心理", "主题价值",
    )
    medium = (
        "later", "payoff", "completion", "progression", "before and after",
        "后续", "兑现", "完成", "推进", "前后",
    )
    if any(token in text for token in high):
        return "high"
    if any(token in text for token in medium) or float(dimension.get("window_seconds", 0)) > 8:
        return "medium"
    return "low"


def build_feedback_cards(
    detail_path: str | Path, rubric_path: str | Path,
    midpoint_limit: float = 0.80, retry_limit: float = 0.20,
    positive_srcc: float = 0.03, negative_srcc: float = -0.03,
) -> tuple[dict, pd.DataFrame]:
    frame = pd.read_csv(detail_path, dtype={"Id": str})
    rubric = json.loads(Path(rubric_path).read_text(encoding="utf-8"))
    valid = frame.dropna(subset=["ECR", "ECR_gt"]).copy()
    total = len(frame)
    valid_count = len(valid)
    retries = [_retry_dimensions(value) for value in valid.get(
        "group_answers_json", pd.Series(["{}"] * valid_count)
    )]
    cards = []
    for dimension in rubric["dimensions"]:
        key = dimension["key"]
        column = f"{key}_score"
        source = (
            valid[column] if column in valid
            else pd.Series(np.nan, index=valid.index, dtype=float)
        )
        values = pd.to_numeric(source, errors="coerce")
        observed = values.notna()
        observed_values = values[observed]
        truth = valid.loc[observed, "ECR_gt"]
        unique = int(observed_values.nunique())
        srcc = (
            float(spearmanr(observed_values, truth).statistic)
            if unique > 1 and len(observed_values) > 2 else None
        )
        midpoint_rate = float(np.isclose(observed_values, 0.5).mean()) if observed.any() else 1.0
        retry_rate = sum(key in item for item in retries) / max(valid_count, 1)
        success_rate = len(observed_values) / max(total, 1)
        risk = _reasoning_risk(dimension)
        binary_gate = unique == 2 and srcc is not None and srcc >= positive_srcc

        reasons = []
        if success_rate < 0.98:
            reasons.append("format_or_missing_rate_above_2pct")
        if retry_rate > retry_limit:
            reasons.append("retry_rate_too_high")
        if midpoint_rate > midpoint_limit:
            reasons.append("midpoint_collapse")
        if unique < 2:
            reasons.append("invariant")
        if srcc is not None and srcc <= negative_srcc:
            reasons.append("negative_ecr_direction")
        if risk == "high":
            reasons.append("small_model_reasoning_risk_high")

        if "negative_ecr_direction" in reasons or "invariant" in reasons:
            decision = "delete"
        elif reasons:
            decision = "split_or_simplify"
        elif binary_gate:
            decision = "retain_as_binary_gate"
        elif srcc is not None and srcc >= positive_srcc:
            decision = "retain"
        else:
            decision = "weak_holdout"
        cards.append({
            "key": key,
            "weight": float(dimension["weight_0_1"]),
            "n_total": total,
            "n_valid_samples": valid_count,
            "n_observed": int(observed.sum()),
            "format_success_rate": success_rate,
            "retry_rate": retry_rate,
            "n_unique": unique,
            "std": float(observed_values.std(ddof=0)) if observed.any() else 0.0,
            "midpoint_rate": midpoint_rate,
            "srcc_train_calibration": srcc,
            "small_model_reasoning_risk": risk,
            "binary_gate": binary_gate,
            "decision": decision,
            "reasons": reasons,
        })
    payload = {
        "schema_version": "rsi-small-model-feedback-v1",
        "source_detail": str(Path(detail_path).resolve()),
        "source_rubric": str(Path(rubric_path).resolve()),
        "thresholds": {
            "format_success_min": 0.98,
            "retry_rate_max": retry_limit,
            "midpoint_rate_max": midpoint_limit,
            "positive_srcc": positive_srcc,
            "negative_srcc": negative_srcc,
        },
        "global": {
            "n_total": total,
            "n_valid": valid_count,
            "failure_rate": 1 - valid_count / max(total, 1),
        },
        "dimension_cards": cards,
    }
    return payload, pd.DataFrame(cards)


def advisor_text(payload: dict) -> str:
    return (
        "Round 3 small-model calibration feedback follows. This is executable evidence, "
        "not optional prose. Delete dimensions marked delete; split or simplify dimensions "
        "marked split_or_simplify into one-hop directly observable states. Preserve stable "
        "binary gates as gates instead of inventing continuous precision. Do not introduce "
        "dimensions requiring stronger textual, psychological, or long-range narrative "
        "reasoning than the local predictor demonstrated. Return 6-12 complementary "
        "dimensions and keep all external-tool prohibitions.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", required=True)
    parser.add_argument("--rubric", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--midpoint-limit", type=float, default=0.80)
    parser.add_argument("--retry-limit", type=float, default=0.20)
    args = parser.parse_args()
    payload, table = build_feedback_cards(
        args.detail, args.rubric, args.midpoint_limit, args.retry_limit
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "small_model_feedback.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    table.to_csv(output / "small_model_feedback.csv", index=False)
    (output / "small_model_advisor.txt").write_text(
        advisor_text(payload), encoding="utf-8"
    )
    print(table[["key", "format_success_rate", "retry_rate", "midpoint_rate",
                 "srcc_train_calibration", "small_model_reasoning_risk",
                 "decision"]].to_string(index=False))


if __name__ == "__main__":
    main()
