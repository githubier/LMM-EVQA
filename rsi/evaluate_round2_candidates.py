#!/usr/bin/env python3
"""Evaluate pre-registered deterministic aggregations for Round 2 P0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rsi.fusion import metric_row

PRIMARY_CANDIDATE = "no_two_audio"
EXCLUDED_AUDIO = {
    "opening_audio_immediate_audible_entry",
    "audio_event_turnover",
}
CROSS_CONTEXT_TEXT = [
    "audio_visual_event_coupling",
    "zero_prior_comprehensibility",
    "caption_specific_event_or_question",
]
CAPTION_ONLY = ["caption_specific_event_or_question"]


def normalized_aggregate(
    frame: pd.DataFrame, weights: dict[str, float], keys: list[str]
) -> np.ndarray:
    missing = [key for key in keys if f"{key}_score" not in frame]
    if missing:
        raise ValueError(f"detail CSV missing dimension scores: {missing}")
    selected_weights = np.asarray([weights[key] for key in keys], dtype=float)
    if selected_weights.sum() <= 0:
        raise ValueError("selected dimension weights must have a positive sum")
    values = frame[[f"{key}_score" for key in keys]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("candidate aggregation received missing/non-finite scores")
    return values @ selected_weights / selected_weights.sum() * 100.0


def evaluate_candidates(detail_path: str | Path, rubric_path: str | Path):
    frame = pd.read_csv(detail_path, dtype={"Id": str})
    valid = frame.dropna(subset=["ECR", "ECR_gt"]).copy().reset_index(drop=True)
    rubric = json.loads(Path(rubric_path).read_text(encoding="utf-8"))
    weights = {item["key"]: float(item["weight_0_1"]) for item in rubric["dimensions"]}
    all_keys = list(weights)
    required = EXCLUDED_AUDIO | set(CROSS_CONTEXT_TEXT)
    if not required.issubset(weights):
        raise ValueError(f"Round 2 rubric missing pre-registered keys: {sorted(required - set(weights))}")

    candidates = {
        "full_round2": valid["ECR"].to_numpy(float),
        PRIMARY_CANDIDATE: normalized_aggregate(
            valid, weights, [key for key in all_keys if key not in EXCLUDED_AUDIO]
        ),
        "cross_context_text": normalized_aggregate(
            valid, weights, CROSS_CONTEXT_TEXT
        ),
        "caption_only_diagnostic": normalized_aggregate(
            valid, weights, CAPTION_ONLY
        ),
    }

    detail = valid[["Id", "ECR_gt"]].rename(columns={"ECR_gt": "truth"})
    rows = []
    truth = detail["truth"].to_numpy(float)
    for name, prediction in candidates.items():
        detail[name] = prediction
        row = metric_row(name, prediction, truth)
        row["role"] = (
            "pre_registered_primary" if name == PRIMARY_CANDIDATE
            else "diagnostic_control"
        )
        rows.append(row)
    return detail, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", required=True)
    parser.add_argument("--rubric", required=True)
    parser.add_argument("--p0-sota", type=float, default=0.2241)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    detail, metrics = evaluate_candidates(args.detail, args.rubric)
    full_final = float(metrics.loc[
        metrics["candidate"] == "full_round2", "final"
    ].iloc[0])
    metrics["delta_vs_full_round2"] = metrics["final"] - full_final
    metrics["reference_p0_sota"] = args.p0_sota
    metrics["delta_vs_reference_p0_sota"] = metrics["final"] - args.p0_sota
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    detail_path = prefix.with_name(prefix.name + "_detail.csv")
    metrics_path = prefix.with_name(prefix.name + "_metrics.csv")
    primary_path = prefix.with_name(prefix.name + "_primary.csv")
    detail.to_csv(detail_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    detail[["Id", PRIMARY_CANDIDATE]].rename(
        columns={PRIMARY_CANDIDATE: "ECR"}
    ).to_csv(primary_path, index=False)
    print(metrics.to_string(index=False))
    print(f"primary candidate: {PRIMARY_CANDIDATE}")
    print(f"detail -> {detail_path}")
    print(f"metrics -> {metrics_path}")
    print(f"primary predictions -> {primary_path}")


if __name__ == "__main__":
    main()
