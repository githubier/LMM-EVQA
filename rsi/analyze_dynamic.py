#!/usr/bin/env python3
"""Audit a prompt-only dynamic ECR prediction run (P0)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rsi.fusion import metric_row


def audit_dynamic(detail_path: str | Path, rubric_path: str | Path):
    frame = pd.read_csv(detail_path, dtype={"Id": str})
    rubric = json.loads(Path(rubric_path).read_text(encoding="utf-8"))
    valid = frame.dropna(subset=["ECR", "ECR_gt"]).copy()
    if len(valid) < 2:
        raise ValueError("at least two valid prompt predictions are required")

    overall = metric_row(
        "prompt_dynamic_d",
        valid["ECR"].to_numpy(float),
        valid["ECR_gt"].to_numpy(float),
    )
    overall["n_total"] = int(len(frame))
    overall["n_failed"] = int(frame["ECR"].isna().sum())
    overall["failure_rate"] = float(frame["ECR"].isna().mean())

    dimension_rows = []
    invariant_weight = 0.0
    midpoint_weight = 0.0
    for item in rubric["dimensions"]:
        key = item["key"]
        weight = float(item["weight_0_1"])
        column = f"{key}_score"
        values = pd.to_numeric(valid.get(column), errors="coerce")
        observed = values.dropna()
        unique = int(observed.nunique())
        std = float(observed.std(ddof=0)) if len(observed) else float("nan")
        midpoint_rate = float(np.isclose(observed, 0.5).mean()) if len(observed) else 1.0
        invariant = unique <= 1
        if invariant:
            invariant_weight += weight
        if midpoint_rate >= 0.8:
            midpoint_weight += weight
        reasons = valid.get(f"{key}_reason", pd.Series(dtype=str)).fillna("").astype(str)
        dimension_rows.append({
            "key": key,
            "weight": weight,
            "n_observed": int(len(observed)),
            "n_unique": unique,
            "std": std,
            "midpoint_0_5_rate": midpoint_rate,
            "invariant": invariant,
            "empty_reason_rate": float(reasons.str.strip().eq("").mean()),
        })

    retry_counts = []
    if "group_answers_json" in valid:
        for raw in valid["group_answers_json"].fillna("{}"):
            try:
                answers = json.loads(raw)
                retry_counts.append(sum("_retry_" in key for key in answers))
            except (TypeError, json.JSONDecodeError):
                retry_counts.append(0)
    else:
        retry_counts = [0] * len(valid)

    overall.update({
        "invariant_dimension_weight": float(invariant_weight),
        "midpoint_dominated_weight": float(midpoint_weight),
        "exact_midpoint_rate_all_scores": float(
            np.mean([
                np.isclose(pd.to_numeric(valid[f"{item['key']}_score"], errors="coerce"), 0.5).mean()
                for item in rubric["dimensions"]
                if f"{item['key']}_score" in valid
            ])
        ),
        "targeted_retries": int(sum(retry_counts)),
        "samples_with_retry": int(sum(count > 0 for count in retry_counts)),
    })
    return pd.DataFrame([overall]), pd.DataFrame(dimension_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", required=True)
    parser.add_argument("--rubric", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    overall, dimensions = audit_dynamic(args.detail, args.rubric)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    overall_path = prefix.with_name(prefix.name + "_metrics.csv")
    dimension_path = prefix.with_name(prefix.name + "_dimensions.csv")
    overall.to_csv(overall_path, index=False)
    dimensions.to_csv(dimension_path, index=False)
    print(overall.to_string(index=False))
    print("\nDimension diagnostics:")
    print(dimensions.sort_values(["invariant", "midpoint_0_5_rate", "weight"], ascending=False).to_string(index=False))
    print(f"metrics -> {overall_path}")
    print(f"dimensions -> {dimension_path}")


if __name__ == "__main__":
    main()
