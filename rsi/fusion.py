#!/usr/bin/env python3
"""Evaluate final ECR predictions from a fine-tuned baseline plus RSI evidence.

The fine-tuned regression output F is always the primary ECR prediction.  The
RSI rubric output D is only allowed to make a bounded/dynamically gated
correction.  Candidate formulas are fixed from the historical SnapUGC
experiments; this module does not tune a new score encoding on the evaluation
labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_squared_error


def align_dynamic_to_baseline(dynamic: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Z-align D to F without using ground-truth labels."""
    dynamic = np.asarray(dynamic, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    std = float(np.std(dynamic))
    if not np.isfinite(std) or std < 1e-8:
        return np.full_like(dynamic, float(np.mean(baseline)))
    return float(np.mean(baseline)) + float(np.std(baseline)) * (
        dynamic - float(np.mean(dynamic))
    ) / std


def weighted_confidence(
    frame: pd.DataFrame, dimension_weights: dict[str, float]
) -> np.ndarray:
    """Return weighted distance from the uninformative midpoint 0.5."""
    numer = np.zeros(len(frame), dtype=float)
    denom = np.zeros(len(frame), dtype=float)
    for key, weight in dimension_weights.items():
        column = f"{key}_score"
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        valid = np.isfinite(values)
        numer[valid] += weight * np.abs(values[valid] - 0.5) / 0.5
        denom[valid] += weight
    result = np.zeros(len(frame), dtype=float)
    np.divide(numer, denom, out=result, where=denom > 0)
    return np.clip(result, 0.0, 1.0)


def fixed_candidates(
    baseline: np.ndarray,
    dynamic: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, np.ndarray]:
    """Apply only formulas already supported by historical experiments."""
    baseline = np.asarray(baseline, dtype=float)
    dynamic = np.asarray(dynamic, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    aligned = align_dynamic_to_baseline(dynamic, baseline)
    gap = aligned - baseline
    return {
        "baseline_f": baseline,
        "dynamic_d": dynamic,
        "z_linear_a090": 0.9 * baseline + 0.1 * aligned,
        "bounded_b010_cap5": baseline + np.clip(0.1 * gap, -5.0, 5.0),
        "confidence_l030": (1.0 - 0.3 * confidence) * baseline
        + 0.3 * confidence * aligned,
        "disagreement_l010_t20": (
            1.0 - 0.1 * np.minimum(np.abs(gap) / 20.0, 1.0)
        )
        * baseline
        + 0.1 * np.minimum(np.abs(gap) / 20.0, 1.0) * aligned,
    }


def _logistic(x, b1, b2, b3, b4, b5):
    logistic_part = 1.0 + np.exp(np.clip(b2 * (x - b3), -500, 500))
    return b1 * (0.5 - 1.0 / logistic_part) + b4 * x + b5


def metric_row(name: str, prediction: np.ndarray, truth: np.ndarray) -> dict[str, float | str | int]:
    prediction = np.asarray(prediction, dtype=float)
    truth = np.asarray(truth, dtype=float)
    srcc = float(spearmanr(truth, prediction).statistic)
    krcc = float(kendalltau(truth, prediction).statistic)
    raw_pearson = float(pearsonr(truth, prediction).statistic)
    raw_rmse = float(np.sqrt(mean_squared_error(truth, prediction)))
    try:
        initial = [10.0, 0.0, float(np.mean(prediction)), 0.1, 0.1]
        params, _ = curve_fit(_logistic, prediction, truth, p0=initial, maxfev=int(1e8))
        fitted = _logistic(prediction, *params)
        plcc = float(pearsonr(truth, fitted).statistic)
        fitted_rmse = float(np.sqrt(mean_squared_error(truth, fitted)))
    except Exception:
        plcc = raw_pearson
        fitted_rmse = raw_rmse
    return {
        "candidate": name,
        "n": int(len(truth)),
        "n_unique": int(len(np.unique(np.round(prediction, 10)))),
        "srcc": srcc,
        "krcc": krcc,
        "raw_pearson": raw_pearson,
        "plcc": plcc,
        "final": 0.6 * srcc + 0.4 * plcc,
        "raw_rmse": raw_rmse,
        "logistic_rmse": fitted_rmse,
        "pred_mean": float(np.mean(prediction)),
        "pred_std": float(np.std(prediction)),
        "pred_min": float(np.min(prediction)),
        "pred_max": float(np.max(prediction)),
    }


def deterministic_derangement(size: int, seed: int = 20261106) -> np.ndarray:
    """Create a reproducible permutation with no fixed points."""
    if size < 2:
        raise ValueError("derangement requires at least two rows")
    rng = np.random.default_rng(seed)
    base = np.arange(size)
    for _ in range(1000):
        perm = rng.permutation(size)
        if np.all(perm != base):
            return perm
    return np.roll(base, 1)


def load_dimension_weights(
    path: str | Path, exclude_keys: set[str] | None = None
) -> dict[str, float]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    excluded = exclude_keys or set()
    weights = {
        item["key"]: float(item["weight_0_1"])
        for item in raw["dimensions"]
        if item["key"] not in excluded
    }
    if not weights:
        raise ValueError("no dimensions remain after exclusions")
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def _load_inputs(
    baseline_path: str | Path, dynamic_path: str | Path
) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_path, dtype={"Id": str})
    dynamic = pd.read_csv(dynamic_path, dtype={"Id": str})
    required_baseline = {"Id", "ECR", "ECR_gt"}
    required_dynamic = {"Id", "ECR", "ECR_gt"}
    if not required_baseline.issubset(baseline.columns):
        raise ValueError(f"baseline missing columns: {required_baseline - set(baseline.columns)}")
    if not required_dynamic.issubset(dynamic.columns):
        raise ValueError(f"dynamic missing columns: {required_dynamic - set(dynamic.columns)}")
    baseline = baseline.rename(columns={"ECR": "F", "ECR_gt": "truth_f"})
    dynamic = dynamic.rename(columns={"ECR": "D", "ECR_gt": "truth_d"})
    merged = baseline[["Id", "F", "truth_f"]].merge(dynamic, on="Id", how="inner")
    merged["truth"] = pd.to_numeric(merged["truth_f"], errors="coerce")
    truth_d = pd.to_numeric(merged["truth_d"], errors="coerce")
    mismatch = np.isfinite(merged["truth"]) & np.isfinite(truth_d) & (np.abs(merged["truth"] - truth_d) > 1e-6)
    if mismatch.any():
        raise ValueError(f"ground-truth mismatch for {int(mismatch.sum())} rows")
    merged["F"] = pd.to_numeric(merged["F"], errors="coerce")
    merged["D"] = pd.to_numeric(merged["D"], errors="coerce")
    return merged.dropna(subset=["F", "D", "truth"]).reset_index(drop=True)


def evaluate(
    baseline_path: str | Path,
    dynamic_path: str | Path,
    rubric_path: str | Path,
    exclude_keys: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = _load_inputs(baseline_path, dynamic_path)
    weights = load_dimension_weights(rubric_path, exclude_keys)
    if exclude_keys:
        values = merged[[f"{key}_score" for key in weights]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("excluded-dimension aggregation has missing scores")
        vector = np.asarray([weights[key] for key in weights], dtype=float)
        merged["D"] = values @ vector * 100.0
    confidence = weighted_confidence(merged, weights)
    baseline = merged["F"].to_numpy(float)
    dynamic = merged["D"].to_numpy(float)
    truth = merged["truth"].to_numpy(float)

    detail = merged[["Id", "truth", "F", "D"]].copy()
    detail["rubric_confidence"] = confidence
    rows: list[dict] = []

    correct = fixed_candidates(baseline, dynamic, confidence)
    for name, values in correct.items():
        detail[name] = np.clip(values, 0.0, 100.0)
        row = metric_row(name, detail[name].to_numpy(float), truth)
        row["control"] = "correct"
        rows.append(row)

    permutation = deterministic_derangement(len(merged))
    shuffled = fixed_candidates(baseline, dynamic[permutation], confidence[permutation])
    for name, values in shuffled.items():
        if name in {"baseline_f", "dynamic_d"}:
            continue
        row = metric_row(name, np.clip(values, 0.0, 100.0), truth)
        row["control"] = "shuffled"
        rows.append(row)

    constant = fixed_candidates(
        baseline, np.full(len(merged), 50.0), np.zeros(len(merged))
    )
    for name, values in constant.items():
        if name in {"baseline_f", "dynamic_d"}:
            continue
        row = metric_row(name, np.clip(values, 0.0, 100.0), truth)
        row["control"] = "constant"
        rows.append(row)

    metrics = pd.DataFrame(rows)
    return detail, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-detail", required=True)
    parser.add_argument("--dynamic-detail", required=True)
    parser.add_argument("--rubric", required=True)
    parser.add_argument(
        "--exclude-dimensions", default="",
        help="Comma-separated frozen dimension keys excluded from D and confidence.",
    )
    parser.add_argument("--p1-sota", type=float, default=0.6974)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    excluded = {
        key.strip() for key in args.exclude_dimensions.split(",") if key.strip()
    }
    detail, metrics = evaluate(
        args.baseline_detail, args.dynamic_detail, args.rubric, excluded
    )
    baseline_final = float(metrics.loc[
        (metrics["candidate"] == "baseline_f") & (metrics["control"] == "correct"),
        "final",
    ].iloc[0])
    metrics["delta_vs_split_f"] = metrics["final"] - baseline_final
    metrics["reference_p1_sota"] = args.p1_sota
    metrics["delta_vs_reference_p1_sota"] = metrics["final"] - args.p1_sota
    metrics["dynamic_variant"] = (
        "exclude:" + ",".join(sorted(excluded)) if excluded else "full_rubric"
    )
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    detail_path = prefix.with_name(prefix.name + "_detail.csv")
    metrics_path = prefix.with_name(prefix.name + "_metrics.csv")
    final_path = prefix.with_name(prefix.name + "_final.csv")
    detail.to_csv(detail_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    # Pre-registered primary prediction: the historically strongest formula.
    detail[["Id", "confidence_l030"]].rename(
        columns={"confidence_l030": "ECR"}
    ).to_csv(final_path, index=False)
    print(metrics.to_string(index=False))
    print(f"detail -> {detail_path}")
    print(f"metrics -> {metrics_path}")
    print(f"fixed final prediction (confidence-aware lambda=0.3) -> {final_path}")


if __name__ == "__main__":
    main()
