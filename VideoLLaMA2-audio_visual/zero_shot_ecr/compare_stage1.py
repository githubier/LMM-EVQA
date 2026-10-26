"""Summarise every stage-1 result file into one comparison table."""

import argparse
import glob
import os

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from compare_results_500 import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="results/stage1_*_detail.csv")
    parser.add_argument("--output", default="results/comparison_stage1.csv")
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(args.pattern)):
        frame = pd.read_csv(path)
        valid = frame.dropna(subset=["ECR", "ECR_gt"])
        if len(valid) < 2:
            continue
        pred = valid["ECR"].to_numpy(float)
        truth = valid["ECR_gt"].to_numpy(float)
        metrics = evaluate(pred, truth)
        raw_pearson, raw_p = pearsonr(truth, pred)
        _, srcc_p = spearmanr(truth, pred)
        first = valid.iloc[0]
        rows.append({
            "run": os.path.basename(path).replace("_detail.csv", ""),
            "rubric_size": int(first["rubric_size"]),
            "flow": first["flow"],
            "score_source": first["score_source"],
            "aggregate": first["aggregate"],
            "python_formula": first.get("python_formula", "none"),
            "fewshot": bool(first["fewshot"]),
            "temporal_windows": bool(first["temporal_windows"]),
            "temporal_mode": first.get("temporal_mode", "windowed"),
            "aggregate_with_dimensions": bool(first.get("aggregate_with_dimensions", False)),
            "n": len(valid),
            "n_failed": int(frame["ECR"].isna().sum()),
            "n_unique": int(valid["ECR"].nunique()),
            "pred_std": float(valid["ECR"].std()),
            "SRCC": metrics["SRCC"],
            "SRCC_p": srcc_p,
            "raw_Pearson": raw_pearson,
            "raw_Pearson_p": raw_p,
            "PLCC": metrics["PLCC"],
            "final_score": metrics["final_score"],
            "KRCC": metrics["KRCC"],
            "RMSE": metrics["RMSE"],
        })

    if not rows:
        print("no stage-1 results found")
        return
    output = pd.DataFrame(rows).sort_values("final_score", ascending=False)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output.to_csv(args.output, index=False)
    pd.set_option("display.width", 240)
    print(output.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nsaved -> {args.output}")


if __name__ == "__main__":
    main()
