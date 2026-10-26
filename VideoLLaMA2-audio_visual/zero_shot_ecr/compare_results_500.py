"""Compare ECR prediction runs on the 500-video test subset.

Reads the *_detail.csv files produced by infer_ecr_prompt.py and prints a
SRCC / PLCC / final-score table.

final_score = 0.6 * SROCC + 0.4 * PLCC   (VQualA 2025 EVQA challenge metric)
"""

import argparse
import os

import numpy as np
import pandas as pd
import scipy.stats
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error


def logistic_func(X, b1, b2, b3, b4, b5):
    # clip the exponent to avoid overflow warnings during curve_fit
    logistic_part = 1 + np.exp(np.clip(np.multiply(b2, X - b3), -500, 500))
    return b1 * (0.5 - np.divide(1, logistic_part)) + np.multiply(b4, X) + b5


def evaluate(y_pred, y_true):
    srcc = scipy.stats.spearmanr(y_true, y_pred)[0]
    try:
        krcc = scipy.stats.kendalltau(y_true, y_pred)[0]
    except Exception:
        krcc = scipy.stats.kendalltau(y_true, y_pred, method="asymptotic")[0]

    # PLCC after the standard 5-param logistic fit used by the challenge
    try:
        popt, _ = curve_fit(
            logistic_func, y_pred, y_true,
            p0=[10, 0, float(np.mean(y_pred)), 0.1, 0.1], maxfev=int(1e8),
        )
        y_fit = logistic_func(y_pred, *popt)
        plcc = scipy.stats.pearsonr(y_true, y_fit)[0]
        rmse = float(np.sqrt(mean_squared_error(y_true, y_fit)))
    except Exception:
        plcc = scipy.stats.pearsonr(y_true, y_pred)[0]
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    return {
        "SRCC": srcc,
        "KRCC": krcc,
        "PLCC": plcc,
        "RMSE": rmse,
        "final_score": 0.6 * srcc + 0.4 * plcc,
    }


DEFAULT_RUNS = [
    ("[ref] fine-tuned + regression head", "results/results_500_finetuned_detail.csv"),
    ("rubric ZH (no fewshot)", "results/results_500_rubric_nofewshot_detail.csv"),
    ("rubric ZH (no fewshot) + distilled", "results/results_500_rubric_nofewshot_rubricfile_detail.csv"),
    ("rubric ZH (with fewshot)", "results/results_500_rubric_detail.csv"),
    ("rubric ZH (with fewshot) + distilled", "results/results_500_rubric_rubricfile_detail.csv"),
    ("soft (rating-token probs)", "results/results_500_soft_detail.csv"),
    ("soft + distilled", "results/results_500_soft_rubricfile_detail.csv"),
    ("generate (plain numeric)", "results/results_500_generate_detail.csv"),
    ("generate + distilled", "results/results_500_gen_rubricfile_detail.csv"),
    ("5D divide a: fewshot, scores+formula", "results/results_500_rubric5_divide_a_fewshot_detail.csv"),
    ("5D divide b: no fewshot, scores+formula", "results/results_500_rubric5_divide_b_nofewshot_detail.csv"),
    ("5D divide c: fewshot, dimensions+scores+formula", "results/results_500_rubric5_divide_c_fewshot_dims_detail.csv"),
    ("5D divide d: no fewshot, dimensions+scores+formula", "results/results_500_rubric5_divide_d_nofewshot_dims_detail.csv"),
    ("12D direct (with fewshot)", "results/results_500_rubric12_fewshot_detail.csv"),
    ("12D direct (no fewshot)", "results/results_500_rubric12_nofewshot_detail.csv"),
    ("12D divide a: fewshot, scores+formula", "results/results_500_rubric12_divide_a_fewshot_detail.csv"),
    ("12D divide b: no fewshot, scores+formula", "results/results_500_rubric12_divide_b_nofewshot_detail.csv"),
    ("12D divide c: fewshot, dimensions+scores+formula", "results/results_500_rubric12_divide_c_fewshot_dims_detail.csv"),
    ("12D divide d: no fewshot, dimensions+scores+formula", "results/results_500_rubric12_divide_d_nofewshot_dims_detail.csv"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None,
                    help="Optional 'label=path.csv' pairs; defaults to the standard set")
    ap.add_argument(
        "--output",
        default="results/comparison_500.csv",
        help="CSV path for the comparison table",
    )
    args = ap.parse_args()

    runs = DEFAULT_RUNS
    if args.runs:
        runs = [tuple(r.split("=", 1)) for r in args.runs]

    rows = []
    for label, path in runs:
        if not os.path.exists(path):
            print(f"[skip] {label}: {path} not found")
            continue
        df = pd.read_csv(path)
        valid = df.dropna(subset=["ECR", "ECR_gt"])
        if len(valid) < 2:
            print(f"[skip] {label}: not enough valid rows")
            continue

        metrics = evaluate(
            valid["ECR"].to_numpy(dtype=float),
            valid["ECR_gt"].to_numpy(dtype=float),
        )
        metrics.update(
            {
                "run": label,
                "n": len(valid),
                "n_failed": int(df["ECR"].isna().sum()),
                "n_unique": int(valid["ECR"].nunique()),
                "pred_std": float(valid["ECR"].std()),
            }
        )
        rows.append(metrics)

    if not rows:
        print("no results found")
        return

    out = pd.DataFrame(rows)[
        ["run", "n", "n_failed", "n_unique", "pred_std",
         "SRCC", "PLCC", "final_score", "KRCC", "RMSE"]
    ]
    out = out.sort_values("final_score", ascending=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 40)
    print()
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print("final_score = 0.6*SROCC + 0.4*PLCC")
    print("reference: fine-tuned VideoLLaMA2 (challenge winner) = 0.710")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nsaved -> {args.output}")


if __name__ == "__main__":
    main()
