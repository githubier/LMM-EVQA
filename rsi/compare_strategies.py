#!/usr/bin/env python3
"""Compare controlled strategy variants on training calibration and freeze one."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from rsi.analyze_dynamic import audit_dynamic


def compare_variants(
    runs_dir: str | Path, strategies_dir: str | Path, rubric_path: str | Path,
    maximum_failure_rate: float = 0.02,
) -> tuple[pd.DataFrame, str]:
    runs = Path(runs_dir)
    strategies = Path(strategies_dir)
    manifest = json.loads((strategies / "strategy_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for name, metadata in manifest.items():
        detail = runs / name / "calibration_detail.csv"
        if not detail.exists():
            raise FileNotFoundError(f"missing calibration result for {name}: {detail}")
        overall, _ = audit_dynamic(detail, rubric_path)
        row = overall.iloc[0].to_dict()
        row["strategy"] = name
        row["calls_per_video"] = int(metadata["calls_per_video"])
        row["strategy_path"] = metadata["path"]
        rows.append(row)
    table = pd.DataFrame(rows)
    eligible = table[table["failure_rate"] <= maximum_failure_rate].copy()
    if eligible.empty:
        raise ValueError(
            f"no strategy meets failure_rate <= {maximum_failure_rate:.2%}"
        )
    eligible = eligible.sort_values(
        ["final", "srcc", "targeted_retries", "calls_per_video"],
        ascending=[False, False, True, True],
    )
    return table.sort_values("strategy").reset_index(drop=True), str(eligible.iloc[0]["strategy"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--strategies-dir", required=True)
    parser.add_argument("--rubric", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--maximum-failure-rate", type=float, default=0.02)
    args = parser.parse_args()
    table, winner = compare_variants(
        args.runs_dir, args.strategies_dir, args.rubric,
        args.maximum_failure_rate,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "strategy_comparison.csv", index=False)
    source = Path(args.strategies_dir) / f"strategy_{winner}.json"
    frozen = output / "strategy_frozen.json"
    shutil.copyfile(source, frozen)
    decision = {
        "schema_version": "rsi-strategy-selection-v1",
        "selection_split": "train_calibration_500",
        "pre_registered_rule": (
            "highest Final among failure_rate<=2%; ties by SRCC, fewer retries, fewer calls"
        ),
        "winner": winner,
        "source": str(source.resolve()),
        "frozen": str(frozen.resolve()),
        "validation_labels_used": False,
    }
    (output / "strategy_selection.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(table[["strategy", "n", "failure_rate", "srcc", "plcc", "final",
                 "exact_midpoint_rate_all_scores", "targeted_retries",
                 "calls_per_video"]].to_string(index=False))
    print(f"frozen strategy: {winner} -> {frozen}")


if __name__ == "__main__":
    main()
