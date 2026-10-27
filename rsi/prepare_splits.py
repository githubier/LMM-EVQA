#!/usr/bin/env python3
"""Create deterministic RSI feedback/selection manifests without touching test data."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def load_joined(data_path: str, truth_path: str, video_dir: str) -> list[dict]:
    with open(truth_path, encoding="utf-8-sig", newline="") as stream:
        truth = {row["Id"]: float(row["ECR"]) * 100 for row in csv.DictReader(stream)}
    rows = []
    with open(data_path, encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            identifier = row["Id"]
            video = Path(video_dir) / f"{identifier}.mp4"
            if identifier in truth and video.is_file():
                rows.append({
                    "id": identifier, "video": str(video.resolve()),
                    "title": row.get("Title", ""), "description": row.get("Description", ""),
                    "ecr": truth[identifier],
                })
    return rows


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--truth-csv", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feedback-size", type=int, default=500)
    parser.add_argument("--selection-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    rows = load_joined(args.data_csv, args.truth_csv, args.video_dir)
    needed = args.feedback_size + args.selection_size
    if len(rows) < needed:
        raise ValueError(f"need {needed} labeled local videos, found {len(rows)}")
    random.Random(args.seed).shuffle(rows)
    out = Path(args.output_dir)
    feedback = rows[:args.feedback_size]
    selection = rows[args.feedback_size:needed]
    write_json(out / "validation_feedback.json", feedback)
    write_json(out / "validation_selection.json", selection)
    write_json(out / "split_manifest.json", {
        "seed": args.seed, "available_local_labeled": len(rows),
        "feedback_ids": [item["id"] for item in feedback],
        "selection_ids": [item["id"] for item in selection],
        "test_set_used": False,
    })
    print(f"feedback={len(feedback)} selection={len(selection)}")


if __name__ == "__main__":
    main()
