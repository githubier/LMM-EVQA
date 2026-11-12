#!/usr/bin/env python3
"""Create disjoint, ECR-stratified training-only RSI splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm")


def load_available_rows(csv_path: str | Path, video_dir: str | Path) -> list[dict]:
    root = Path(video_dir).resolve()
    rows = []
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                ecr = float(row.get("ECR", ""))
            except ValueError:
                continue
            identifier = str(row.get("Id", "")).strip()
            video = next((root / f"{identifier}{extension}" for extension in VIDEO_EXTENSIONS
                          if (root / f"{identifier}{extension}").is_file()
                          and (root / f"{identifier}{extension}").stat().st_size > 0), None)
            if video is None:
                continue
            rows.append({
                "id": identifier,
                "video": str(video),
                "title": "" if row.get("Title") is None else str(row.get("Title")),
                "description": "" if row.get("Description") is None else str(row.get("Description")),
                "ecr": ecr,
            })
    if not rows:
        raise ValueError("no labeled rows with local videos")
    return rows


def stratified_disjoint_splits(
    rows: list[dict], distill_size: int, calibration_size: int,
    bins: int, seed: int,
) -> tuple[list[dict], list[dict], dict]:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if distill_size % bins or calibration_size % bins:
        raise ValueError("split sizes must be divisible by bins")
    ordered = sorted(rows, key=lambda item: (item["ecr"], item["id"]))
    rng = random.Random(seed)
    distill_per_bin = distill_size // bins
    calibration_per_bin = calibration_size // bins
    distill, calibration, summaries = [], [], []
    for index in range(bins):
        start = len(ordered) * index // bins
        end = len(ordered) * (index + 1) // bins
        bucket = ordered[start:end]
        required = distill_per_bin + calibration_per_bin
        if len(bucket) < required:
            raise ValueError(f"bin {index} needs {required} rows, found {len(bucket)}")
        selected = rng.sample(bucket, required)
        distill.extend(selected[:distill_per_bin])
        calibration.extend(selected[distill_per_bin:])
        summaries.append({
            "bin": index,
            "population": len(bucket),
            "ecr_min": min(item["ecr"] for item in bucket),
            "ecr_max": max(item["ecr"] for item in bucket),
            "distill": distill_per_bin,
            "calibration": calibration_per_bin,
        })
    rng.shuffle(distill)
    rng.shuffle(calibration)
    overlap = {item["id"] for item in distill} & {item["id"] for item in calibration}
    if overlap:
        raise AssertionError(f"training splits overlap: {sorted(overlap)[:5]}")
    metadata = {
        "schema_version": "rsi-training-splits-v1",
        "seed": seed,
        "bins": bins,
        "available_population": len(rows),
        "distill_size": len(distill),
        "calibration_size": len(calibration),
        "overlap_count": 0,
        "bin_summaries": summaries,
    }
    return distill, calibration, metadata


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--distill-size", type=int, default=500)
    parser.add_argument("--calibration-size", type=int, default=500)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args()
    rows = load_available_rows(args.input_csv, args.video_dir)
    distill, calibration, metadata = stratified_disjoint_splits(
        rows, args.distill_size, args.calibration_size, args.bins, args.seed
    )
    output = Path(args.output_dir)
    write_json(output / "train_distill_500.json", distill)
    write_json(output / "train_calibration_500.json", calibration)
    write_json(output / "training_split_metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
