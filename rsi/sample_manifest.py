"""Load and validate sample manifests used by RSI predictors."""

from __future__ import annotations

import json
from pathlib import Path


def load_rsi_samples(path: str | Path) -> list[dict]:
    """Load direct ``id/video/title/description/ecr`` RSI manifest records."""
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list):
        raise ValueError("sample manifest must be a JSON array")

    samples = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"sample {index} must be an object")
        missing = {"id", "video"} - set(item)
        if missing:
            raise ValueError(f"sample {index} missing fields: {sorted(missing)}")
        ecr_value = item.get("ecr", item.get("ECR"))
        samples.append({
            "id": str(item["id"]),
            "video": str(item["video"]),
            "title": "" if item.get("title") is None else str(item["title"]),
            "description": (
                "" if item.get("description") is None
                else str(item["description"])
            ),
            "ecr": None if ecr_value is None else float(ecr_value),
        })
    return samples
