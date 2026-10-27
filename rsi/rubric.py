"""Dependency-light schema validation and prompt builders for dynamic ECR rubrics."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ALLOWED_DIRECTIONS = {"positive", "negative", "u_shaped"}
ALLOWED_MODALITIES = {"video", "audio", "audio_visual", "text", "multimodal"}


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return value or "dimension"


def normalize_rubric(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate a Mario rubric and normalize weights/keys without Pydantic."""
    if not isinstance(raw, dict):
        raise ValueError("rubric must be a JSON object")
    dimensions = raw.get("dimensions")
    if not isinstance(dimensions, list) or not 8 <= len(dimensions) <= 14:
        raise ValueError("rubric.dimensions must contain 8 to 14 items")

    normalized = []
    used = set()
    for index, item in enumerate(dimensions, 1):
        if not isinstance(item, dict):
            raise ValueError(f"dimension {index} must be an object")
        name = str(item.get("name", "")).strip()
        look = str(item.get("what_to_look_for", "")).strip()
        heuristic = str(item.get("detection_heuristic", "")).strip()
        if not name or not look or not heuristic:
            raise ValueError(f"dimension {index} misses name/look_for/heuristic")
        key = _slug(str(item.get("key") or name))
        base = key
        suffix = 2
        while key in used:
            key = f"{base}_{suffix}"
            suffix += 1
        used.add(key)
        direction = str(item.get("direction", "positive")).lower()
        if direction not in ALLOWED_DIRECTIONS:
            raise ValueError(f"invalid direction for {key}: {direction}")
        modality = str(item.get("modality", "multimodal")).lower()
        if modality not in ALLOWED_MODALITIES:
            raise ValueError(f"invalid modality for {key}: {modality}")
        try:
            weight = float(item.get("weight_0_1", 0))
            window = float(item.get("window_seconds", 5.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid numeric field for {key}") from exc
        if weight < 0 or not 0 < window <= 30:
            raise ValueError(f"invalid weight/window for {key}")
        normalized.append({
            "key": key,
            "name": name,
            "what_to_look_for": look,
            "direction": direction,
            "weight_0_1": weight,
            "modality": modality,
            "window_seconds": window,
            "detection_heuristic": heuristic,
        })
    total = sum(item["weight_0_1"] for item in normalized)
    if total <= 0:
        raise ValueError("dimension weights must have a positive sum")
    for item in normalized:
        item["weight_0_1"] = round(item["weight_0_1"] / total, 8)

    groups = raw.get("recommended_call_groups") or []
    if not isinstance(groups, list):
        raise ValueError("recommended_call_groups must be a list")
    checked_groups = []
    seen = set()
    for group in groups:
        if not isinstance(group, list) or not group:
            raise ValueError("each call group must be a non-empty key list")
        unknown = set(group) - used
        if unknown:
            raise ValueError(f"call group has unknown dimensions: {sorted(unknown)}")
        duplicate = set(group) & seen
        if duplicate:
            raise ValueError(f"dimensions occur in multiple groups: {sorted(duplicate)}")
        seen.update(group)
        checked_groups.append(group)
    for key in [item["key"] for item in normalized if item["key"] not in seen]:
        checked_groups.append([key])

    return {
        "schema_version": "rsi-rubric-v1",
        "dimensions": normalized,
        "scoring_formula": str(raw.get("scoring_formula", "weighted_sum")).strip(),
        "few_shot_calibration": raw.get("few_shot_calibration") or [],
        "caveats": [str(x) for x in raw.get("caveats") or []],
        "recommended_call_groups": checked_groups,
        "revision_summary": str(raw.get("revision_summary", "")).strip(),
    }


def load_rubric(path: str | Path) -> dict[str, Any]:
    return normalize_rubric(json.loads(Path(path).read_text(encoding="utf-8")))


def dimension_instruction(dimension: dict[str, Any]) -> str:
    return (
        f"### {dimension['name']} ({dimension['key']})\n"
        f"- direction: {dimension['direction']}\n"
        f"- modality: {dimension['modality']}\n"
        f"- observation window: first {dimension['window_seconds']:g} seconds\n"
        f"- what to observe: {dimension['what_to_look_for']}\n"
        f"- scoring anchors: {dimension['detection_heuristic']}"
    )


def build_group_prompt(
    rubric: dict[str, Any], group_keys: list[str], title: str, description: str
) -> str:
    index = {item["key"]: item for item in rubric["dimensions"]}
    definitions = "\n\n".join(dimension_instruction(index[key]) for key in group_keys)
    output = "\n".join(
        f"dimension: {key} ; score: 0.xxx ; reason: observable evidence" for key in group_keys
    )
    return f"""You are an expert in short-video ECR, defined as the probability of continuing past 5 seconds.
Assess only the dimensions below from the supplied audio/video evidence. Scores must be continuous values in [0,1], where 0 always means evidence harmful to continuation and 1 always means evidence supportive of continuation; apply the dimension's direction when converting raw observations into that contribution score. Do not infer unavailable evidence. Do not calculate final ECR.

{definitions}

Video title: {title or '<empty>'}
Video description: {description or '<empty>'}

Return exactly one line per dimension, in this order:
{output}
"""


def parse_group_answer(text: str, expected_keys: list[str]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    pattern = re.compile(
        r"dimension\s*:\s*([^;\n]+)\s*;\s*score\s*:\s*([0-9]*\.?[0-9]+)\s*;\s*reason\s*:\s*(.*)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text or ""):
        key = match.group(1).strip()
        value = float(match.group(2))
        if value > 1 and value <= 100:
            value /= 100
        if key in expected_keys and key not in results and 0 <= value <= 1:
            results[key] = {"score": value, "reason": match.group(3).strip()}
    missing = [key for key in expected_keys if key not in results]
    if missing:
        raise ValueError(f"cannot parse dimensions {missing} from answer: {text!r}")
    return results


def weighted_score(rubric: dict[str, Any], scores: dict[str, dict[str, Any]]) -> float:
    """Safe executable default; free-form Mario formulas remain advisory metadata."""
    value = sum(
        item["weight_0_1"] * float(scores[item["key"]]["score"])
        for item in rubric["dimensions"]
    )
    return min(max(value, 0.0), 1.0)
