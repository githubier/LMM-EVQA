"""Dependency-light schema validation and prompt builders for dynamic ECR rubrics."""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any

ALLOWED_DIRECTIONS = {"positive", "negative", "u_shaped"}
ALLOWED_MODALITIES = {"video", "audio", "audio_visual", "text", "multimodal"}
FORBIDDEN_PREDICTOR_TOOL_PATTERNS = {
    "OCR": r"\bocr\b|tesseract|文字识别",
    "ASR": r"\basr\b|speech[- ]to[- ]text|语音识别|转写器",
    "optical flow": r"optical flow|光流",
    "signal measurement": r"\brms\b|sobel|fft|频谱|自相关|帧差|ycrcb|\bhsv\b",
    "fabricated implementation detail": r"\bconf\s*=|\bpsm\s*=|16\s*khz|像素坐标|bounding box",
}


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return value or "dimension"


def normalize_rubric(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate a Mario rubric and normalize weights/keys without Pydantic."""
    if not isinstance(raw, dict):
        raise ValueError("rubric must be a JSON object")
    dimensions = raw.get("dimensions")
    if not isinstance(dimensions, list) or not 6 <= len(dimensions) <= 14:
        raise ValueError("rubric.dimensions must contain 6 to 14 items")

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
    group_metadata = []
    grouping_notes = []
    seen = set()
    for group_index, group in enumerate(groups, 1):
        metadata = None
        if isinstance(group, dict):
            metadata = dict(group)
            group = next(
                (
                    group.get(alias)
                    for alias in ("dimension_keys", "keys", "dimensions", "members")
                    if isinstance(group.get(alias), list)
                ),
                None,
            )
            if "group_key" not in metadata:
                for alias in ("group_id", "name", "id"):
                    if metadata.get(alias):
                        metadata["group_key"] = str(metadata[alias])
                        break
        if not isinstance(group, list):
            grouping_notes.append(f"ignored call group {group_index}: no key list")
            continue
        candidate_keys = [key for key in group if isinstance(key, str) and key]
        unknown = [key for key in candidate_keys if key not in used]
        duplicate = [key for key in candidate_keys if key in seen]
        accepted = [key for key in candidate_keys if key in used and key not in seen]
        if unknown:
            grouping_notes.append(
                f"call group {group_index}: ignored unknown keys {unknown}"
            )
        if duplicate:
            grouping_notes.append(
                f"call group {group_index}: ignored duplicate keys {duplicate}"
            )
        if not accepted:
            grouping_notes.append(f"ignored empty call group {group_index}")
            continue
        seen.update(accepted)
        checked_groups.append(accepted)
        if metadata is not None:
            metadata["dimension_keys"] = accepted
            for alias in ("keys", "dimensions", "members", "group_id", "name", "id"):
                metadata.pop(alias, None)
            metadata.setdefault("group_key", f"group_{group_index}")
            group_metadata.append(metadata)
        else:
            group_metadata.append({
                "group_key": f"group_{group_index}",
                "dimension_keys": accepted,
            })
    for key in [item["key"] for item in normalized if item["key"] not in seen]:
        checked_groups.append([key])
        group_metadata.append({
            "group_key": f"group_{len(group_metadata) + 1}",
            "dimension_keys": [key],
            "source": "added_by_normalizer",
        })
        grouping_notes.append(f"added singleton call group for missing key {key}")

    return {
        "schema_version": "rsi-rubric-v1",
        "dimensions": normalized,
        "scoring_formula": str(raw.get("scoring_formula", "weighted_sum")).strip(),
        "few_shot_calibration": raw.get("few_shot_calibration") or [],
        "caveats": [str(x) for x in raw.get("caveats") or []],
        "recommended_call_groups": checked_groups,
        "call_group_metadata": group_metadata,
        "normalization_notes": grouping_notes,
        "revision_summary": str(raw.get("revision_summary", "")).strip(),
    }


def validate_predictor_safe_rubric(rubric: dict[str, Any]) -> None:
    """Reject instructions the current predictor cannot actually execute.

    VideoLLaMA2 receives sampled audio/video plus metadata, but no external OCR,
    ASR, optical-flow, or signal-processing outputs. Mentioning such procedures
    in a scoring dimension caused Round 1 to fabricate measurements.
    """
    violations = []
    for item in rubric["dimensions"]:
        combined = "\n".join((
            item["name"], item["what_to_look_for"], item["detection_heuristic"]
        ))
        for label, pattern in FORBIDDEN_PREDICTOR_TOOL_PATTERNS.items():
            if re.search(pattern, combined, flags=re.IGNORECASE):
                violations.append(f"{item['key']}: unsupported {label}")
        if len(item["what_to_look_for"]) > 1200:
            violations.append(f"{item['key']}: what_to_look_for is too long/compound")
        if len(item["detection_heuristic"]) > 900:
            violations.append(f"{item['key']}: scoring anchors are too long/compound")
        anchors = item["detection_heuristic"].lower()
        has_numeric = all(token in anchors for token in ("0", "0.5", "1"))
        has_ordinal = all(token in anchors for token in ("low", "medium", "high")) or all(
            token in anchors for token in ("低", "中", "高")
        )
        if not (has_numeric or has_ordinal):
            violations.append(f"{item['key']}: missing low/mid/high scoring anchors")
    if violations:
        raise ValueError("predictor-unsafe rubric: " + "; ".join(violations))


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
    ordered_keys = "\n".join(f"- {key}" for key in group_keys)
    return f"""You are an expert in short-video ECR, defined as the probability of continuing past 5 seconds.
Assess only the dimensions below from the supplied audio/video evidence. Scores must be continuous values in [0,1], where 0 always means evidence harmful to continuation and 1 always means evidence supportive of continuation; apply the dimension's direction when converting raw observations into that contribution score. Do not infer unavailable evidence. Do not calculate final ECR.

Score every dimension independently from the current video's directly visible or audible evidence. Do not reuse a default score, copy a placeholder, or force different dimensions to share a score. The reason must name concrete visible or audible evidence from this video and explain how it supports the score. Generic reasons, including claims that evidence is merely present, are invalid. Evidence absence is not automatically neutral: apply that dimension's explicit low/mid/high anchors. Do not invent measurements or claim to have run OCR, ASR, speech-to-text, optical flow, frame-difference, RMS, spectrum, Sobel, or any other external detector; none of those tool outputs are supplied.

{definitions}

Video title: {title or '<empty>'}
Video description: {description or '<empty>'}

Return exactly one line per dimension in the order listed below. Each line must use the fields `dimension`, `score`, and `reason`, separated by semicolons. Use the exact dimension key, a decimal score from zero to one, and a concise evidence-specific reason. Return no bullets, commentary, or extra lines.

Required dimension order:
{ordered_keys}
"""


def _canonical_dimension_key(value: str) -> str:
    """Normalize harmless model formatting drift in dimension identifiers."""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def parse_group_answer(
    text: str,
    expected_keys: list[str],
    require_all: bool = True,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    key_aliases = {
        _canonical_dimension_key(key): key for key in expected_keys
    }

    def add_result(raw_key: str, raw_score: str, reason: str) -> None:
        key = key_aliases.get(_canonical_dimension_key(raw_key.strip()))
        try:
            value = float(raw_score)
        except (TypeError, ValueError):
            return
        if value > 1 and value <= 100:
            value /= 100
        if key is not None and key not in results and 0 <= value <= 1:
            results[key] = {"score": value, "reason": reason.strip()}

    labeled_pattern = re.compile(
        r"(?:^|\n)\s*[-*]?\s*dimension\s*[:=]\s*([^;\n]+)\s*"
        r"(?:;|\n)\s*[-*]?\s*score\s*[:=]\s*"
        r"([0-9]*\.?[0-9]+)\s*(?:;|\n)\s*[-*]?\s*"
        r"reason\s*[:=]\s*(.*?)"
        r"(?=\n\s*(?:###.*\n)?\s*[-*]?\s*dimension\s*[:=]|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    for match in labeled_pattern.finditer(text or ""):
        add_result(match.group(1), match.group(2), match.group(3))

    shorthand_pattern = re.compile(
        r"^\s*[-*]?\s*([A-Za-z0-9_ ]+?)\s*[:;]\s*"
        r"([0-9]*\.?[0-9]+)\s*;\s*(?:reason\s*:\s*)?(.*)$",
        re.IGNORECASE | re.MULTILINE,
    )
    for match in shorthand_pattern.finditer(text or ""):
        add_result(match.group(1), match.group(2), match.group(3))

    markdown_table_pattern = re.compile(
        r"^\s*\|?\s*([A-Za-z0-9_ ]+)\s*\|\s*"
        r"([0-9]*\.?[0-9]+)\s*\|\s*(.*?)\s*\|?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    for match in markdown_table_pattern.finditer(text or ""):
        if match.group(1).strip().lower() not in {"dimension", "---"}:
            add_result(match.group(1), match.group(2), match.group(3))

    try:
        rows = list(csv.reader(io.StringIO(text or "")))
    except csv.Error:
        rows = []
    for row in rows:
        if len(row) < 3 or row[0].strip().lower() == "dimension":
            continue
        add_result(row[0], row[1], ",".join(row[2:]))

    missing = [key for key in expected_keys if key not in results]
    if require_all and missing:
        raise ValueError(f"cannot parse dimensions {missing} from answer: {text!r}")
    return results


def weighted_score(rubric: dict[str, Any], scores: dict[str, dict[str, Any]]) -> float:
    """Safe executable default; free-form Mario formulas remain advisory metadata."""
    value = sum(
        item["weight_0_1"] * float(scores[item["key"]]["score"])
        for item in rubric["dimensions"]
    )
    return min(max(value, 0.0), 1.0)
