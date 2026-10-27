"""Prediction strategy validation, kept separate from distilled rubric semantics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def default_strategy(rubric: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "rsi-strategy-v1",
        "predictor": "videollama2",
        "modal_type": "av",
        "call_groups": rubric["recommended_call_groups"],
        "window_overrides": {},
        "prompt_prefix": "",
        "prompt_suffix": "",
        "aggregation": {"method": "weighted_sum"},
    }


def normalize_strategy(raw: dict[str, Any], rubric: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("strategy must be a JSON object")
    result = default_strategy(rubric)
    result.update(raw)
    keys = {item["key"] for item in rubric["dimensions"]}
    groups = result.get("call_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("strategy.call_groups must be non-empty")
    flattened = [key for group in groups if isinstance(group, list) for key in group]
    if len(flattened) != len(set(flattened)) or set(flattened) != keys:
        raise ValueError("call_groups must contain every dimension key exactly once")
    overrides = result.get("window_overrides") or {}
    if set(overrides) - keys:
        raise ValueError("window_overrides contains unknown dimension keys")
    for key, value in overrides.items():
        value = float(value)
        if not 0 < value <= 30:
            raise ValueError(f"invalid window override for {key}")
        overrides[key] = value
    aggregation = result.get("aggregation") or {}
    if aggregation.get("method", "weighted_sum") != "weighted_sum":
        raise ValueError("only safe deterministic weighted_sum is currently supported")
    if result.get("modal_type") not in ("v", "av"):
        raise ValueError("modal_type must be v or av")
    result["call_groups"] = groups
    result["window_overrides"] = overrides
    return result


def load_strategy(path: str | Path | None, rubric: dict[str, Any]) -> dict[str, Any]:
    if not path:
        return default_strategy(rubric)
    return normalize_strategy(json.loads(Path(path).read_text(encoding="utf-8")), rubric)
