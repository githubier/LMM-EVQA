#!/usr/bin/env python3
"""Gate a frozen Round 3 rubric before any validation-label evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def assess_feedback(payload: dict) -> dict:
    cards = payload["dimension_cards"]
    reasons = []
    if payload["global"]["failure_rate"] > 0.02:
        reasons.append("global_failure_rate_above_2pct")
    delete = [card["key"] for card in cards if card["decision"] == "delete"]
    simplify = [
        card["key"] for card in cards
        if card["decision"] == "split_or_simplify"
    ]
    high_risk = [
        card["key"] for card in cards
        if card["small_model_reasoning_risk"] == "high"
    ]
    if delete:
        reasons.append("delete_dimensions_remain:" + ",".join(delete))
    if simplify:
        reasons.append("unsimplified_dimensions_remain:" + ",".join(simplify))
    if high_risk:
        reasons.append("high_reasoning_risk_dimensions_remain:" + ",".join(high_risk))
    usable = {"retain", "retain_as_binary_gate"}
    usable_weight = sum(
        card["weight"] for card in cards if card["decision"] in usable
    )
    if usable_weight < 0.50:
        reasons.append(f"usable_dimension_weight_below_0.50:{usable_weight:.6f}")
    return {
        "schema_version": "rsi-validation-gate-v1",
        "eligible_for_validation": not reasons,
        "failure_rate": payload["global"]["failure_rate"],
        "usable_dimension_weight": usable_weight,
        "delete_dimensions": delete,
        "unsimplified_dimensions": simplify,
        "high_reasoning_risk_dimensions": high_risk,
        "reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.feedback).read_text(encoding="utf-8"))
    result = assess_feedback(payload)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["eligible_for_validation"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
