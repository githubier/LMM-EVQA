import json
import tempfile
import unittest
from pathlib import Path

from rsi.mario import extract_json_object
from rsi.rubric import (
    build_group_prompt, normalize_rubric, parse_group_answer, weighted_score,
)
from rsi.strategy import default_strategy, normalize_strategy


def sample_rubric():
    dimensions = []
    for index in range(8):
        dimensions.append({
            "key": f"D{index}", "name": f"Dimension {index}",
            "what_to_look_for": "observable evidence",
            "direction": "positive", "weight_0_1": index + 1,
            "modality": "audio_visual", "window_seconds": index % 5 + 1,
            "detection_heuristic": "0 low; 0.5 medium; 1 high",
        })
    return {
        "dimensions": dimensions, "scoring_formula": "weighted sum",
        "recommended_call_groups": [[f"D{i}", f"D{i + 1}"] for i in range(0, 8, 2)],
    }


class RubricTests(unittest.TestCase):
    def test_normalization_and_weighted_score(self):
        rubric = normalize_rubric(sample_rubric())
        self.assertAlmostEqual(sum(item["weight_0_1"] for item in rubric["dimensions"]), 1)
        scores = {item["key"]: {"score": 0.75} for item in rubric["dimensions"]}
        self.assertAlmostEqual(weighted_score(rubric, scores), 0.75)

    def test_group_prompt_and_parser(self):
        rubric = normalize_rubric(sample_rubric())
        prompt = build_group_prompt(rubric, ["D0", "D1"], "title", "description")
        self.assertIn("first 1 seconds", prompt)
        parsed = parse_group_answer(
            "dimension: D0 ; score: 0.25 ; reason: a\n"
            "dimension: D1 ; score: 75 ; reason: b", ["D0", "D1"]
        )
        self.assertEqual(parsed["D0"]["score"], 0.25)
        self.assertEqual(parsed["D1"]["score"], 0.75)

    def test_rejects_duplicate_group_keys(self):
        raw = sample_rubric()
        raw["recommended_call_groups"] = [["D0"], ["D0"]]
        with self.assertRaises(ValueError):
            normalize_rubric(raw)

    def test_strategy_must_cover_every_dimension(self):
        rubric = normalize_rubric(sample_rubric())
        strategy = default_strategy(rubric)
        strategy["call_groups"] = [["D0"]]
        with self.assertRaises(ValueError):
            normalize_strategy(strategy, rubric)


class MarioParsingTests(unittest.TestCase):
    def test_extract_json_with_surrounding_text(self):
        self.assertEqual(extract_json_object("before ```json\n{\"ok\": true}\n``` after"), {"ok": True})

    def test_no_json_is_error(self):
        with self.assertRaises(ValueError):
            extract_json_object("not a JSON result")


if __name__ == "__main__":
    unittest.main()
