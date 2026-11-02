import csv
import json
import tempfile
import unittest
from pathlib import Path

from rsi.mario import MarioAPIError, MarioClient, extract_json_object
from rsi.rubric import (
    build_group_prompt, normalize_rubric, parse_group_answer, weighted_score,
)
from rsi.strategy import default_strategy, normalize_strategy
from rsi.distill import (
    MAX_MARIO_MESSAGE_CHARS, REDUCE_BATCH_SIZE, build_reduce_prompt, load_rows,
)


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

    def test_repairs_duplicate_group_keys(self):
        raw = sample_rubric()
        raw["recommended_call_groups"] = [["D0", "D1"], ["D0", "D2"]]
        rubric = normalize_rubric(raw)
        flattened = [
            key for group in rubric["recommended_call_groups"] for key in group
        ]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), {f"D{i}" for i in range(8)})
        self.assertTrue(any("duplicate" in note for note in rubric["normalization_notes"]))

    def test_strategy_must_cover_every_dimension(self):
        rubric = normalize_rubric(sample_rubric())
        strategy = default_strategy(rubric)
        strategy["call_groups"] = [["D0"]]
        with self.assertRaises(ValueError):
            normalize_strategy(strategy, rubric)
    def test_accepts_rich_call_group_objects(self):
        raw = sample_rubric()
        raw["recommended_call_groups"] = [
            {
                "group_key": "visual",
                "dimension_keys": [f"D{i}", f"D{i + 1}"],
                "window_seconds": 5,
                "modality_focus": "video",
            }
            for i in range(0, 8, 2)
        ]
        rubric = normalize_rubric(raw)
        self.assertEqual(rubric["recommended_call_groups"][0], ["D0", "D1"])
        self.assertEqual(rubric["call_group_metadata"][0]["group_key"], "visual")
        self.assertEqual(rubric["call_group_metadata"][0]["window_seconds"], 5)

    def test_accepts_alternative_rich_group_keys(self):
        raw = sample_rubric()
        raw["recommended_call_groups"] = [
            {
                "group_id": f"G{i}",
                "keys": [f"D{i}", f"D{i + 1}"],
                "call": "one multimodal call",
            }
            for i in range(0, 8, 2)
        ]
        rubric = normalize_rubric(raw)
        self.assertEqual(rubric["recommended_call_groups"][0], ["D0", "D1"])
        self.assertEqual(rubric["call_group_metadata"][0]["group_key"], "G0")
        self.assertNotIn("keys", rubric["call_group_metadata"][0])
        self.assertNotIn("group_id", rubric["call_group_metadata"][0])

    def test_reduce_prompt_enforces_character_limit(self):
        rubric = normalize_rubric(sample_rubric())
        prompt = build_reduce_prompt(
            [rubric, rubric, rubric], 1, "test reduce", "advisor", "",
        )
        self.assertLessEqual(len(prompt), MAX_MARIO_MESSAGE_CHARS)
        oversized = json.loads(json.dumps(rubric))
        oversized["dimensions"][0]["detection_heuristic"] = "x" * MAX_MARIO_MESSAGE_CHARS
        with self.assertRaises(ValueError):
            build_reduce_prompt([oversized], 1, "oversized", "", "")
        self.assertEqual(REDUCE_BATCH_SIZE, 3)

    def test_training_loader_skips_zero_byte_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "good.mp4").write_bytes(b"video")
            (root / "empty.mp4").touch()
            csv_path = root / "train.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["Id", "Title", "Description", "ECR"]
                )
                writer.writeheader()
                writer.writerow({"Id": "good", "Title": "", "Description": "", "ECR": 0.8})
                writer.writerow({"Id": "empty", "Title": "", "Description": "", "ECR": 0.2})
            rows = load_rows(str(csv_path), str(root), limit=10, seed=1)
            self.assertEqual([item["id"] for item in rows], ["good"])


class MarioParsingTests(unittest.TestCase):
    def test_busy_completion_interrupts_then_resends(self):
        class BusyMario(MarioClient):
            def __init__(self):
                self.send_count = 0
                self.interrupt_count = 0

            def send_message(self, session_id, text, file_ids):
                self.send_count += 1
                self.asserted_files = list(file_ids)
                if self.send_count == 1:
                    raise MarioAPIError(409, {
                        "type": "session_busy", "message": "interrupt first",
                    })
                return "turn-2"

            def interrupt(self, session_id):
                self.interrupt_count += 1

        client = BusyMario()
        turn_id, interrupted = client.send_completion_message("session", "continue")
        self.assertEqual(turn_id, "turn-2")
        self.assertTrue(interrupted)
        self.assertEqual(client.send_count, 2)
        self.assertEqual(client.interrupt_count, 1)
        self.assertEqual(client.asserted_files, [])

    def test_completion_followup_reuses_session_without_files(self):
        class FakeMario(MarioClient):
            def __init__(self):
                self.sent = []
                self.turns = iter([
                    ("still analyzing", "end_turn"),
                    ('{"dimensions": []}', "end_turn"),
                ])

            def update_system_prompt(self, agent_id, prompt):
                return None

            def upload_file(self, path):
                return f"file:{path}"

            def create_session(self, agent_id, title):
                return "session-1"

            def send_message(self, session_id, text, file_ids):
                self.sent.append((session_id, text, list(file_ids)))
                return f"turn-{len(self.sent)}"

            def _collect_turn(self, session_id, turn_id):
                return next(self.turns)

        client = FakeMario()
        result, trace = client.run("agent", "system", "task", ["video.mp4"], "title")
        self.assertEqual(result, {"dimensions": []})
        self.assertEqual(trace["completion_followups"], 1)
        self.assertEqual(client.sent[0][2], ["file:video.mp4"])
        self.assertEqual(client.sent[1][2], [])
        self.assertEqual({item["turn_id"] for item in trace["turns"]}, {"turn-1", "turn-2"})

    def test_extract_json_with_surrounding_text(self):
        self.assertEqual(extract_json_object("before ```json\n{\"ok\": true}\n``` after"), {"ok": True})

    def test_no_json_is_error(self):
        with self.assertRaises(ValueError):
            extract_json_object("not a JSON result")


if __name__ == "__main__":
    unittest.main()
