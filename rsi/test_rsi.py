import csv
import json
import tempfile
import unittest
from pathlib import Path

from rsi.mario import MarioAPIError, MarioClient, extract_json_object
from rsi.sample_manifest import load_rsi_samples
from rsi.rubric import (
    build_group_prompt, normalize_rubric, parse_group_answer,
    validate_predictor_safe_rubric, weighted_score,
)
from rsi.strategy import default_strategy, normalize_strategy
from rsi.distill import (
    DEEP_REDUCE_BATCH_SIZE, MAX_MARIO_MESSAGE_BYTES, REDUCE_BATCH_SIZE,
    build_reduce_prompt, final_repair_prompt, load_rows, reduce_hierarchically,
    run_map_with_repair, run_mario_with_transport_retry, validate_final_rubric,
)
from rsi.analyze_dynamic import audit_dynamic
from rsi.evaluate_round2_candidates import (
    PRIMARY_CANDIDATE, evaluate_candidates,
)
from rsi.fusion import (
    align_dynamic_to_baseline, deterministic_derangement, evaluate,
    fixed_candidates, weighted_confidence,
)
from rsi.predict_regression import build_prompt, parse_regression_output
from rsi.prepare_strategy import bounded_groups, build_strategy
from rsi.prepare_training_splits import stratified_disjoint_splits
from rsi.calibration_feedback import build_feedback_cards
from rsi.strategy_variants import build_variants
from rsi.validation_gate import assess_feedback
from rsi.distill import select_stratified_full_groups


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


class ManifestTests(unittest.TestCase):
    def test_loads_direct_rsi_manifest_and_normalizes_optional_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            path.write_text(json.dumps([
                {
                    "id": "sample-1", "video": "/videos/one.mp4",
                    "title": "Title", "description": "Description",
                    "ecr": 95.25,
                },
                {"id": "sample-2", "video": "/videos/two.mp4", "ECR": "40"},
            ]), encoding="utf-8")
            samples = load_rsi_samples(path)

        self.assertEqual(samples[0], {
            "id": "sample-1", "video": "/videos/one.mp4",
            "title": "Title", "description": "Description", "ecr": 95.25,
        })
        self.assertEqual(samples[1], {
            "id": "sample-2", "video": "/videos/two.mp4",
            "title": "", "description": "", "ecr": 40.0,
        })

    def test_rejects_non_array_and_missing_required_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON array"):
                load_rsi_samples(path)
            path.write_text('[{"id":"sample-1"}]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing fields: \\[.*video.*\\]"):
                load_rsi_samples(path)


class Round3WorkflowTests(unittest.TestCase):
    def test_training_splits_are_balanced_disjoint_and_reproducible(self):
        rows = [
            {"id": str(index), "ecr": index / 1000, "video": f"{index}.mp4",
             "title": "", "description": ""}
            for index in range(1000)
        ]
        first = stratified_disjoint_splits(rows, 500, 500, 10, 7)
        second = stratified_disjoint_splits(rows, 500, 500, 10, 7)
        distill, calibration, metadata = first
        self.assertEqual(
            [item["id"] for item in distill],
            [item["id"] for item in second[0]],
        )
        self.assertEqual(len(distill), 500)
        self.assertEqual(len(calibration), 500)
        self.assertFalse(
            {item["id"] for item in distill} &
            {item["id"] for item in calibration}
        )
        self.assertEqual(metadata["overlap_count"], 0)
        self.assertTrue(all(item["distill"] == 50 for item in metadata["bin_summaries"]))

    def test_stratified_full_groups_use_every_sample_once(self):
        rows = [
            {"id": str(index), "ecr": index / 500, "video": f"{index}.mp4",
             "title": "", "description": ""}
            for index in range(500)
        ]
        groups = select_stratified_full_groups(rows, 50, 10, 9)
        self.assertEqual(len(groups), 50)
        self.assertTrue(all(len(group) == 10 for group in groups))
        ids = [item["id"] for group in groups for item in group]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {item["id"] for item in rows})
        self.assertTrue(all(
            {item["ecr_bin"] for item in group} == set(range(10))
            for group in groups
        ))

    def test_small_model_feedback_drives_dimension_decisions(self):
        import pandas as pd

        rubric = {
            "dimensions": [
                {"key": "negative", "weight_0_1": 0.4, "name": "Direct object",
                 "what_to_look_for": "visible object", "detection_heuristic": "direct",
                 "window_seconds": 3},
                {"key": "collapsed", "weight_0_1": 0.3, "name": "Overall narrative",
                 "what_to_look_for": "overall audience psychology",
                 "detection_heuristic": "global", "window_seconds": 20},
                {"key": "gate", "weight_0_1": 0.3, "name": "Caption question",
                 "what_to_look_for": "specific caption question",
                 "detection_heuristic": "direct", "window_seconds": 3},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rubric_path = root / "rubric.json"
            detail_path = root / "detail.csv"
            rubric_path.write_text(json.dumps(rubric), encoding="utf-8")
            pd.DataFrame({
                "Id": [str(i) for i in range(10)],
                "ECR": [50] * 10,
                "ECR_gt": list(range(10)),
                "negative_score": list(reversed(range(10))),
                "collapsed_score": [0.5] * 10,
                "gate_score": [0] * 5 + [1] * 5,
                "group_answers_json": [
                    '{"g_retry_collapsed":"x"}' for _ in range(3)
                ] + ["{}"] * 7,
            }).to_csv(detail_path, index=False)
            payload, _ = build_feedback_cards(detail_path, rubric_path)
        cards = {item["key"]: item for item in payload["dimension_cards"]}
        self.assertEqual(cards["negative"]["decision"], "delete")
        self.assertEqual(cards["collapsed"]["decision"], "delete")
        self.assertEqual(cards["gate"]["decision"], "retain_as_binary_gate")

    def test_strategy_variants_cover_each_dimension_once_and_gate_rejects_delete(self):
        rubric = normalize_rubric(sample_rubric())
        variants = build_variants(rubric)
        expected = {item["key"] for item in rubric["dimensions"]}
        self.assertEqual(set(variants), {"modality3", "mechanism", "temporal3", "small_group"})
        for strategy in variants.values():
            flattened = [key for group in strategy["call_groups"] for key in group]
            self.assertEqual(set(flattened), expected)
            self.assertEqual(len(flattened), len(set(flattened)))
        result = assess_feedback({
            "global": {"failure_rate": 0.0},
            "dimension_cards": [
                {"key": "bad", "decision": "delete", "weight": 0.4,
                 "small_model_reasoning_risk": "low"},
                {"key": "good", "decision": "retain", "weight": 0.6,
                 "small_model_reasoning_risk": "low"},
            ],
        })
        self.assertFalse(result["eligible_for_validation"])
        self.assertIn("bad", result["delete_dimensions"])


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
        self.assertIn("Required dimension order:\n- D0\n- D1", prompt)
        self.assertIn("reason must name concrete visible or audible evidence", prompt)
        self.assertNotIn("score: 0.xxx", prompt)
        self.assertNotIn("reason: observable evidence", prompt)
        parsed = parse_group_answer(
            "dimension: D0 ; score: 0.25 ; reason: a\n"
            "dimension: D1 ; score: 75 ; reason: b", ["D0", "D1"]
        )
        self.assertEqual(parsed["D0"]["score"], 0.25)
        self.assertEqual(parsed["D1"]["score"], 0.75)

    def test_parser_tolerates_key_formatting_drift(self):
        key = "hook_open_loop_payoff_at_gate"
        parsed = parse_group_answer(
            "dimension: hook_open_loop_payoff_at Gate; "
            "score: 0.5; reason: visible evidence",
            [key],
        )
        self.assertEqual(parsed[key]["score"], 0.5)

    def test_parser_accepts_multiline_fields_and_csv(self):
        key = "opening_subject_scene_legibility"
        multiline = parse_group_answer(
            "dimension: opening_subject_scene_legibility\n"
            "score: 0.5\n"
            "reason: A skateboarder is visible, but glare obscures details.",
            [key],
        )
        self.assertEqual(multiline[key]["score"], 0.5)
        self.assertIn("skateboarder", multiline[key]["reason"])

        csv_answer = parse_group_answer(
            "dimension,score,reason\n"
            "opening_subject_scene_legibility,0.75,Two children are clearly visible.",
            [key],
        )
        self.assertEqual(csv_answer[key]["score"], 0.75)
        self.assertEqual(
            csv_answer[key]["reason"], "Two children are clearly visible."
        )

    def test_parser_accepts_additional_model_format_drift(self):
        expected = [
            "opening_premise_hook",
            "audio_entry_and_speech_informativeness",
        ]
        markdown = parse_group_answer(
            "### Opening (opening_premise_hook)\n"
            "- dimension: opening_premise_hook\n"
            "- score: 0.5\n"
            "- reason: The premise remains unclear.\n"
            "### Audio (audio_entry_and_speech_informativeness)\n"
            "- dimension: audio_entry_and_speech_informativeness\n"
            "- score: 0.2\n"
            "- reason: No speech is audible in the opening.",
            expected,
        )
        self.assertEqual(set(markdown), set(expected))

        key = "production_quality_and_source_cleanliness"
        equals = parse_group_answer(
            "dimension=production_quality_and_source_cleanliness; "
            "score=0.5; reason=The frame is stable but slightly compressed.",
            [key],
        )
        self.assertEqual(equals[key]["score"], 0.5)

        bare = parse_group_answer(
            "production_quality_and_source_cleanliness; 0.5; "
            "The frame is stable but slightly compressed.",
            [key],
        )
        self.assertEqual(bare[key]["score"], 0.5)

        title_key = "title_metadata_specificity"
        shorthand = parse_group_answer(
            "title_metadata_specificity: 0.5; reason: The title is empty.",
            [title_key],
        )
        self.assertEqual(shorthand[title_key]["score"], 0.5)

    def test_partial_parser_supports_targeted_retry(self):
        answer = "dimension: D0 ; score: 0.25 ; reason: a"
        parsed = parse_group_answer(
            answer, ["D0", "D1"], require_all=False
        )
        self.assertEqual(set(parsed), {"D0"})
        with self.assertRaisesRegex(ValueError, "D1"):
            parse_group_answer(answer, ["D0", "D1"])

    def test_parser_accepts_markdown_table(self):
        parsed = parse_group_answer(
            "dimension|score|reason\n---|---|---\n"
            "D0|0.2|A concrete visible event.",
            ["D0"],
        )
        self.assertEqual(parsed["D0"]["score"], 0.2)
        self.assertIn("visible event", parsed["D0"]["reason"])

    def test_predictor_safety_rejects_external_tool_instructions(self):
        rubric = normalize_rubric(sample_rubric())
        rubric["dimensions"][0]["detection_heuristic"] = (
            "0 low; 0.5 medium; 1 high; run OCR and optical flow"
        )
        with self.assertRaisesRegex(ValueError, "unsupported OCR"):
            validate_predictor_safe_rubric(rubric)

        safe = normalize_rubric(sample_rubric())
        validate_predictor_safe_rubric(safe)

    def test_round2_strategy_splits_groups_to_two_dimensions(self):
        rubric = normalize_rubric(sample_rubric())
        rubric["recommended_call_groups"] = [
            ["D0", "D1", "D2", "D3"],
            ["D4", "D5", "D6", "D7"],
        ]
        self.assertEqual(
            bounded_groups(rubric["recommended_call_groups"]),
            [["D0", "D1"], ["D2", "D3"], ["D4", "D5"], ["D6", "D7"]],
        )
        strategy = build_strategy(rubric)
        self.assertTrue(all(len(group) <= 2 for group in strategy["call_groups"]))
        self.assertIn("Never claim OCR", strategy["prompt_prefix"])

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
        self.assertLessEqual(len(prompt.encode("utf-8")), MAX_MARIO_MESSAGE_BYTES)
        oversized = json.loads(json.dumps(rubric))
        oversized["dimensions"][0]["what_to_look_for"] = (
            "中" * MAX_MARIO_MESSAGE_BYTES
        )
        with self.assertRaises(ValueError):
            build_reduce_prompt([oversized], 1, "oversized", "", "")
        self.assertEqual(REDUCE_BATCH_SIZE, 3)
        self.assertEqual(DEEP_REDUCE_BATCH_SIZE, 2)

    def test_map_repairs_single_dimension_json(self):
        complete = sample_rubric()

        class RepairingClient:
            def __init__(self, answers):
                self.answers = iter(answers)
                self.calls = []

            def run(self, agent_id, system_prompt, prompt, files, title):
                self.calls.append((prompt, list(files)))
                return next(self.answers), {"turn": len(self.calls)}

        invalid = {"key": "only_one", "name": "incomplete"}
        with tempfile.TemporaryDirectory() as directory:
            client = RepairingClient([invalid, complete])
            rubric, trace = run_map_with_repair(
                client, "agent", "original map prompt", ["video.mp4"],
                "map title", Path(directory), "map_001",
            )
            self.assertEqual(len(rubric["dimensions"]), 8)
            self.assertEqual(len(client.calls), 2)
            self.assertEqual(client.calls[1][1], ["video.mp4"])
            self.assertIn("禁止只输出单个维度对象", client.calls[1][0])
            self.assertEqual(trace["semantic_attempts"][0]["status"], "schema_rejected")
            self.assertTrue((Path(directory) / "map_001_repair_01_raw.json").exists())

    def test_map_resume_rejects_invalid_raw_and_runs_only_repair(self):
        complete = sample_rubric()

        class RepairingClient:
            def __init__(self):
                self.calls = []

            def run(self, agent_id, system_prompt, prompt, files, title):
                self.calls.append((prompt, list(files)))
                return complete, {"turn": 1}

        invalid = {"key": "only_one", "name": "incomplete"}
        with tempfile.TemporaryDirectory() as directory:
            client = RepairingClient()
            rubric, trace = run_map_with_repair(
                client, "agent", "original map prompt", ["video.mp4"],
                "map title", Path(directory), "map_012", existing_raw=invalid,
            )
            self.assertEqual(len(rubric["dimensions"]), 8)
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(
                trace["semantic_attempts"][0]["status"],
                "schema_rejected_existing_raw",
            )
            self.assertTrue((Path(directory) / "map_012_repair_01_raw.json").exists())

    def test_map_transport_failure_retries_then_succeeds(self):
        complete = sample_rubric()

        class FlakyClient:
            def __init__(self):
                self.calls = 0

            def run(self, *args):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError(
                        "Mario session failed: {'message': 'unknown error'}"
                    )
                return complete, {"session_id": "success"}

        delays = []
        client = FlakyClient()
        raw, trace = run_mario_with_transport_retry(
            client, "agent", "prompt", ["video.mp4"], "map 37",
            sleep_fn=delays.append,
        )
        self.assertEqual(raw, complete)
        self.assertEqual(client.calls, 2)
        self.assertEqual(delays, [2])
        self.assertEqual(trace["transport_retries"], 1)
        self.assertIn("Mario session failed", trace["transport_failures"][0])

    def test_map_transport_does_not_retry_programming_error(self):
        class BrokenClient:
            def __init__(self):
                self.calls = 0

            def run(self, *args):
                self.calls += 1
                raise RuntimeError("local programming error")

        client = BrokenClient()
        with self.assertRaisesRegex(RuntimeError, "programming error"):
            run_mario_with_transport_retry(
                client, "agent", "prompt", [], "map 37",
                sleep_fn=lambda _: None,
            )
        self.assertEqual(client.calls, 1)

    def test_final_reduce_repairs_semantically_incomplete_json(self):
        complete = sample_rubric()

        class RepairingClient:
            def __init__(self):
                self.prompts = []

            def run(self, agent_id, system_prompt, prompt, files, title):
                self.prompts.append(prompt)
                if len(self.prompts) == 1:
                    return {"scale": "0 / 0.5 / 1"}, {"turn": 1}
                return complete, {"turn": 2}

        with tempfile.TemporaryDirectory() as directory:
            client = RepairingClient()
            final, trace = reduce_hierarchically(
                client, "agent", [complete, complete], Path(directory),
                2, "advisor", "",
            )
            self.assertEqual(len(final["dimensions"]), 8)
            self.assertEqual(len(client.prompts), 2)
            self.assertIn("上一份 JSON 不能作为最终 rubric 使用", client.prompts[1])
            self.assertEqual(trace["semantic_attempts"][0]["status"], "schema_rejected")
            self.assertEqual(trace["semantic_attempts"][1]["status"], "success")
            json.dumps(trace)
            self.assertTrue((Path(directory) / "reduce_raw.json").exists())
            self.assertTrue((Path(directory) / "reduce_repair_01_raw.json").exists())

    def test_final_reduce_reuses_valid_final_artifact_without_mario(self):
        complete = normalize_rubric(sample_rubric())

        class ForbiddenClient:
            def run(self, *args, **kwargs):
                raise AssertionError("Mario must not run when final artifact is valid")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "final_rubric.json").write_text(
                json.dumps(complete), encoding="utf-8"
            )
            final, trace = reduce_hierarchically(
                ForbiddenClient(), "agent", [complete, complete], root,
                2, "advisor", "",
            )
            self.assertEqual(len(final["dimensions"]), 8)
            self.assertEqual(trace["status"], "reused_valid_final_rubric")
            json.dumps(trace)

    def test_final_repair_prompt_requires_dimensions_first(self):
        prompt = final_repair_prompt(
            "original", {"scale": "0/0.5/1"}, ValueError("missing dimensions")
        )
        self.assertIn("顶层第一个字段必须是 dimensions", prompt)
        self.assertIn("6~12", prompt)

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


class FusionTests(unittest.TestCase):
    def test_alignment_matches_baseline_scale_without_labels(self):
        import numpy as np

        baseline = np.array([20.0, 40.0, 60.0, 80.0])
        dynamic = np.array([45.0, 50.0, 55.0, 60.0])
        aligned = align_dynamic_to_baseline(dynamic, baseline)
        self.assertAlmostEqual(float(aligned.mean()), float(baseline.mean()))
        self.assertAlmostEqual(float(aligned.std()), float(baseline.std()))

    def test_constant_dynamic_alignment_is_safe(self):
        import numpy as np

        baseline = np.array([20.0, 40.0, 60.0])
        aligned = align_dynamic_to_baseline(np.array([50.0] * 3), baseline)
        self.assertTrue(np.allclose(aligned, baseline.mean()))

    def test_weighted_confidence_and_fixed_candidates(self):
        import numpy as np
        import pandas as pd

        frame = pd.DataFrame({
            "a_score": [0.5, 1.0],
            "b_score": [0.5, 0.0],
        })
        confidence = weighted_confidence(frame, {"a": 0.75, "b": 0.25})
        self.assertTrue(np.allclose(confidence, [0.0, 1.0]))
        candidates = fixed_candidates(
            np.array([40.0, 60.0]), np.array([30.0, 80.0]), confidence
        )
        self.assertEqual(set(candidates), {
            "baseline_f", "dynamic_d", "z_linear_a090",
            "bounded_b010_cap5", "confidence_l030",
            "disagreement_l010_t20",
        })
        self.assertAlmostEqual(candidates["confidence_l030"][0], 40.0)

    def test_fusion_exclusion_recomputes_dynamic_score(self):
        import numpy as np
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "f.csv"
            dynamic = root / "d.csv"
            rubric = root / "rubric.json"
            pd.DataFrame({
                "Id": ["1", "2", "3"],
                "ECR": [20.0, 50.0, 80.0],
                "ECR_gt": [25.0, 45.0, 85.0],
            }).to_csv(baseline, index=False)
            pd.DataFrame({
                "Id": ["1", "2", "3"],
                "ECR": [50.0, 50.0, 50.0],
                "ECR_gt": [25.0, 45.0, 85.0],
                "keep_score": [0.2, 0.5, 0.8],
                "drop_score": [1.0, 0.0, 1.0],
            }).to_csv(dynamic, index=False)
            rubric.write_text(json.dumps({"dimensions": [
                {"key": "keep", "weight_0_1": 0.5},
                {"key": "drop", "weight_0_1": 0.5},
            ]}), encoding="utf-8")
            detail, _ = evaluate(
                baseline, dynamic, rubric, exclude_keys={"drop"}
            )
        self.assertTrue(np.allclose(detail["D"], [20.0, 50.0, 80.0]))

    def test_derangement_is_reproducible_and_has_no_fixed_points(self):
        import numpy as np

        first = deterministic_derangement(50)
        second = deterministic_derangement(50)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.all(first != np.arange(50)))

    def test_regression_prompt_and_output_scale(self):
        prompt = build_prompt("A title", "A description")
        self.assertTrue(prompt.startswith("\u003cvideo\u003e\nHow would you judge"))
        self.assertIn("The title of the video is A title", prompt)
        self.assertEqual(parse_regression_output((0.75,)), 75.0)
        self.assertEqual(parse_regression_output((75.0,)), 75.0)
        self.assertEqual(parse_regression_output((125.0,)), 100.0)

    def test_p0_audit_reports_midpoint_collapse_and_retries(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rubric_path = root / "rubric.json"
            detail_path = root / "detail.csv"
            rubric_path.write_text(json.dumps({
                "dimensions": [
                    {"key": "a", "weight_0_1": 0.7},
                    {"key": "b", "weight_0_1": 0.3},
                ]
            }), encoding="utf-8")
            pd.DataFrame({
                "Id": ["1", "2", "3"],
                "ECR": [40.0, 50.0, 60.0],
                "ECR_gt": [35.0, 55.0, 65.0],
                "a_score": [0.5, 0.5, 0.5],
                "b_score": [0.2, 0.5, 0.8],
                "a_reason": ["a", "b", "c"],
                "b_reason": ["d", "e", "f"],
                "group_answers_json": [
                    '{"group_1": "ok", "group_1_retry_a": "ok"}',
                    '{"group_1": "ok"}',
                    '{"group_1": "ok"}',
                ],
            }).to_csv(detail_path, index=False)
            overall, dimensions = audit_dynamic(detail_path, rubric_path)

        self.assertAlmostEqual(overall.iloc[0]["invariant_dimension_weight"], 0.7)
        self.assertAlmostEqual(overall.iloc[0]["midpoint_dominated_weight"], 0.7)
        self.assertEqual(overall.iloc[0]["targeted_retries"], 1)
        self.assertEqual(overall.iloc[0]["samples_with_retry"], 1)
        self.assertTrue(dimensions.set_index("key").loc["a", "invariant"])

    def test_round2_candidate_evaluation_uses_frozen_primary(self):
        import numpy as np
        import pandas as pd

        keys = [
            "opening_audio_immediate_audible_entry",
            "audio_event_turnover",
            "audio_visual_event_coupling",
            "zero_prior_comprehensibility",
            "caption_specific_event_or_question",
            "other_visual",
        ]
        rubric = {
            "dimensions": [
                {"key": key, "weight_0_1": 1 / len(keys)} for key in keys
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rubric_path = root / "rubric.json"
            detail_path = root / "detail.csv"
            rubric_path.write_text(json.dumps(rubric), encoding="utf-8")
            data = {
                "Id": ["1", "2", "3"],
                "ECR": [50.0, 60.0, 70.0],
                "ECR_gt": [40.0, 65.0, 80.0],
            }
            for index, key in enumerate(keys):
                data[f"{key}_score"] = (
                    np.array([0.2, 0.5, 0.8]) + index * 0.01
                )
            pd.DataFrame(data).to_csv(detail_path, index=False)
            candidate_detail, metrics = evaluate_candidates(
                detail_path, rubric_path
            )

        self.assertEqual(PRIMARY_CANDIDATE, "no_two_audio")
        self.assertEqual(
            set(metrics["candidate"]),
            {
                "full_round2", "no_two_audio", "cross_context_text",
                "caption_only_diagnostic",
            },
        )
        expected = np.mean([
            np.array(data[f"{key}_score"])
            for key in keys
            if key not in {
                "opening_audio_immediate_audible_entry", "audio_event_turnover"
            }
        ], axis=0) * 100
        self.assertTrue(np.allclose(candidate_detail["no_two_audio"], expected))


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
