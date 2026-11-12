#!/usr/bin/env python3
"""Budgeted, resumable Mario Map/Reduce distillation for one RSI round."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import requests

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsi.mario import BASE_URLS, MarioClient, api_key_from_env
from rsi.rubric import normalize_rubric, validate_predictor_safe_rubric
from rsi.sample_manifest import load_rsi_samples

SYSTEM_PROMPT = """你是 Qwen3.8-Max ECR Rubric Distiller。ECR 是观看超过 5 秒的用户比例。
你要从带真实标签的短视频中总结可迁移、可观察、可由本地多模态模型直接执行的预测维度。
本地 Predictor 只接收视频帧、原始音轨、标题和描述，不会收到 OCR、ASR、光流、RMS、频谱、Sobel、帧差或其他外部程序的测量结果。禁止要求 Predictor 运行、引用或虚构这些工具结果。
只能输出一个合法 JSON 对象，不要输出 Markdown。维度必须原子化、机制互补、模态和时间窗口明确，禁止同义重复。"""

MAP_TEMPLATE = """这是 RSI 第 {round_index} 轮的第 {group_index} 个 Map 组。
附件按下面顺序对应样本：
{labels}

请比较不同 ECR 区间的直接可观察差异，同时关注相邻区间之间影响全局排序的细微边界；不要只总结最高与最低样本。输出 6~12 维候选 rubric。JSON 必须包含：
- dimensions: 每项含 key、name、what_to_look_for、direction(positive/negative/u_shaped)、weight_0_1、modality(video/audio/audio_visual/text/multimodal)、window_seconds(0,30]、detection_heuristic；判分锚点必须统一为 0=不利于继续观看、1=有利于继续观看，direction 只描述原始观察信号与 ECR 的关系；
- scoring_formula；few_shot_calibration；caveats；
- recommended_call_groups: 必须是对象数组；每个对象必须且只能用 group_key、dimension_keys、window_seconds、modality_focus、call_purpose 这些字段，其中 dimension_keys 是字符串数组；禁止使用 group_id 或 keys 作为字段名；每个维度 key 恰好出现一次；
- revision_summary。
每个判分标准必须包含低/中/高锚点，并引用本组至少两个不同样本的直接可见或可听证据。每个维度只判断一个主要机制；不得把主题、钩子、相关性、制作质量、节奏等多个机制塞进同一维度。detection_heuristic 只能描述 Predictor 根据输入视频、音轨、标题和描述直接作出的观察与判分，不得包含 OCR、ASR、speech-to-text、光流、帧差、RMS、频谱、Sobel、像素阈值、坐标、置信度或任何外部算法步骤。
{advisor}
{previous}
"""

REDUCE_TEMPLATE = """这是 RSI 第 {round_index} 轮的 {stage}。以下是 {count} 个独立候选 rubric：
{rubrics}

融合为一个最终 JSON rubric。顶层第一个字段必须是 dimensions，且 dimensions 必须是包含 6~12 个完整维度对象的数组；禁止只输出 scale、anchors、direction_rule、摘要或部分 schema。保留跨组稳定且能由 Predictor 直接观察的信号，去除同义重复；输出 6~12 个原子、互补维度，权重和应为 1；为每个维度确定主要模态和 (0,30] 秒观察窗口；所有维度评分统一为 0=不利于继续观看、1=有利于继续观看；recommended_call_groups 必须是对象数组，每个对象只使用 group_key、dimension_keys、window_seconds、modality_focus、call_purpose，其中 dimension_keys 为字符串数组且每个维度 key 恰好出现一次，禁止使用 group_id 或 keys 字段。不得包含 OCR、ASR、光流、帧差、RMS、频谱、Sobel、像素阈值、坐标或其他外部工具步骤；不得虚构没有出现在候选结果中的视频证据。只输出 JSON。
最低完整结构必须同时包含：dimensions、scoring_formula、few_shot_calibration、caveats、recommended_call_groups、revision_summary。
{advisor}
{previous}
"""

REDUCE_BATCH_SIZE = 3
DEEP_REDUCE_BATCH_SIZE = 2
FINAL_SCHEMA_RETRIES = 2
MAP_SCHEMA_RETRIES = 2
MAP_TRANSPORT_RETRIES = 2
MAX_MARIO_MESSAGE_BYTES = 60000
REDUCE_HEURISTIC_CHARS = 240


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_rows(path: str, video_dir: str, limit: int, seed: int) -> list[dict]:
    """Reservoir-sample available rows so the candidate budget is order-independent."""
    video_root = Path(video_dir).resolve()
    extensions = (".mp4", ".mov", ".mkv", ".webm")
    rows = []
    available = 0
    rng = random.Random(seed)
    with open(path, encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                ecr = float(row.get("ECR", ""))
            except ValueError:
                continue
            identifier = str(row.get("Id", "")).strip()
            candidates = [video_root / f"{identifier}{ext}" for ext in extensions]
            video = next(
                (item for item in candidates if item.is_file() and item.stat().st_size > 0),
                None,
            )
            if not video:
                continue
            item = {
                "id": identifier, "ecr": ecr, "video": str(video),
                "title": row.get("Title", ""), "description": row.get("Description", ""),
            }
            available += 1
            if not limit or len(rows) < limit:
                rows.append(item)
            else:
                replacement = rng.randrange(available)
                if replacement < limit:
                    rows[replacement] = item
    if not rows:
        raise ValueError("no labeled rows with matching local video files")
    return rows


def select_groups(rows: list[dict], groups: int, videos_per_group: int) -> list[list[dict]]:
    if videos_per_group < 4 or videos_per_group % 2:
        raise ValueError("--videos-per-group must be an even number >= 4")
    ordered = sorted(rows, key=lambda item: item["ecr"])
    each = videos_per_group // 2
    needed = groups * each
    if len(ordered) < needed * 2:
        raise ValueError(f"need at least {needed * 2} available videos, found {len(ordered)}")
    lows = ordered[:needed]
    highs = list(reversed(ordered[-needed:]))
    return [highs[i * each:(i + 1) * each] + lows[i * each:(i + 1) * each]
            for i in range(groups)]


def select_stratified_full_groups(
    rows: list[dict], groups: int, videos_per_group: int, seed: int
) -> list[list[dict]]:
    """Use every balanced manifest row once, with one row per ECR bin/group."""
    if len(rows) != groups * videos_per_group:
        raise ValueError(
            "stratified_full requires rows == groups * videos_per_group "
            f"({len(rows)} != {groups} * {videos_per_group})"
        )
    ordered = sorted(rows, key=lambda item: (item["ecr"], item["id"]))
    bins = [
        ordered[len(ordered) * index // videos_per_group:
                len(ordered) * (index + 1) // videos_per_group]
        for index in range(videos_per_group)
    ]
    rng = random.Random(seed)
    for bucket in bins:
        rng.shuffle(bucket)
    result = []
    for group_index in range(groups):
        group = []
        for bin_index, bucket in enumerate(bins):
            item = dict(bucket[group_index])
            item["ecr_bin"] = bin_index
            group.append(item)
        result.append(group)
    identifiers = [item["id"] for group in result for item in group]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("stratified_full groups must use every id exactly once")
    return result


def compact_rubric(rubric: dict) -> dict:
    """Keep cross-candidate semantics while bounding recursive Reduce growth."""
    dimensions = []
    for item in rubric["dimensions"]:
        dimensions.append({
            "key": item["key"],
            "name": item["name"],
            "what_to_look_for": item["what_to_look_for"],
            "direction": item["direction"],
            "weight_0_1": item["weight_0_1"],
            "modality": item["modality"],
            "window_seconds": item["window_seconds"],
            "detection_heuristic": item["detection_heuristic"][:REDUCE_HEURISTIC_CHARS],
        })
    return {"dimensions": dimensions}


def build_reduce_prompt(
    rubrics: list[dict], round_index: int, stage: str,
    advisor: str, previous: str,
) -> str:
    summaries = "\n\n".join(
        f"CANDIDATE {index}:\n{json.dumps(compact_rubric(item), ensure_ascii=False)}"
        for index, item in enumerate(rubrics, 1)
    )
    prompt = REDUCE_TEMPLATE.format(
        round_index=round_index, stage=stage, count=len(rubrics),
        rubrics=summaries, advisor=advisor, previous=previous,
    )
    prompt_bytes = len(prompt.encode("utf-8"))
    if prompt_bytes > MAX_MARIO_MESSAGE_BYTES:
        raise ValueError(
            f"{stage} prompt has {prompt_bytes} UTF-8 bytes, exceeding the safe "
            f"Mario database limit {MAX_MARIO_MESSAGE_BYTES}; reduce the batch size"
        )
    return prompt


def validate_final_rubric(raw: dict) -> dict:
    """Normalize and enforce the executable Round 2 final-rubric contract."""
    final = normalize_rubric(raw)
    if not 6 <= len(final["dimensions"]) <= 12:
        raise ValueError("final rubric must contain 6 to 12 dimensions")
    validate_predictor_safe_rubric(final)
    return final


def validate_map_rubric(raw: dict) -> dict:
    """Enforce the candidate Map contract before it enters Reduce."""
    rubric = normalize_rubric(raw)
    if not 6 <= len(rubric["dimensions"]) <= 12:
        raise ValueError("Map rubric must contain 6 to 12 dimensions")
    return rubric


def map_repair_prompt(original_prompt: str, invalid: dict, error: Exception) -> str:
    return (
        original_prompt
        + "\n\n你上一份 JSON 不能作为 Map 候选 rubric 使用。"
        + f"校验错误：{type(error).__name__}: {error}。\n"
        + "上一份无效 JSON：\n"
        + json.dumps(invalid, ensure_ascii=False)
        + "\n请重新观察本次附件视频并输出完整替代 JSON，不要解释错误。"
        + "禁止只输出单个维度对象。顶层第一个字段必须是 dimensions，"
        + "包含 6~12 个完整维度；同时包含 scoring_formula、"
        + "few_shot_calibration、caveats、recommended_call_groups、revision_summary。"
    )


def is_transient_mario_error(exc: Exception) -> bool:
    """Retry only transport failures and server-side Mario session failures."""
    return isinstance(exc, requests.RequestException) or (
        isinstance(exc, RuntimeError)
        and str(exc).startswith("Mario session failed:")
    )


def run_mario_with_transport_retry(
    client: MarioClient, agent_id: str, prompt: str, files: list[str],
    title: str, sleep_fn=time.sleep,
) -> tuple[dict, dict]:
    failures = []
    for retry in range(MAP_TRANSPORT_RETRIES + 1):
        try:
            raw, trace = client.run(
                agent_id, SYSTEM_PROMPT, prompt, files,
                f"{title} transport {retry + 1}",
            )
            if failures:
                trace = dict(trace)
                trace["transport_failures"] = failures
                trace["transport_retries"] = len(failures)
            return raw, trace
        except Exception as exc:
            if not is_transient_mario_error(exc) or retry >= MAP_TRANSPORT_RETRIES:
                raise
            failure = f"{type(exc).__name__}: {exc}"
            failures.append(failure)
            delay = 2 ** (retry + 1)
            print(
                f"{title}: transient Mario failure; retry "
                f"{retry + 1}/{MAP_TRANSPORT_RETRIES} in {delay}s: {failure}",
                file=sys.stderr, flush=True,
            )
            sleep_fn(delay)
    raise AssertionError("unreachable transport retry loop")


def run_map_with_repair(
    client: MarioClient, agent_id: str, prompt: str, files: list[str],
    title: str, output: Path, prefix: str, existing_raw: dict | None = None,
) -> tuple[dict, dict]:
    """Recover or run one Map group without getting stuck on invalid raw JSON."""
    attempts = []
    active_prompt = prompt
    start_attempt = 0
    if existing_raw is not None:
        try:
            return validate_map_rubric(existing_raw), {
                "status": "recovered_from_raw", "semantic_attempts": [],
            }
        except ValueError as exc:
            attempts.append({
                "attempt": 1, "status": "schema_rejected_existing_raw",
                "error": f"{type(exc).__name__}: {exc}",
            })
            active_prompt = map_repair_prompt(prompt, existing_raw, exc)
            start_attempt = 1

    rubric = None
    last_error = None
    for attempt in range(start_attempt, MAP_SCHEMA_RETRIES + 1):
        suffix = "" if attempt == 0 else f"_repair_{attempt:02d}"
        raw, trace = run_mario_with_transport_retry(
            client, agent_id, active_prompt, files,
            f"{title} attempt {attempt + 1}",
        )
        atomic_json(output / f"{prefix}{suffix}_raw.json", raw)
        atomic_json(output / f"{prefix}{suffix}_trace.json", trace)
        try:
            rubric = validate_map_rubric(raw)
            attempts.append({
                "attempt": attempt + 1, "status": "success", **trace,
            })
            break
        except ValueError as exc:
            last_error = exc
            attempts.append({
                "attempt": attempt + 1, "status": "schema_rejected",
                "error": f"{type(exc).__name__}: {exc}", **trace,
            })
            if attempt >= MAP_SCHEMA_RETRIES:
                raise ValueError(
                    f"Map remained invalid after {attempt + 1} semantic attempts: {exc}"
                ) from exc
            active_prompt = map_repair_prompt(prompt, raw, exc)
    if rubric is None:
        raise AssertionError(f"Map validation ended without a rubric: {last_error}")
    return rubric, {
        "status": "success", "semantic_attempts": attempts,
        **attempts[-1],
    }


def final_repair_prompt(original_prompt: str, invalid: dict, error: Exception) -> str:
    """Ask for a complete replacement after a schema-valid but unusable JSON."""
    return (
        original_prompt
        + "\n\n你上一份 JSON 不能作为最终 rubric 使用。"
        + f"校验错误：{type(error).__name__}: {error}。\n"
        + "上一份无效 JSON：\n"
        + json.dumps(invalid, ensure_ascii=False)
        + "\n请重新输出一个完整替代 JSON。不要解释错误，不要只输出评分尺度。"
        + "顶层第一个字段必须是 dimensions，包含 6~12 个完整维度；同时必须包含 "
        + "scoring_formula、few_shot_calibration、caveats、recommended_call_groups、revision_summary。"
    )


def reduce_hierarchically(
    client: MarioClient, agent_id: str, rubrics: list[dict], out: Path,
    round_index: int, advisor: str, previous: str,
) -> tuple[dict, dict]:
    """Recursively reduce in resumable batches below Mario's message limit."""
    final_artifact = out / "final_rubric.json"
    if final_artifact.exists():
        final = validate_final_rubric(
            json.loads(final_artifact.read_text(encoding="utf-8"))
        )
        return final, {
            "status": "reused_valid_final_rubric",
            "artifact": str(final_artifact.resolve()),
            "semantic_attempts": [],
            "hierarchical_batches": [],
            "levels": 0,
        }
    current = list(rubrics)
    all_traces = []
    level = 1
    while len(current) > DEEP_REDUCE_BATCH_SIZE:
        batch_size = REDUCE_BATCH_SIZE if level == 1 else DEEP_REDUCE_BATCH_SIZE
        next_level = []
        for start in range(0, len(current), batch_size):
            batch = current[start:start + batch_size]
            batch_number = start // batch_size + 1
            if len(batch) == 1:
                next_level.append(batch[0])
                all_traces.append({
                    "level": level, "batch": batch_number,
                    "status": "singleton_passthrough",
                })
                continue
            prefix = f"reduce_level_{level:02d}_batch_{batch_number:03d}"
            artifact = out / f"{prefix}.json"
            raw_artifact = out / f"{prefix}_raw.json"
            trace_artifact = out / f"{prefix}_trace.json"
            if artifact.exists():
                reduced = normalize_rubric(json.loads(artifact.read_text(encoding="utf-8")))
                trace = {"level": level, "batch": batch_number, "status": "reused_standard"}
            elif raw_artifact.exists():
                reduced = normalize_rubric(json.loads(raw_artifact.read_text(encoding="utf-8")))
                atomic_json(artifact, reduced)
                trace = {"level": level, "batch": batch_number, "status": "recovered_from_raw"}
            else:
                prompt = build_reduce_prompt(
                    batch, round_index,
                    f"hierarchical reduce level {level} batch {batch_number}",
                    advisor, "",
                )
                raw, mario_trace = client.run(
                    agent_id, SYSTEM_PROMPT, prompt, [],
                    f"ECR RSI round {round_index} reduce L{level} B{batch_number}",
                )
                atomic_json(raw_artifact, raw)
                reduced = normalize_rubric(raw)
                atomic_json(artifact, reduced)
                atomic_json(trace_artifact, mario_trace)
                trace = {
                    "level": level, "batch": batch_number,
                    "status": "success", **mario_trace,
                }
            next_level.append(reduced)
            all_traces.append(trace)
        current = next_level
        level += 1

    final_prompt = build_reduce_prompt(
        current, round_index, "final reduce", advisor, previous,
    )
    raw = None
    final = None
    final_attempts = []
    prompt = final_prompt
    for attempt in range(FINAL_SCHEMA_RETRIES + 1):
        raw, attempt_trace = client.run(
            agent_id, SYSTEM_PROMPT, prompt, [],
            f"ECR RSI round {round_index} final reduce attempt {attempt + 1}",
        )
        suffix = "" if attempt == 0 else f"_repair_{attempt:02d}"
        atomic_json(out / f"reduce{suffix}_raw.json", raw)
        atomic_json(out / f"reduce{suffix}_trace.json", attempt_trace)
        try:
            final = validate_final_rubric(raw)
            final_attempts.append({
                "attempt": attempt + 1, "status": "success", **attempt_trace,
            })
            break
        except ValueError as exc:
            final_attempts.append({
                "attempt": attempt + 1,
                "status": "schema_rejected",
                "error": f"{type(exc).__name__}: {exc}",
                **attempt_trace,
            })
            if attempt >= FINAL_SCHEMA_RETRIES:
                raise ValueError(
                    f"final Reduce remained invalid after {attempt + 1} attempts: {exc}"
                ) from exc
            prompt = final_repair_prompt(final_prompt, raw, exc)
    if final is None:
        raise AssertionError("final rubric validation loop ended without a result")
    final_trace = dict(final_attempts[-1])
    final_trace["semantic_attempts"] = final_attempts
    final_trace["hierarchical_batches"] = all_traces
    final_trace["levels"] = level
    return final, final_trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caller", required=True)
    parser.add_argument("--agent-id", default=os.environ.get("MARIO_AGENT_ID"))
    parser.add_argument("--env", choices=sorted(BASE_URLS), default="prt-idc")
    parser.add_argument("--base-url")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-csv")
    input_group.add_argument("--input-manifest")
    parser.add_argument("--video-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--videos-per-group", type=int, default=8)
    parser.add_argument("--max-training-videos", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument(
        "--selection-mode", choices=("extremes", "stratified_full"),
        default="extremes",
    )
    parser.add_argument("--advisor-instructions")
    parser.add_argument("--previous-rubric")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.agent_id:
        parser.error("set --agent-id or MARIO_AGENT_ID")
    if args.groups * args.videos_per_group > args.max_training_videos:
        parser.error("groups * videos-per-group exceeds --max-training-videos")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "run_manifest.json"
    if args.input_csv and not args.video_dir:
        parser.error("--video-dir is required with --input-csv")
    if args.selection_mode == "stratified_full" and not args.input_manifest:
        parser.error("stratified_full requires --input-manifest")
    input_path = Path(args.input_manifest or args.input_csv)
    identity = {
        "round": args.round,
        "input_csv": str(Path(args.input_csv).resolve()) if args.input_csv else None,
        "input_manifest": str(Path(args.input_manifest).resolve()) if args.input_manifest else None,
        "input_size": input_path.stat().st_size,
        "video_dir": str(Path(args.video_dir).resolve()) if args.video_dir else None,
        "candidate_seed": args.seed,
        "selection_mode": args.selection_mode,
        "map_groups": args.groups,
        "videos_per_group": args.videos_per_group,
        "max_training_videos": args.max_training_videos,
        "previous_rubric": str(Path(args.previous_rubric).resolve()) if args.previous_rubric else None,
        "advisor_instructions": str(Path(args.advisor_instructions).resolve()) if args.advisor_instructions else None,
    }
    existing = [item for item in out.iterdir() if item != manifest_path]
    if args.resume:
        if not manifest_path.exists():
            parser.error("--resume requires an existing run_manifest.json")
        previous_identity = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_identity.get("identity") != identity:
            parser.error("resume settings differ from run_manifest.json; use a new output directory")
    elif existing or manifest_path.exists():
        parser.error("output directory is not empty; choose another or use --resume")
    atomic_json(manifest_path, {"identity": identity, "status": "running"})
    advisor = Path(args.advisor_instructions).read_text(encoding="utf-8") \
        if args.advisor_instructions else "本轮无额外 Advisor 修改要求。"
    previous = ""
    if args.previous_rubric:
        previous = "上一轮 rubric（只按 Advisor 要求修改，稳定部分应保留）：\n" + \
            Path(args.previous_rubric).read_text(encoding="utf-8")

    if args.input_manifest:
        rows = load_rsi_samples(args.input_manifest)
        if any(item["ecr"] is None for item in rows):
            raise ValueError("distillation manifest requires ecr for every sample")
    else:
        rows = load_rows(
            args.input_csv, args.video_dir, args.max_training_videos, args.seed
        )
    if args.selection_mode == "stratified_full":
        groups = select_stratified_full_groups(
            rows, args.groups, args.videos_per_group, args.seed
        )
    else:
        groups = select_groups(rows, args.groups, args.videos_per_group)
    client = MarioClient(api_key_from_env(), args.caller, args.base_url or BASE_URLS[args.env])
    map_rubrics = []
    provenance_path = out / "provenance.json"
    provenance = []
    if args.resume and provenance_path.exists():
        loaded = json.loads(provenance_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            parser.error("existing provenance.json must be a list")
        provenance = loaded
    traced_groups = {item.get("group") for item in provenance if isinstance(item, dict)}
    for index, samples in enumerate(groups, 1):
        artifact = out / f"map_{index:03d}.json"
        raw_artifact = out / f"map_{index:03d}_raw.json"
        if args.resume and artifact.exists():
            map_rubrics.append(normalize_rubric(json.loads(artifact.read_text(encoding="utf-8"))))
            if index not in traced_groups:
                provenance.append({
                    "group": index, "ids": [item["id"] for item in samples],
                    "status": "recovered_artifact_without_trace",
                })
                atomic_json(provenance_path, provenance)
            continue
        existing_raw = None
        if args.resume and raw_artifact.exists():
            existing_raw = json.loads(raw_artifact.read_text(encoding="utf-8"))
            try:
                rubric = validate_map_rubric(existing_raw)
                atomic_json(artifact, rubric)
                map_rubrics.append(rubric)
                if index not in traced_groups:
                    provenance.append({
                        "group": index, "ids": [item["id"] for item in samples],
                        "status": "recovered_from_raw",
                    })
                    atomic_json(provenance_path, provenance)
                continue
            except ValueError:
                pass
        labels = "\n".join(
            (
                f"{n + 1}. ECR_BIN={item.get('ecr_bin', 'extreme')} "
                f"id={item['id']}, ECR={item['ecr']:.6f}, "
                f"title={item['title'][:120]}"
            )
            for n, item in enumerate(samples)
        )
        prompt = MAP_TEMPLATE.format(
            round_index=args.round, group_index=index, labels=labels,
            advisor=advisor, previous=previous,
        )
        prefix = f"map_{index:03d}"
        rubric, trace = run_map_with_repair(
            client, args.agent_id, prompt,
            [item["video"] for item in samples],
            f"ECR RSI round {args.round} map {index}", out, prefix,
            existing_raw=existing_raw,
        )
        atomic_json(artifact, rubric)
        map_rubrics.append(rubric)
        provenance.append({
            "group": index, "ids": [item["id"] for item in samples],
            **trace,
        })
        atomic_json(provenance_path, provenance)

    final, trace = reduce_hierarchically(
        client, args.agent_id, map_rubrics, out, args.round, advisor, previous
    )
    validate_final_rubric(final)
    atomic_json(out / "final_rubric.json", final)
    atomic_json(out / "reduce_trace.json", trace)
    atomic_json(manifest_path, {
        "identity": identity,
        "status": "completed",
        "candidate_rows_sampled": len(rows),
        "uploaded_video_uses": args.groups * args.videos_per_group,
        "unique_video_ids": sorted({item["id"] for group in groups for item in group}),
    })
    print(out / "final_rubric.json")


if __name__ == "__main__":
    main()
