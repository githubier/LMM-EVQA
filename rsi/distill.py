#!/usr/bin/env python3
"""Budgeted, resumable Mario Map/Reduce distillation for one RSI round."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsi.mario import BASE_URLS, MarioClient, api_key_from_env
from rsi.rubric import normalize_rubric

SYSTEM_PROMPT = """你是 Qwen3.8-Max ECR Rubric Distiller。ECR 是观看超过 5 秒的用户比例。
你要从带真实标签的短视频中总结可迁移、可观察、可由本地多模态模型执行的预测维度。
只能输出一个合法 JSON 对象，不要输出 Markdown。维度必须区分机制、模态和时间窗口，禁止同义重复。"""

MAP_TEMPLATE = """这是 RSI 第 {round_index} 轮的第 {group_index} 个 Map 组。
附件按下面顺序对应样本：
{labels}

请对比 HIGH 与 LOW 样本，输出 8~14 维候选 rubric。JSON 必须包含：
- dimensions: 每项含 key、name、what_to_look_for、direction(positive/negative/u_shaped)、weight_0_1、modality(video/audio/audio_visual/text/multimodal)、window_seconds(0,30]、detection_heuristic；判分锚点必须统一为 0=不利于继续观看、1=有利于继续观看，direction 只描述原始观察信号与 ECR 的关系；
- scoring_formula；few_shot_calibration；caveats；
- recommended_call_groups: 用 key 将互补且观察窗口接近的维度分成调用组，每个 key 恰好出现一次；
- revision_summary。
每个判分标准必须包含低/中/高锚点，并引用本组至少两个不同样本的可观察证据。
{advisor}
{previous}
"""

REDUCE_TEMPLATE = """这是 RSI 第 {round_index} 轮的 Reduce。以下是 {count} 个独立 Map rubric：
{rubrics}

融合为一个最终 JSON rubric。保留跨组稳定信号，去除同义重复；输出 8~14 个互补维度，权重和应为 1；为每个维度确定主要模态和 (0,30] 秒观察窗口；所有维度评分统一为 0=不利于继续观看、1=有利于继续观看；recommended_call_groups 中每个 key 恰好出现一次。不要虚构没有出现在 Map 结果中的视频证据。只输出 JSON。
{advisor}
{previous}
"""


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
            video = next((item for item in candidates if item.is_file()), None)
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


def compact_rubric(rubric: dict) -> dict:
    return {
        "dimensions": rubric["dimensions"],
        "scoring_formula": rubric["scoring_formula"],
        "recommended_call_groups": rubric["recommended_call_groups"],
        "caveats": rubric["caveats"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caller", required=True)
    parser.add_argument("--agent-id", default=os.environ.get("MARIO_AGENT_ID"))
    parser.add_argument("--env", choices=sorted(BASE_URLS), default="prt-idc")
    parser.add_argument("--base-url")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--videos-per-group", type=int, default=8)
    parser.add_argument("--max-training-videos", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261026)
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
    identity = {
        "round": args.round,
        "input_csv": str(Path(args.input_csv).resolve()),
        "input_size": Path(args.input_csv).stat().st_size,
        "video_dir": str(Path(args.video_dir).resolve()),
        "candidate_seed": args.seed,
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

    rows = load_rows(
        args.input_csv, args.video_dir, args.max_training_videos, args.seed
    )
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
        if args.resume and artifact.exists():
            map_rubrics.append(normalize_rubric(json.loads(artifact.read_text(encoding="utf-8"))))
            if index not in traced_groups:
                provenance.append({
                    "group": index, "ids": [item["id"] for item in samples],
                    "status": "recovered_artifact_without_trace",
                })
                atomic_json(provenance_path, provenance)
            continue
        labels = "\n".join(
            f"{n + 1}. {band} #{band_index}: id={item['id']}, "
            f"ECR={item['ecr']:.6f}, title={item['title'][:120]}"
            for n, item in enumerate(samples)
            for band, band_index in [(
                ("HIGH", n + 1) if n < len(samples) // 2
                else ("LOW", n + 1 - len(samples) // 2)
            )]
        )
        prompt = MAP_TEMPLATE.format(
            round_index=args.round, group_index=index, labels=labels,
            advisor=advisor, previous=previous,
        )
        raw, trace = client.run(
            args.agent_id, SYSTEM_PROMPT, prompt,
            [item["video"] for item in samples],
            f"ECR RSI round {args.round} map {index}",
        )
        atomic_json(out / f"map_{index:03d}_raw.json", raw)
        rubric = normalize_rubric(raw)
        atomic_json(artifact, rubric)
        map_rubrics.append(rubric)
        provenance.append({
            "group": index, "ids": [item["id"] for item in samples],
            "status": "success", **trace,
        })
        atomic_json(provenance_path, provenance)

    summaries = "\n\n".join(
        f"MAP {index}:\n{json.dumps(compact_rubric(item), ensure_ascii=False)}"
        for index, item in enumerate(map_rubrics, 1)
    )
    raw, trace = client.run(
        args.agent_id, SYSTEM_PROMPT,
        REDUCE_TEMPLATE.format(
            round_index=args.round, count=len(map_rubrics), rubrics=summaries,
            advisor=advisor, previous=previous,
        ), [], f"ECR RSI round {args.round} reduce",
    )
    atomic_json(out / "reduce_raw.json", raw)
    final = normalize_rubric(raw)
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
