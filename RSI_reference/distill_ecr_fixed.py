#!/usr/bin/env python3
"""Induce a validated 8-14 dimension ECR rubric with resumable map-reduce."""
import argparse
import json
from pathlib import Path
from typing import List

from pydantic import BaseModel

from ecr_distill_common import (
    add_common_args,
    call_llm,
    compatible_validator,
    constrained_list,
    ensure_manifest,
    load_optional_dotenv,
    load_samples,
    restore_state,
    sample_text,
    select_group,
    video_payload,
    write_json,
)


class Dimension(BaseModel):
    name: str
    what_to_look_for: str
    direction: str
    weight_0_1: float
    detection_heuristic: str


class FewShot(BaseModel):
    title: str
    description: str
    expected_ecr_band: str
    why: str


class FinalRubric(BaseModel):
    dimensions: constrained_list(Dimension, 8, 14)
    scoring_formula: str
    few_shot_calibration: List[FewShot]
    caveats: List[str]


MAP_PROMPT = '你会看到 {n_high} 条 HIGH ECR 视频（标签 HIGH #1..#{n_high}，ECR 值已附）和 {n_low} 条 LOW ECR 视频（标签 LOW #1..#{n_low}）。\n\n**任务**：从对比这两组视频中，归纳出一份**通用、可复用**的 ECR 评分规则（rubric），供另一个未训练过的模型直接照着打分。\n\n**关键要求**：\n1. **必须跨样本归纳**，不能只看某一条视频。每个维度请至少在 2 条视频里被验证（高组 or 低组），并在 detection_heuristic 里明确点名 "如 HIGH #3、LOW #1 所示"。\n2. **必须捕捉视觉/听觉信号**（画面、动作、剪辑、音效、人物表情、色彩），标题/描述只作为次要输入。\n3. 输出 **8~14 个维度**，每个维度：\n   - name\n   - what_to_look_for: 这个维度捕捉什么信号、为什么它跟 ECR 相关\n   - direction: positive | negative | u_shaped\n   - weight_0_1: 相对权重（后处理会归一化到和=1.0）\n   - detection_heuristic: **3~6 行的分档标准**，必须包含：\n       * 0.0~0.2 什么表现（举一个视频作证）\n       * 0.4~0.6 什么表现\n       * 0.8~1.0 什么表现（举一个视频作证）\n4. **scoring_formula**：给出可执行的合成公式，例如 "raw = Σ w_i*s_i; 若\'开头钩子\' s<0.3 则 raw*=0.7; ECR_pred = clip(raw, 0, 1)"\n5. **few_shot_calibration**：至少 6 条（low/mid/high 各 2 条），基于你看到的真实视频\n6. **caveats**：至少 3 条使用注意事项\n\n**禁止**：\n- 只从单个视频抽维度\n- 使用模糊描述如"节奏快"、"氛围好"，必须落到可观察的具体特征\n- dimensions 数组为空\n\n严格按 FinalRubric schema 输出 JSON。'

MAP_CARDINALITY_GUARD = """

【维度数量与粒度硬要求】
- 最终建议输出 10~12 个维度，绝对不得少于 8 个，也不得超过 14 个。
- 仅输出 5~7 个宏观类别（例如只给 Hook/Narrative/Visual/Emotion/Production）视为不合格。
- 在生成最终 JSON 前，先在内部从样本中枚举至少 12 个候选可观察信号，再按机制去重为 8~14 个最终维度；不要输出内部枚举过程。
- 只有当两个候选的作用机制、观察时间窗口和主要模态都基本相同时，才允许合并。
- 时间窗口不同（如首帧、前1秒、前3秒、中段）或模态不同（视觉、声音、字幕、语义）的信号应优先拆开保留。
- 禁止用同义改写、名称拆词或重复定义凑数量；每个维度必须具有独立观察证据和独立判分标准。
- 输出 JSON 前必须自行核对 dimensions 数量；若少于 8 个，继续从尚未覆盖的具体信号中补充后再提交。
"""

REDUCE_CARDINALITY_GUARD = """

【最终融合数量硬要求】
- 最终建议保留 10~12 个互补维度，绝对不得少于 8 个，也不得超过 14 个。
- 不得把 opening、narrative、visual、audio、emotion、clarity 等不同机制过度压缩成 5~7 个宏观类别。
- 仅当作用机制、时间窗口和主要模态均基本一致时才合并；不同时间窗口或不同模态的互补信号应分别保留。
- 禁止用同义维度凑数量。输出前必须自行核对 dimensions 数量，不符合 8~14 时不得提交。
"""

REDUCE_PROMPT = '你之前独立地基于 {n_groups} 组不同视频，产出了 {n_groups} 份候选 rubric。现在需要把它们**融合成一份最终的、泛化性最好的 rubric**。\n\n以下是 {n_groups} 份候选 rubric 的维度摘要（每份都是独立从 ~{videos_per_group} 条视频归纳的）：\n\n{groups_summary}\n\n**融合原则**：\n1. **高频维度 = 核心维度**：如果一个维度（或语义相近的维度）在多组里反复出现，说明它是稳定信号，必须保留，权重相对较高。\n2. **组间近义合并**：不同组可能给同一个概念起了不同名字（如 "Hook Strength" vs "Opening Hook" vs "First 3-Second Attention Grab"），请归到一个维度下，取最好的描述。\n3. **低频维度谨慎保留**：只出现在 1~2 组的维度可能是那组样本的偏差，可以降权或丢弃，除非它明显有解释力。\n4. **权重按出现频次 + 各组权重的加权**决定，最终归一化到求和=1.0。\n5. **detection_heuristic 必须保留详细的 0/0.5/1 分档锚点**，取多组里描述最清晰的版本。\n6. **8~14 个最终维度**。\n\n**产出 FinalRubric JSON**：\n- dimensions: 融合后的维度列表，每个维度需 3~6 行的 detection_heuristic 分档\n- scoring_formula: 一段清晰的合成公式（可包含惩罚项 / 非线性项）\n- few_shot_calibration: 至少 6 条，覆盖 low/mid/high 各 2 条\n- caveats: 使用注意事项，必须包含 "本 rubric 融合自 {n_groups} 组、共 ~{total_videos} 条视频的归纳" 这条\n\n严格按 FinalRubric schema 输出。'


def render_prompt(rubric: dict) -> str:
    lines = [
        "# ECR Scoring Rubric (portable)",
        "",
        "你正在预估一条短视频的 ECR（观看超过5秒的用户占比，[0,1]）。",
        "对每个维度打 0~1 分，然后按下方公式合成，最后输出 ECR × 100。",
        "",
        "## 维度",
        "",
    ]
    for d in rubric.get("dimensions", []) or []:
        lines += [
            f"### {d.get('name','?')}  "
            f"(weight={d.get('weight_0_1','?')}, direction={d.get('direction','?')})",
            f"- **看什么:** {d.get('what_to_look_for','')}",
            f"- **怎么判:** {d.get('detection_heuristic','')}",
            "",
        ]
    lines += [
        "## 评分公式",
        "",
        "```",
        str(rubric.get("scoring_formula", "")),
        "```",
        "",
        "## Few-shot 校准锚点",
        "",
        "```json",
        json.dumps(
            rubric.get("few_shot_calibration", []), ensure_ascii=False, indent=2
        ),
        "```",
        "",
        "## 注意事项",
        "",
    ]
    for c in rubric.get("caveats", []) or []:
        lines.append(f"- {c}")
    return "\n".join(lines)


def normalize_weights(rubric):
    if isinstance(rubric, dict) and isinstance(rubric.get("dimensions"), list):
        total = sum(float(d.get("weight_0_1", 0)) for d in rubric["dimensions"]) or 1.0
        for d in rubric["dimensions"]:
            try:
                d["weight_0_1"] = round(float(d["weight_0_1"]) / total, 4)
            except Exception:
                pass
    return rubric


def fixed_samples_text(highs, lows, title_max, desc_max):
    def fmt(items):
        return "\n".join(
            "  ECR={:.4f} | 标题: {} | 描述: {}".format(
                x["ecr"], x["title"][:title_max], x["description"][:desc_max]
            )
            for x in items
        )

    return "HIGH:\n{}\n\nLOW:\n{}".format(fmt(highs), fmt(lows))


def map_group(highs, lows, args):
    prompt = (MAP_PROMPT + MAP_CARDINALITY_GUARD).format(
        n_high=len(highs), n_low=len(lows)
    )
    if args.mode != "video_only":
        prompt += "\n\n" + fixed_samples_text(
            highs, lows, args.title_max, args.desc_max
        )
    videos = [x["url"] for x in highs + lows]
    labels = [
        "HIGH #{} ECR={:.4f}".format(i + 1, x["ecr"]) for i, x in enumerate(highs)
    ] + ["LOW  #{} ECR={:.4f}".format(i + 1, x["ecr"]) for i, x in enumerate(lows)]
    result = call_llm(
        args.model,
        "Round 1：一次性从 HIGH/LOW 视频组归纳完整 rubric。",
        prompt,
        FinalRubric,
        videos if args.mode != "text" else None,
        labels if args.mode != "text" else None,
        args.max_retries,
        schema_name="ecr_final_rubric",
        legacy_server_schema=True,
    )
    return normalize_weights(result)


def reduce_once(groups, args):
    summaries = []
    for index, rubric in enumerate(groups, 1):
        lines = ["### Group {}".format(index)]
        for dimension in rubric.get("dimensions", []):
            lines.append(
                "- {} [w={}, {}]: {}".format(
                    dimension.get("name", "?"),
                    dimension.get("weight_0_1", "?"),
                    dimension.get("direction", "?"),
                    dimension.get("what_to_look_for", "")[:100],
                )
            )
        summaries.append("\n".join(lines))
    prompt = (REDUCE_PROMPT + REDUCE_CARDINALITY_GUARD).format(
        n_groups=len(summaries),
        videos_per_group=args.videos_per_group,
        total_videos=len(summaries) * args.videos_per_group,
        groups_summary="\n\n".join(summaries),
    )
    result = call_llm(
        args.model,
        "Reduce step: 融合多组独立归纳的 rubric 成最终版。",
        prompt,
        FinalRubric,
        retries=args.max_retries,
        schema_name="ecr_final_rubric",
        legacy_server_schema=True,
    )
    return normalize_weights(result)


def reduce_groups(groups, args):
    """Default is one-shot reduce; positive batch size enables hierarchical reduce."""
    if not args.reduce_batch_size or len(groups) <= args.reduce_batch_size:
        return reduce_once(groups, args)
    current = list(groups)
    level = 1
    while len(current) > args.reduce_batch_size:
        print(
            "  hierarchical reduce level {}: {} inputs, batch_size={}".format(
                level, len(current), args.reduce_batch_size
            ),
            flush=True,
        )
        current = [
            reduce_once(current[i : i + args.reduce_batch_size], args)
            for i in range(0, len(current), args.reduce_batch_size)
        ]
        level += 1
    return reduce_once(current, args)


def finalize_provenance(final, out, provenance, group_count, args):
    """Replace model-invented sample counts with authoritative local metadata."""
    successful_numbers = {
        int(match.group(1))
        for path in out.glob("group*_rubric.json")
        for match in [__import__("re").search(r"group(\d+)_rubric\.json$", path.name)]
        if match
    }
    actual_video_count = 0
    for entry in provenance:
        if entry.get("group") in successful_numbers:
            actual_video_count += len(entry.get("high_ids", []))
            actual_video_count += len(entry.get("low_ids", []))
    if not actual_video_count:
        actual_video_count = group_count * args.videos_per_group

    caveats = []
    for item in final.get("caveats", []) or []:
        text = str(item)
        lower = text.lower()
        if ("derived" in lower or "sample" in lower or "融合自" in text) and (
            "video" in lower or "视频" in text
        ):
            continue
        caveats.append(text)
    authoritative = "本 rubric 融合自 {} 个成功组、共 {} 条入模视频的归纳。".format(
        group_count, actual_video_count
    )
    caveats.append(authoritative)
    final["caveats"] = caveats
    metadata = {
        "successful_group_count": group_count,
        "input_video_count": actual_video_count,
        "videos_per_group_configured": args.videos_per_group,
        "reduce_strategy": (
            "one_shot" if not args.reduce_batch_size else "hierarchical"
        ),
        "reduce_batch_size": args.reduce_batch_size,
        "source": "computed locally from group artifacts and provenance",
    }
    write_json(out / "final_rubric_meta.json", metadata)
    return final


def main(args):
    if args.videos_per_group < 4 or args.videos_per_group % 2:
        raise ValueError("--videos_per_group 必须是 >=4 的偶数")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ensure_manifest(out, args, "fixed-8-to-14")
    samples = load_samples(args.input_file, args.col_names)
    high_to_low = sorted(samples, key=lambda x: x["ecr"], reverse=True)
    low_to_high = list(reversed(high_to_low))
    groups, provenance, used, start = restore_state(
        out,
        "group*_rubric.json",
        r"group(\d+)_rubric\.json$",
        "group_provenance.json",
        "dimensions",
        args.resume,
    )

    each = args.videos_per_group // 2
    for number in range(start, args.n_groups + 1):
        print(f"\n===== GROUP {number}/{args.n_groups} =====", flush=True)
        highs, lows = select_group(high_to_low, low_to_high, each, used, args)
        if len(highs) < 2 or len(lows) < 2:
            print("fewer than 2 fresh/alive samples in HIGH or LOW; stopping")
            break
        if len(highs) < each or len(lows) < each:
            print(
                "  continuing with partial group: HIGH={}, LOW={}".format(
                    len(highs), len(lows)
                ),
                flush=True,
            )
        selected = {x["id"] for x in highs + lows}
        used.update(selected)
        attempt_entry = {
            "group": number,
            "status": "pending",
            "high_ids": [x["id"] for x in highs],
            "low_ids": [x["id"] for x in lows],
        }
        # Persist consumption before entering LLM. A crash, empty response or
        # validation failure must never make these videos eligible again.
        provenance.append(attempt_entry)
        write_json(out / "group_provenance.json", provenance)
        try:
            rubric = map_group(highs, lows, args)
            write_json(out / f"group{number:03d}_rubric.json", rubric)
            groups.append(rubric)
            attempt_entry["status"] = "success"
            write_json(out / "group_provenance.json", provenance)
        except Exception as exc:
            attempt_entry["status"] = "failed"
            attempt_entry["error"] = "{}: {}".format(type(exc).__name__, exc)
            write_json(out / "group_provenance.json", provenance)
            print(
                "  group {} failed and was skipped; selected videos remain "
                "consumed: {}".format(number, exc),
                flush=True,
            )
            continue
        if (
            args.checkpoint_every
            and number % args.checkpoint_every == 0
            and len(groups) >= 2
        ):
            try:
                final = reduce_groups(groups, args)
                final = finalize_provenance(final, out, provenance, len(groups), args)
                write_json(out / f"checkpoint_after_group{number:03d}.json", final)
                write_json(out / "final_rubric.json", final)
                (out / "reusable_prompt.md").write_text(
                    render_prompt(final), encoding="utf-8"
                )
            except Exception as exc:
                print("  checkpoint reduce failed: {}".format(exc), flush=True)

    if not groups:
        raise RuntimeError("no valid group rubrics")
    try:
        final = reduce_groups(groups, args)
    except Exception as exc:
        existing_final = out / "final_rubric.json"
        if existing_final.exists():
            raise RuntimeError(
                "final reduce failed; existing final/checkpoint was preserved and "
                "was NOT replaced by a single group: {}".format(exc)
            )
        raise RuntimeError(
            "final reduce failed; no final artifact was written: {}".format(exc)
        )
    final = finalize_provenance(final, out, provenance, len(groups), args)
    write_json(out / "final_rubric.json", final)
    (out / "reusable_prompt.md").write_text(render_prompt(final), encoding="utf-8")
    print(f"done: {out / 'final_rubric.json'}")


if __name__ == "__main__":
    load_optional_dotenv()
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument(
        "--reduce_batch_size",
        type=int,
        default=0,
        help="0=保持原版一次性Reduce；>0时显式启用分层Reduce",
    )
    main(parser.parse_args())
