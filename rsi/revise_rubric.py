#!/usr/bin/env python3
"""Revise a distilled rubric using training-only small-model feedback."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rsi.distill import (
    FINAL_SCHEMA_RETRIES, SYSTEM_PROMPT, atomic_json, final_repair_prompt,
    validate_final_rubric,
)
from rsi.mario import BASE_URLS, MarioClient, api_key_from_env

REVISION_PROMPT = """这是 RSI Round 3 的小模型闭环修订阶段。

候选 rubric 来自实际覆盖 500 条训练视频的分层 Map/Reduce：
{rubric}

以下反馈来自与蒸馏视频完全不重叠的 500 条训练校准视频，由最终本地小参数 VideoLLaMA2 实际执行候选维度后生成：
{feedback}

必须根据执行证据修订，而不是自由重新命名：
1. decision=delete 的维度必须删除；
2. decision=split_or_simplify 的维度必须改写为一跳、直接可见/可听状态，或拆成互补的原子维度；
3. retain_as_binary_gate 应保留为明确 gate，不得伪造连续精度；
4. high small_model_reasoning_risk 不得保留相同复杂推理要求；
5. 优先保留跨样本稳定、有 ECR 正向证据且格式稳定的机制；
6. 不得引入 OCR、ASR、光流、RMS、频谱、Sobel、帧差或外部工具；
7. 推荐分组只是语义建议，后续会独立做分治消融。

只输出完整 JSON rubric。顶层第一个字段必须是 dimensions，含 6~12 个完整维度；同时包含 scoring_formula、few_shot_calibration、caveats、recommended_call_groups、revision_summary。所有权重和为 1。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caller", required=True)
    parser.add_argument("--agent-id", default=os.environ.get("MARIO_AGENT_ID"))
    parser.add_argument("--env", choices=sorted(BASE_URLS), default="prt-idc")
    parser.add_argument("--base-url")
    parser.add_argument("--rubric", required=True)
    parser.add_argument("--feedback", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--round", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.agent_id:
        parser.error("set --agent-id or MARIO_AGENT_ID")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / "final_rubric.json"
    if args.resume and final_path.exists():
        validate_final_rubric(json.loads(final_path.read_text(encoding="utf-8")))
        print(final_path)
        return
    rubric = json.loads(Path(args.rubric).read_text(encoding="utf-8"))
    feedback = json.loads(Path(args.feedback).read_text(encoding="utf-8"))
    prompt = REVISION_PROMPT.format(
        rubric=json.dumps(rubric, ensure_ascii=False, indent=2),
        feedback=json.dumps(feedback, ensure_ascii=False, indent=2),
    )
    client = MarioClient(
        api_key_from_env(), args.caller, args.base_url or BASE_URLS[args.env]
    )
    final = None
    traces = []
    active_prompt = prompt
    for attempt in range(FINAL_SCHEMA_RETRIES + 1):
        raw, trace = client.run(
            args.agent_id, SYSTEM_PROMPT, active_prompt, [],
            f"ECR RSI round {args.round} small-model rubric revision attempt {attempt + 1}",
        )
        suffix = "" if attempt == 0 else f"_repair_{attempt:02d}"
        atomic_json(output / f"revision{suffix}_raw.json", raw)
        atomic_json(output / f"revision{suffix}_trace.json", trace)
        try:
            final = validate_final_rubric(raw)
            traces.append({"attempt": attempt + 1, "status": "success", **trace})
            break
        except ValueError as exc:
            traces.append({
                "attempt": attempt + 1, "status": "schema_rejected",
                "error": f"{type(exc).__name__}: {exc}", **trace,
            })
            if attempt >= FINAL_SCHEMA_RETRIES:
                raise
            active_prompt = final_repair_prompt(prompt, raw, exc)
    if final is None:
        raise AssertionError("rubric revision ended without a valid final rubric")
    atomic_json(output / "revision_attempts.json", traces)
    atomic_json(final_path, final)
    atomic_json(output / "run_manifest.json", {
        "status": "completed",
        "round": args.round,
        "source_rubric": str(Path(args.rubric).resolve()),
        "source_feedback": str(Path(args.feedback).resolve()),
        "dimensions": len(final["dimensions"]),
    })
    print(final_path)


if __name__ == "__main__":
    main()
