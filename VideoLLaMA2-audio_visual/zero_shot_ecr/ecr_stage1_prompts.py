"""Prompt builders and temporal-window metadata for stage-1 ECR experiments."""

from ecr_rubric_prompt import (
    CAVEATS as CAVEATS_5,
    DIMENSIONS as DIMENSIONS_5,
    FEWSHOT_ANCHORS as FEWSHOT_5,
    ROLE_PLAY as ROLE_5,
    SCORING_FORMULA as FORMULA_5,
    VIDEO_INFO as VIDEO_INFO_5,
    build_rubric_prompt as build_full_5,
)
from ecr_rubric_12_prompt import (
    CAVEATS as CAVEATS_12,
    DIMENSIONS as DIMENSIONS_12,
    FEWSHOT_ANCHORS as FEWSHOT_12,
    ROLE_PLAY as ROLE_12,
    SCORING_FORMULA as FORMULA_12,
    VIDEO_INFO as VIDEO_INFO_12,
    build_rubric_prompt as build_full_12,
)

# A dimension sees only the evidence window requested by its rubric. Dimensions
# without an explicit bound use the first five seconds because ECR itself is
# defined by continuation beyond five seconds.
WINDOWS_5 = {
    "Hook_Strength": 3.0,
    "Narrative_Progression": 5.0,
    "Visual_Dynamism": 5.0,
    "Emotional_Resonance": 5.0,
    "Production_Quality": 5.0,
}

WINDOWS_12 = {
    "First_Frame_Hook_Strength": 0.5,
    "Visual_Pacing_Camera_Dynamism": 3.0,
    "Facial_Expression_Intensity_Synchronization": 2.0,
    "Text_Overlay_Timing_Emotional_Charge": 1.0,
    "Narrative_Arc_Completion": 5.0,
    "Audio_Visual_Sync_Impact": 2.0,
    "Color_Contrast_Saturation": 3.0,
    "Interpersonal_Interaction_Density": 3.0,
    "Physical_Motion_Intensity_Consequence": 3.0,
    "Topic_Relatability_Concept_Accessibility": 2.0,
    "Production_Quality_Technical_Clarity": 5.0,
    "Direct_Viewer_Engagement_CTA": 5.0,
}


def get_config(rubric_size):
    if rubric_size == 5:
        return {
            "role": ROLE_5,
            "dimensions": DIMENSIONS_5,
            "fewshot": FEWSHOT_5,
            "caveats": CAVEATS_5,
            "formula": FORMULA_5,
            "video_info": VIDEO_INFO_5,
            "full_builder": build_full_5,
            "windows": WINDOWS_5,
        }
    if rubric_size == 12:
        return {
            "role": ROLE_12,
            "dimensions": DIMENSIONS_12,
            "fewshot": FEWSHOT_12,
            "caveats": CAVEATS_12,
            "formula": FORMULA_12,
            "video_info": VIDEO_INFO_12,
            "full_builder": build_full_12,
            "windows": WINDOWS_12,
        }
    raise ValueError(f"unsupported rubric size: {rubric_size}")


def build_direct_continuous_prompt(
    rubric_size, title, description, use_fewshot, input_scope="视频开头5秒"
):
    """Complete rubric in one call; force a continuous ECR in [0, 1]."""
    config = get_config(rubric_size)
    base = config["full_builder"](title, description, use_fewshot=use_fewshot)
    return base + (
        f"\n\n重要补充要求：你当前看到的是{input_scope}。请不要输出 low/mid/high 档位，也不要照抄 few-shot 的 ECR。"
        "必须根据当前视频独立计算并输出 0 到 1 之间的连续小数，至少保留三位小数。"
        "最终仅按如下格式输出：ECR: 0.xxx ; reason: 对实际画面的简短说明"
    )


def build_direct_ordinal_prompt(
    rubric_size, title, description, use_fewshot, input_scope="视频开头5秒"
):
    """Complete rubric in one call; next token is an ECR ordinal label."""
    config = get_config(rubric_size)
    parts = [
        config["role"],
        "\n\n请综合以下全部维度判断视频开头5秒后的观看延续率。\n\n",
        "\n\n".join(item[3] for item in config["dimensions"]),
        "\n\n",
        config["formula"],
    ]
    if use_fewshot:
        parts.append(config["fewshot"])
    parts.extend([
        config["caveats"],
        config["video_info"].format(title=title, description=description),
        f"\n你当前看到的是{input_scope}。综合所有维度后，将 ECR 评为 low、medium 或 high。",
        "\n不要解释，不要输出数字。下一项只能填写一个英文等级词。\nECR rating:",
    ])
    return "".join(parts)


def build_dimension_continuous_prompt(
    rubric_size, title, description, dimension_index, use_fewshot,
    input_scope=None,
) -> str:
    """One dimension; model generates a continuous score in [0, 1]."""
    config = get_config(rubric_size)
    key, name, _, instruction = config["dimensions"][dimension_index]
    window = config["windows"][key]
    input_scope = input_scope or f"视频开头 {window:g} 秒"
    parts = [
        config["role"],
        f"\n\n你当前看到的是{input_scope}。本次只评估一个维度，不要计算最终 ECR。\n\n",
        instruction,
    ]
    if use_fewshot:
        parts.append(config["fewshot"])
    parts.extend([
        config["caveats"],
        config["video_info"].format(title=title, description=description),
        "\n请仅根据当前时间窗口内实际可见、可听的证据评分。",
        "不要把分数限制在 0、0.5、1 三档，而要输出 0 到 1 之间的连续小数，至少保留三位小数。",
        f"\n最终仅按如下格式输出：dimension: {key} ; score: 0.xxx ; reason: 简短证据",
    ])
    return "".join(parts)


def build_dimension_ordinal_prompt(
    rubric_size, title, description, dimension_index, use_fewshot,
    input_scope=None,
) -> str:
    """One dimension; next token is low, medium, or high."""
    config = get_config(rubric_size)
    key, name, _, instruction = config["dimensions"][dimension_index]
    window = config["windows"][key]
    input_scope = input_scope or f"视频开头 {window:g} 秒"
    parts = [
        config["role"],
        f"\n\n你当前看到的是{input_scope}。本次只评估下面一个维度。\n\n",
        instruction,
    ]
    if use_fewshot:
        parts.append(config["fewshot"])
    parts.extend([
        config["caveats"],
        config["video_info"].format(title=title, description=description),
        "\n请仅根据当前时间窗口内实际可见、可听的证据，将该维度评为 low、medium 或 high。",
        "\n不要解释，不要输出数字。下一项只能填写一个英文等级词。",
        f"\n{key} rating:",
    ])
    return "".join(parts)


def build_dimension_discrete_prompt(
    rubric_size, title, description, dimension_index, use_fewshot, window_text
):
    """One dimension; model generates exactly one of 0, 0.5, or 1."""
    config = get_config(rubric_size)
    key, _, _, instruction = config["dimensions"][dimension_index]
    parts = [
        config["role"],
        f"\n\n你当前看到{window_text}。本次只评估下面一个维度，不要计算最终 ECR。\n\n",
        instruction,
    ]
    if use_fewshot:
        parts.append(config["fewshot"])
    parts.extend([
        config["caveats"],
        config["video_info"].format(title=title, description=description),
        "\n只能依据实际可见、可听证据，并严格从 0、0.5、1 中选择一个分数。",
        f"\n最终仅按如下格式输出：dimension: {key} ; score: 0/0.5/1 ; reason: 简短证据",
    ])
    return "".join(parts)


def build_direct_discrete_prompt(
    rubric_size, title, description, use_fewshot,
    input_scope="视频开头5秒",
):
    """Original complete rubric: discrete dimensions and one generated ECR."""
    return get_config(rubric_size)["full_builder"](
        title, description, use_fewshot=use_fewshot
    ) + f"\n\n输入范围说明：你当前看到的是{input_scope}，只能依据该范围内的视听证据评分。"


def build_llm_aggregate_prompt(
    rubric_size, title, description, results, use_fewshot,
    include_dimensions=False,
):
    """Aggregate continuous per-dimension scores using an additional LLM call."""
    config = get_config(rubric_size)
    lines = []
    for key, name, weight, _ in config["dimensions"]:
        item = results[key]
        lines.append(
            f"- {key} ({name}, weight={weight}): score={item['score']:.6f}; "
            f"evidence={item.get('reason', item.get('probabilities', ''))}"
        )
    parts = [config["role"], "\n\n"]
    if include_dimensions:
        parts.extend([
            "## 完整维度判断说明\n\n",
            "\n\n".join(item[3] for item in config["dimensions"]),
            "\n\n",
        ])
    parts.extend([
        "## 前置分维度评分结果\n",
        "\n".join(lines),
        "\n\n",
        config["formula"],
    ])
    if use_fewshot:
        parts.append(config["fewshot"])
    parts.extend([
        config["caveats"],
        config["video_info"].format(title=title, description=description),
        "\n必须使用给定 score 计算，不得重打维度分。输出 0 到 1 之间至少三位小数的连续 ECR，",
        "不要照抄 few-shot 数字。最终仅按如下格式输出：ECR: 0.xxx ; reason: 简短说明",
    ])
    return "".join(parts)
