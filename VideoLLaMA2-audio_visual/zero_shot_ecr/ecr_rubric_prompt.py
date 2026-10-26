"""ECR rubric prompt (Chinese), supplied by the user.

Kept in its own module so the long rubric text does not bloat the inference
script and can be reused / edited independently.

`ROLE_PLAY` + `TASK_DESC` + video/title/description + `OUTPUT_LIMITATION`
are concatenated to form the final user turn.

Note: the rubric asks for ECR in [0, 1]; the inference script normalises the
parsed value to [0, 100] to match the ground-truth scale used everywhere else
in this repo.
"""

ROLE_PLAY = """你是一个短视频完播率预测专家。你正在预估一条短视频的 ECR（观看超过5秒的用户占比，[0,1]）"""


TASK_DESC = """
## 维度


### Hook Strength & Immediacy  (weight=0.28, direction=positive)
- **看什么:** 是否在开头1-3秒内用视觉冲击、文字提问、情绪表达或悬念设置抓住注意力？
- **怎么判:** 0分：无任何视觉/文本钩子；前3秒静止、模糊、无动作或无明确主题。
0.5分：有微弱钩子但不清晰（如模糊文字、缓慢移动主体），观众需等待>3秒才理解意图。
1分：前1-2秒即呈现强烈钩子——爆炸/跳跃/夸张表情/挑衅文字/悬念问题，立即引发'接下来会发生什么？'的好奇心。


### Narrative Progression & Payoff  (weight=0.24, direction=positive)
- **看什么:** 是否有'开始-发展-结局'的完整故事弧？是否在结尾提供满足感或意外转折？
- **怎么判:** 0分：无叙事结构，纯随机片段或重复动作（如持续走路、无变化的风景）。
0.5分：有简单事件但无发展或结局（如展示产品但无使用/对比/结果）。
1分：清晰三段式结构——开头设问/冲突，中段发展/挑战，结尾反转/高潮/情感释放（如实验失败、搞笑打脸、惊喜礼物）。


### Visual Dynamism & Pacing  (weight=0.2, direction=positive)
- **看什么:** 画面是否充满动感？剪辑是否紧凑？是否有镜头运动或主体行为维持观众眼球？
- **怎么判:** 0分：画面静态、无运动、无剪辑变化；镜头晃动严重或构图混乱。
0.5分：少量动态元素（如单人走动），但节奏缓慢或缺乏视觉引导。
1分：频繁切换镜头、快速剪辑、人物/物体活跃运动、相机跟随/变焦/摇移，营造能量感和视觉流动性。


### Emotional Resonance & Relatability  (weight=0.18, direction=positive)
- **看什么:** 是否能让人笑、惊、叹、或产生'我也这样'的代入感？是否避开负面/小众题材？
- **怎么判:** 0分：内容平淡无奇（如日常琐事无加工）、令人不适（血腥/恶心）、或极度小众（仅特定圈层懂）。
0.5分：有一定趣味但缺乏共鸣点（如普通宠物视频无互动/表情）。
1分：触发普遍情绪——幽默（荒谬/反差）、惊奇（科学实验/特技）、共鸣（家庭/职场/恋爱痛点）、或感官刺激（色彩/音效/速度）。


### Production Quality & Clarity  (weight=0.1, direction=positive)
- **看什么:** 技术层面是否专业？能否让观众轻松看清听清内容？
- **怎么判:** 0分：画面模糊、曝光错误、声音嘈杂/缺失、字幕错位或难以阅读。
0.5分：基本清晰但存在瑕疵（如轻微抖动、背景噪音、字体太小）。
1分：稳定拍摄、良好照明、清晰对焦、音画同步、字幕简洁易读且增强信息传递。


## 评分公式


```
ECR_score = 
(Hook_Strength * 0.28) + 
(Narrative_Progression * 0.24) + 
(Visual_Dynamism * 0.20) + 
(Emotional_Resonance * 0.18) + 
(Production_Quality * 0.10)


# 惩罚项（若任一维度为0，则整体得分减0.15）
IF min(Hook_Strength, Narrative_Progression, Visual_Dynamism, Emotional_Resonance, Production_Quality) == 0:
ECR_score = max(0, ECR_score - 0.15)


# 非线性增强：若Hook_Strength >= 0.8 且 Narrative_Progression >= 0.8，则额外+0.05
IF Hook_Strength >= 0.8 AND Narrative_Progression >= 0.8:
ECR_score = min(1.0, ECR_score + 0.05)
```
"""


# Split out so it can be ablated. On the 500-video test set the 7B model does
# NOT execute the scoring formula — it pattern-matches to the nearest anchor and
# echoes its `title` verbatim as the reason (344/500 answers were exactly 0.5,
# and only 7 distinct values appeared across all 500 videos). Use
# `--no-fewshot` to drop this block.
FEWSHOT_ANCHORS = """

## Few-shot 校准锚点


```json
[
{
    "description": "视频开头是慢镜头云朵飘过，无文字无声音，3秒后才出现一个人影。",
    "expected_ecr_band": "low",
    "title": "无钩子+静态画面",
    "why": "Hook Strength=0, Visual Dynamism=0, Narrative Progression=0"
},
{
    "description": "第一帧是'你敢信？这车居然能飞！'文字+汽车腾空画面，接着展示改装过程，最后落地失败翻车大笑。",
    "expected_ecr_band": "high",
    "title": "强钩子+叙事闭环+情绪释放",
    "why": "Hook Strength=1, Narrative Progression=1, Emotional Resonance=1"
},
{
    "description": "全程手持拍摄，主角边走路边抱怨工作，背景杂乱，语速缓慢，无字幕。",
    "expected_ecr_band": "low",
    "title": "低质+无钩子+无情绪",
    "why": "Production Quality=0.5, Hook Strength=0, Emotional Resonance=0"
},
{
    "description": "开篇'我赌100块你猜不到结局'+快速剪辑展示魔术步骤，结尾揭晓是假手，观众爆笑。",
    "expected_ecr_band": "high",
    "title": "悬念钩子+动态剪辑+喜剧 payoff",
    "why": "Hook Strength=1, Visual Dynamism=1, Narrative Progression=1, Emotional Resonance=1"
},
{
    "description": "固定镜头拍猫吃鱼，无音乐无字幕，持续10秒无变化。",
    "expected_ecr_band": "mid",
    "title": "基础吸引力+无叙事",
    "why": "Hook Strength=0.5 (可爱动物), Visual Dynamism=0, Narrative Progression=0"
},
{
    "description": "开头'为什么女生总说「随便」？'+快速切换情侣对话场景，结尾男友崩溃大喊'那我选火锅！'，配夸张音效。",
    "expected_ecr_band": "high",
    "title": "社会话题钩子+节奏快+情绪共鸣",
    "why": "Hook Strength=1, Visual Dynamism=1, Emotional Resonance=1, Narrative Progression=1"
}
]
```
"""


CAVEATS = """

## 注意事项


- 本 rubric 融合自 36 组、共 ~288 条视频的归纳，适用于预测 ECR（观看时长超过5秒的用户占比）。
- 维度权重基于高频出现频次与各组原始权重加权平均后归一化，非绝对真理，需结合具体平台调优。
- 低频维度（如'Niche Specificity'）已降权或合并，避免过拟合特定样本偏差。
- 检测启发式规则（detection_heuristic）取多组中最清晰、可操作性最强的版本，确保人工标注一致性。
"""


OUTPUT_LIMITATION = "输出预估的观看5s以上的用户达成率及预估原因。按照如下格式输出：ECR: xxx ; reason: xxx"

# Used with --no-fewshot: without the anchors the model has no scale reference,
# so state the expected output range explicitly and forbid copying.
OUTPUT_LIMITATION_NO_FEWSHOT = (
    "先在心里给 5 个维度分别打 0~1 分，再套用上面的评分公式算出 ECR。\n"
    "ECR 必须是 0 到 1 之间的小数，请给出精确到两位小数的具体数值（例如 0.37、0.62），"
    "不要只输出 0.5 或 0.85 这类整齐的数。\n"
    "reason 必须描述你在这条视频里实际看到的画面内容，不要复述评分标准的措辞。\n"
    "按照如下格式输出：ECR: xxx ; reason: xxx"
)


VIDEO_INFO = """
## 视频信息

标题：{title}
描述：{description}
"""


def build_rubric_prompt(title: str, description: str, use_fewshot: bool = True) -> str:
    """Assemble the full rubric prompt for one video.

    Args:
        use_fewshot: include the few-shot calibration anchors. Set to False to
            run the ablation described in FEWSHOT_ANCHORS' docstring.
    """
    parts = [ROLE_PLAY, "\n", TASK_DESC]
    if use_fewshot:
        parts.append(FEWSHOT_ANCHORS)
    parts.append(CAVEATS)
    parts.append(VIDEO_INFO.format(title=title, description=description))
    parts.append("\n")
    parts.append(OUTPUT_LIMITATION if use_fewshot else OUTPUT_LIMITATION_NO_FEWSHOT)
    return "".join(parts)


# Structured copies used by the divide-and-conquer experiment. Keeping these
# separate from TASK_DESC lets each call see exactly one dimension, while the
# sixth call receives only the already-produced scores and the aggregation
# rules (unless include_dimensions=True is explicitly requested).
DIMENSIONS = [
    (
        "Hook_Strength",
        "Hook Strength & Immediacy",
        0.28,
        """### Hook Strength & Immediacy  (weight=0.28, direction=positive)
- **看什么:** 是否在开头1-3秒内用视觉冲击、文字提问、情绪表达或悬念设置抓住注意力？
- **怎么判:** 0分：无任何视觉/文本钩子；前3秒静止、模糊、无动作或无明确主题。
0.5分：有微弱钩子但不清晰（如模糊文字、缓慢移动主体），观众需等待>3秒才理解意图。
1分：前1-2秒即呈现强烈钩子——爆炸/跳跃/夸张表情/挑衅文字/悬念问题，立即引发“接下来会发生什么？”的好奇心。""",
    ),
    (
        "Narrative_Progression",
        "Narrative Progression & Payoff",
        0.24,
        """### Narrative Progression & Payoff  (weight=0.24, direction=positive)
- **看什么:** 是否有“开始-发展-结局”的完整故事弧？是否在结尾提供满足感或意外转折？
- **怎么判:** 0分：无叙事结构，纯随机片段或重复动作（如持续走路、无变化的风景）。
0.5分：有简单事件但无发展或结局（如展示产品但无使用/对比/结果）。
1分：清晰三段式结构——开头设问/冲突，中段发展/挑战，结尾反转/高潮/情感释放（如实验失败、搞笑打脸、惊喜礼物）。""",
    ),
    (
        "Visual_Dynamism",
        "Visual Dynamism & Pacing",
        0.20,
        """### Visual Dynamism & Pacing  (weight=0.20, direction=positive)
- **看什么:** 画面是否充满动感？剪辑是否紧凑？是否有镜头运动或主体行为维持观众眼球？
- **怎么判:** 0分：画面静态、无运动、无剪辑变化；镜头晃动严重或构图混乱。
0.5分：少量动态元素（如单人走动），但节奏缓慢或缺乏视觉引导。
1分：频繁切换镜头、快速剪辑、人物/物体活跃运动、相机跟随/变焦/摇移，营造能量感和视觉流动性。""",
    ),
    (
        "Emotional_Resonance",
        "Emotional Resonance & Relatability",
        0.18,
        """### Emotional Resonance & Relatability  (weight=0.18, direction=positive)
- **看什么:** 是否能让人笑、惊、叹、或产生“我也这样”的代入感？是否避开负面/小众题材？
- **怎么判:** 0分：内容平淡无奇（如日常琐事无加工）、令人不适（血腥/恶心）、或极度小众（仅特定圈层懂）。
0.5分：有一定趣味但缺乏共鸣点（如普通宠物视频无互动/表情）。
1分：触发普遍情绪——幽默（荒谬/反差）、惊奇（科学实验/特技）、共鸣（家庭/职场/恋爱痛点）、或感官刺激（色彩/音效/速度）。""",
    ),
    (
        "Production_Quality",
        "Production Quality & Clarity",
        0.10,
        """### Production Quality & Clarity  (weight=0.10, direction=positive)
- **看什么:** 技术层面是否专业？能否让观众轻松看清听清内容？
- **怎么判:** 0分：画面模糊、曝光错误、声音嘈杂/缺失、字幕错位或难以阅读。
0.5分：基本清晰但存在瑕疵（如轻微抖动、背景噪音、字体太小）。
1分：稳定拍摄、良好照明、清晰对焦、音画同步、字幕简洁易读且增强信息传递。""",
    ),
]

SCORING_FORMULA = """## 评分公式
```
ECR_score =
(Hook_Strength * 0.28) +
(Narrative_Progression * 0.24) +
(Visual_Dynamism * 0.20) +
(Emotional_Resonance * 0.18) +
(Production_Quality * 0.10)

# 惩罚项（若任一维度为0，则整体得分减0.15）
IF min(Hook_Strength, Narrative_Progression, Visual_Dynamism, Emotional_Resonance, Production_Quality) == 0:
ECR_score = max(0, ECR_score - 0.15)

# 非线性增强：若Hook_Strength >= 0.8 且 Narrative_Progression >= 0.8，则额外+0.05
IF Hook_Strength >= 0.8 AND Narrative_Progression >= 0.8:
ECR_score = min(1.0, ECR_score + 0.05)
```"""


def build_dimension_prompt(
    title: str,
    description: str,
    dimension_index: int,
    use_fewshot: bool = True,
) -> str:
    """Build one of the five independent dimension-scoring prompts."""
    key, name, _, instruction = DIMENSIONS[dimension_index]
    parts = [
        ROLE_PLAY,
        "\n\n本次只评估下面一个维度，不要计算最终 ECR。只能按规则输出 0、0.5 或 1。\n\n",
        instruction,
    ]
    if use_fewshot:
        parts.append(FEWSHOT_ANCHORS)
    parts.extend([
        CAVEATS,
        VIDEO_INFO.format(title=title, description=description),
        "\n请根据视频内容、标题和描述评估该维度。不要输出其他维度或最终 ECR。\n",
        f"按照如下格式输出：dimension: {key} ; score: 0/0.5/1 ; reason: 对实际视频内容的简短说明",
    ])
    return "".join(parts)


def build_aggregate_prompt(
    title: str,
    description: str,
    dimension_results,
    use_fewshot: bool = True,
    include_dimensions: bool = False,
) -> str:
    """Build the sixth call that aggregates five prior dimension results."""
    score_lines = []
    for key, name, _, _ in DIMENSIONS:
        result = dimension_results[key]
        score_lines.append(
            f"- {key} ({name}): score={result['score']}; reason={result['reason']}"
        )

    parts = [ROLE_PLAY, "\n\n"]
    if include_dimensions:
        parts.extend(["## 五个维度判断说明\n\n", "\n\n".join(d[3] for d in DIMENSIONS), "\n\n"])
    parts.extend([
        "## 前置五次调用的评分结果\n",
        "\n".join(score_lines),
        "\n\n",
        SCORING_FORMULA,
    ])
    if use_fewshot:
        parts.append(FEWSHOT_ANCHORS)
    parts.extend([
        CAVEATS,
        VIDEO_INFO.format(title=title, description=description),
        "\n必须使用给定的五个 score 按公式计算，不要重新修改维度分数。",
        "\n输出预估的观看5s以上的用户达成率及预估原因。按照如下格式输出：ECR: xxx ; reason: xxx",
    ])
    return "".join(parts)

