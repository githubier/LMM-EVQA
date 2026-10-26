"""Twelve-dimension ECR rubric and divide-and-conquer prompt builders.

The model-facing ECR is in [0, 1]. The inference script converts the parsed
answer to [0, 100], matching the SnapUGC ground-truth scale.
"""

ROLE_PLAY = "你是一个短视频完播率预测专家。你正在预估一条短视频的 ECR（观看超过5秒的用户占比，[0,1]）"

DIMENSIONS = [
    (
        "First_Frame_Hook_Strength",
        "First-Frame Hook Strength",
        0.1176,
        """### First-Frame Hook Strength  (weight=0.1176, direction=positive)
- **看什么:** 首帧是否包含视觉或听觉冲击元素，如夸张表情、突然动作、强对比色块或突发音效。
- **怎么判:** 0: 首帧无动态/表情/声音触发；0.5: 首帧有轻微动作或模糊表情，但无明确情绪或悬念；1: 首帧含强烈面部表情（惊恐/大笑）、突发运动（跳跃/摔倒）、高对比色彩或尖锐音效（尖叫/爆炸），在0.5秒内激活注意力系统。""",
    ),
    (
        "Visual_Pacing_Camera_Dynamism",
        "Visual Pacing & Camera Dynamism (0-3s)",
        0.1078,
        """### Visual Pacing & Camera Dynamism (0-3s)  (weight=0.1078, direction=positive)
- **看什么:** 前3秒内镜头切换频率、相机运动强度及与音频/动作的同步程度。
- **怎么判:** 0: 前3秒无明显剪辑节奏或镜头运动；0.5: 有1~2次剪辑或轻微运镜，但节奏平缓；1: 前3秒内≥3次剪辑或显著镜头运动（推拉摇移），且与音频节拍或动作同步，制造紧迫感或能量感。""",
    ),
    (
        "Facial_Expression_Intensity_Synchronization",
        "Facial Expression Intensity & Synchronization (0-2s)",
        0.098,
        """### Facial Expression Intensity & Synchronization (0-2s)  (weight=0.098, direction=positive)
- **看什么:** 人物面部是否在早期展现高唤醒度表情，并与视觉/音频事件精确对齐。
- **怎么判:** 0: 人物面部不可见或表情平淡；0.5: 有人物面部，但表情轻微或延迟出现（>2s）；1: 前2秒内出现清晰、高强度面部表情（惊讶/喜悦/愤怒），且与关键事件同步，触发镜像神经元反应。""",
    ),
    (
        "Text_Overlay_Timing_Emotional_Charge",
        "Text Overlay Timing & Emotional Charge (≤1s)",
        0.0882,
        """### Text Overlay Timing & Emotional Charge (≤1s)  (weight=0.0882, direction=positive)
- **看什么:** 字幕是否在第一秒内出现，语义是否制造认知缺口或情绪张力，而非单纯说明。
- **怎么判:** 0: 无文本叠加或文本出现晚于3s且无关；0.5: 文本在1~3s出现，内容模糊或仅装饰；1: 文本在≤1s出现，内容具悬念、幽默或冲突性（如提问、断言、反转），并强化视觉钩子。""",
    ),
    (
        "Narrative_Arc_Completion",
        "Narrative Arc Completion (≤5s)",
        0.098,
        """### Narrative Arc Completion (≤5s)  (weight=0.098, direction=positive)
- **看什么:** 视频是否在5秒内构建起“起承转合”的微型叙事闭环，即使简单也需有明确收尾。
- **怎么判:** 0: 无明确叙事结构或问题提出；0.5: 有模糊前提但无发展或结局；1: 前5秒内完整呈现微型故事弧：设置（问题/情境）→ 发展（行动/冲突）→ 解决（笑点/反转/情感释放）。""",
    ),
    (
        "Audio_Visual_Sync_Impact",
        "Audio-Visual Sync & Impact (0-2s)",
        0.0882,
        """### Audio-Visual Sync & Impact (0-2s)  (weight=0.0882, direction=positive)
- **看什么:** 关键视觉时刻是否匹配高冲击力音效，实现视听协同放大效果。
- **怎么判:** 0: 音频无声、单调或与画面脱节；0.5: 有基础音效或背景音乐，但无情绪强化；1: 前2秒内出现精准同步的音效（如撞击声、笑声、音乐重音），增强视觉冲击或情绪共鸣。""",
    ),
    (
        "Color_Contrast_Saturation",
        "Color Contrast & Saturation (0-3s)",
        0.0784,
        """### Color Contrast & Saturation (0-3s)  (weight=0.0784, direction=positive)
- **看什么:** 画面是否通过色彩强度和对比度引导视线，制造视觉锚点，提升信息流中的识别度。
- **怎么判:** 0: 画面灰暗、色彩单一或主体不突出；0.5: 有基本色彩对比，但饱和度不足或构图杂乱；1: 使用高饱和度、高对比度色彩，主体与背景分离清晰，视觉焦点明确，增强记忆点。""",
    ),
    (
        "Interpersonal_Interaction_Density",
        "Interpersonal Interaction Density (0-3s)",
        0.0686,
        """### Interpersonal Interaction Density (0-3s)  (weight=0.0686, direction=positive)
- **看什么:** 是否存在多人互动或反应镜头，激活观众的社会认知机制。
- **怎么判:** 0: 无角色互动或社交信号；0.5: 有两人同框但无交流或反应；1: 前3秒内出现角色间物理接触、眼神交流、对话或反应镜头，制造社会张力或幽默感。""",
    ),
    (
        "Physical_Motion_Intensity_Consequence",
        "Physical Motion Intensity & Consequence (0-3s)",
        0.0784,
        """### Physical Motion Intensity & Consequence (0-3s)  (weight=0.0784, direction=positive)
- **看什么:** 是否有高速、高幅度的动作，并产生即时、可视化的结果，触发注意力系统。
- **怎么判:** 0: 动作稀少或缓慢；0.5: 有少量动作但无后果或节奏；1: 前3秒内出现高能量物理动作（跳跃、碰撞、快速移动），并伴随可见后果（如物体飞溅、表情变化），制造动能吸引力。""",
    ),
    (
        "Topic_Relatability_Concept_Accessibility",
        "Topic Relatability & Concept Accessibility (0-2s)",
        0.0686,
        """### Topic Relatability & Concept Accessibility (0-2s)  (weight=0.0686, direction=positive)
- **看什么:** 主题是否贴近大众生活经验，能在2秒内被观众直觉理解并产生代入感。
- **怎么判:** 0: 视频依赖文字解释或缺乏核心主题；0.5: 主题模糊或需额外知识理解；1: 内容聚焦单一、具体主题（如科技故障、家庭趣事、体育失误），无需背景知识即可理解，具有普适共鸣点。""",
    ),
    (
        "Production_Quality_Technical_Clarity",
        "Production Quality & Technical Clarity",
        0.0588,
        """### Production Quality & Technical Clarity  (weight=0.0588, direction=positive)
- **看什么:** 视频技术执行层面是否干净利落，避免因画质问题导致用户提前划走。
- **怎么判:** 0: 画面抖动、失焦、过曝或压缩伪影严重；0.5: 技术质量一般，有轻微瑕疵但不影响观看；1: 画面稳定、对焦准确、曝光合理、无明显压缩痕迹，体现专业制作水准。""",
    ),
    (
        "Direct_Viewer_Engagement_CTA",
        "Direct Viewer Engagement & Call-to-Action Design",
        0.049,
        """### Direct Viewer Engagement & Call-to-Action Design  (weight=0.049, direction=positive)
- **看什么:** 是否通过语言、镜头或行为设计将观众置于情境中心，而非旁观者。
- **怎么判:** 0: 无直接观众互动或引导；0.5: 有间接提示（如“你猜怎么着”）但无行为指令；1: 明确邀请观众参与（如“暂停猜结局”、“评论区告诉我”），或使用第二人称视角制造代入感。""",
    ),
]

SCORING_FORMULA = """## 评分公式
```
Final Score = Σ(dimension_score_i × weight_i)
其中每个维度按 detection_heuristic 分档打分（0, 0.5, 1），权重归一化至总和=1.0。
若存在以下情况，可酌情扣减0.05~0.15分：
- 文本过度依赖（视频核心信息必须靠读字幕才能理解）
- 音画严重不同步（如动作发生后1秒才响音效）
- 结尾无明确收束（如突然黑屏或未解决悬念）
- 画面技术缺陷（严重抖动、失焦、过曝）
```"""

FEWSHOT_ANCHORS = """
## Few-shot 校准锚点
```json
[
  {
    "title": "如何安装Python？",
    "description": "低分案例：静态讲解视频，首帧为纯文字标题，无表情、无动作、无音效，前3秒仅一人说话，无剪辑变化，ECR≈0.25",
    "expected_ecr_band": "low",
    "why": "缺乏所有核心钩子：无视觉/听觉冲击、无表情、无剪辑、无叙事、无互动。"
  },
  {
    "title": "现代舞欣赏",
    "description": "低分案例：慢动作舞蹈片段，首帧为远景，人物表情模糊，无字幕，无音效同步，前5秒无情节推进，ECR≈0.30",
    "expected_ecr_band": "low",
    "why": "虽有美感但缺乏紧迫感、情绪强度和叙事驱动，无法在前3秒抓住注意力。"
  },
  {
    "title": "朋友偷吃我的零食后...",
    "description": "中分案例：搞笑短剧开头，首帧有夸张表情+文字‘千万别眨眼！’，前3秒有2次剪辑，有音效但不同步，结尾无明确笑点，ECR≈0.55",
    "expected_ecr_band": "mid",
    "why": "具备部分钩子（表情、文字、剪辑），但叙事不完整、音效不同步，削弱整体吸引力。"
  },
  {
    "title": "实测XX神器，结果出乎意料",
    "description": "中分案例：产品测评视频，首帧为产品特写+文字‘这玩意儿真能用？’，前3秒有3次剪辑，有音效但无情绪表达，无角色互动，ECR≈0.58",
    "expected_ecr_band": "mid",
    "why": "有悬念文本和剪辑节奏，但缺乏情感表达和人际互动，难以维持深度投入。"
  },
  {
    "title": "当男友发现我藏了他的游戏手柄...",
    "description": "高分案例：短视频喜剧，首帧为人物惊恐表情+爆炸音效，前3秒有5次剪辑+同步音效，有对话冲突+字幕‘他居然敢…’，5秒内完成反转笑点，ECR≈0.85",
    "expected_ecr_band": "high",
    "why": "完美融合所有核心维度：强首帧钩子、高剪辑密度、情绪同步、叙事闭环、互动与悬念。"
  },
  {
    "title": "极限挑战：从10米跳水不喊疼",
    "description": "高分案例：挑战类视频，首帧为人物跳下高台+尖叫声，前3秒有4次剪辑+慢动作+音效同步，有字幕‘你能坚持几秒？’，5秒内展示失败瞬间+观众反应，ECR≈0.88",
    "expected_ecr_band": "high",
    "why": "高动能、高情绪、高互动、高叙事完成度，多重信号叠加强化吸引力。"
  }
]
```"""

CAVEATS = """
## 注意事项
- 本 rubric 融合自 58 组、共 ~464 条视频的归纳，适用于预测 ECR（观看时长超过5秒的用户占比）。
- 维度设计基于多组独立归纳的高频信号，非单一视频样本的主观打分。
- 评分公式为线性加权合成，未包含复杂非线性交互项，实际应用中可结合模型校准。
- 各维度检测需在指定时间窗口内完成，超出窗口的信号不计入评分。
- 低频维度（出现<5组）已降权或合并，部分边缘信号可能被过滤，不代表其无价值。
"""

VIDEO_INFO = """
## 视频信息
标题：{title}
描述：{description}
"""

OUTPUT_LIMITATION = "输出预估的观看5s以上的用户达成率及预估原因。按照如下格式输出：ECR: xxx ; reason: xxx"
OUTPUT_LIMITATION_NO_FEWSHOT = (
    "先在心里严格按 12 个维度分别打 0、0.5 或 1 分，再套用评分公式，并检查是否需要扣分。\n"
    "ECR 必须是 0 到 1 之间的小数，请给出精确到两位小数的数值。\n"
    "reason 必须描述这条视频的实际内容，不要复述 rubric。\n"
    "按照如下格式输出：ECR: xxx ; reason: xxx"
)


def _dimension_text():
    return "\n\n".join(item[3] for item in DIMENSIONS)


def build_rubric_prompt(title: str, description: str, use_fewshot: bool = True) -> str:
    """Build one direct 12-dimension -> ECR call."""
    parts = [
        ROLE_PLAY,
        "\n\n对每个维度打 0~1 分，然后按下方公式合成，最后输出 ECR。\n\n## 维度\n\n",
        _dimension_text(),
        "\n\n",
        SCORING_FORMULA,
    ]
    if use_fewshot:
        parts.append(FEWSHOT_ANCHORS)
    parts.extend([
        CAVEATS,
        VIDEO_INFO.format(title=title, description=description),
        OUTPUT_LIMITATION if use_fewshot else OUTPUT_LIMITATION_NO_FEWSHOT,
    ])
    return "".join(parts)


def build_dimension_prompt(
    title: str,
    description: str,
    dimension_index: int,
    use_fewshot: bool = True,
) -> str:
    """Build one of twelve independent dimension-scoring calls."""
    key, _, _, instruction = DIMENSIONS[dimension_index]
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
        "\n请严格遵守该维度的时间窗口。不要输出其他维度或最终 ECR。\n",
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
    """Build the thirteenth call that aggregates twelve prior scores."""
    score_lines = []
    for key, name, _, _ in DIMENSIONS:
        result = dimension_results[key]
        score_lines.append(
            f"- {key} ({name}): score={result['score']}; reason={result['reason']}"
        )

    parts = [ROLE_PLAY, "\n\n"]
    if include_dimensions:
        parts.extend(["## 十二个维度判断说明\n\n", _dimension_text(), "\n\n"])
    parts.extend([
        "## 前置十二次调用的评分结果\n",
        "\n".join(score_lines),
        "\n\n",
        SCORING_FORMULA,
    ])
    if use_fewshot:
        parts.append(FEWSHOT_ANCHORS)
    parts.extend([
        CAVEATS,
        VIDEO_INFO.format(title=title, description=description),
        "\n必须使用给定的十二个 score 按公式计算，不要重新修改维度分数。",
        "\n输出预估的观看5s以上的用户达成率及预估原因。按照如下格式输出：ECR: xxx ; reason: xxx",
    ])
    return "".join(parts)
