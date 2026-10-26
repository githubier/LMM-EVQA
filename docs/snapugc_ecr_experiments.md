# SnapUGC ECR 实验方案与阶段性结果

> 本文用于恢复机器故障前已经完成、但尚未系统沉淀到仓库中的实验背景。当前只记录方案、结果和结论，不补写已经丢失的实验代码，也不重新运行实验。
>
> 信息来源包括：工作区根目录的 `SnapUGC数据集DBDE结果.csv`（重点为第 29–86 行）、`后训练ECR与12D分治融合实验说明.md`（对应 CSV 第 76–86 行的详细记录），以及仓库中 `VideoLLaMA2-audio_visual/zero_shot_ecr/` 保留的实现、运行脚本、逐样本结果和汇总表。除非特别说明，指标均在 500 条 SnapUGC 测试样本上计算。

## 1. 项目现状与研究路线

原仓库以 VideoLLaMA2 / Qwen2.5-VL 微调后直接回归 ECR 为主。当前 VideoLLaMA2 后训练回归模型是强基线，Final Score 约为 `0.693`。后续探索的核心问题是：

1. 原始 VideoLLaMA2 能否通过 prompt 零样本直接判断 ECR；
2. 将整体 ECR 判断拆成 5D 或 12D 后，分维度推理能否改善结果；
3. 12D 分治信息能否在强后训练回归模型之上提供可归因的增量。

已有代码和结果主要保存在 `VideoLLaMA2-audio_visual/zero_shot_ecr/`。其中：

- v1.5.1 prompt 指 5 维拆分逻辑；
- v1.6 prompt 指 12 维拆分逻辑；
- `dimension soft` 指维度分数不再只使用硬离散判断，而使用连续值或等级概率期望；
- `few-shot` 消融用于检查锚点复制、输出坍缩和模型真实判断能力；
- “分治”指先独立得到维度判断，再由 LLM 或 Python 聚合；
- 本文后半部分的融合实验指将 12D 分治结果与后训练 ECR 回归结果结合。

评价指标统一为：

```text
Final Score = 0.6 × SRCC + 0.4 × PLCC
```

## 2. 5D 分治实验（v1.5.1 prompt）

### 2.1 基本分组

5 个维度以编号 `1...5` 表示。尝试了以下拆分：

- 2 组：`[1+2+4, 3+5]`；
- 3 组：`[1+2, 3+5, 4]`；
- 按模态拆分的 2 组：`[1+2+4, 3+5]`，其中 `3+5` 只看视频画面，`1+2+4` 使用视频画面和音频；
- 按模态拆分的 3 组：`[1+4, 3+5, 2]`，其中 `3+5` 只看视频画面，`1+4` 和 `2` 使用视频画面和音频。

所有下表结果均用 Python 合并分组结果。

### 2.2 结果

| 分组 | prompt 变体 | SRCC | PLCC | Final |
|---|---|---:|---:|---:|
| `[1+2+4, 3+5]` | 原始 | -0.0423 | 0.0374 | -0.0104 |
| `[1+2+4, 3+5]` | dimension soft | -0.1168 | 0.1731 | -0.0008 |
| `[1+2+4, 3+5]` | 去 few-shot | 0.0986 | 0.1618 | **0.1239** |
| `[1+2+4, 3+5]` | 去 few-shot + dimension soft | 0.0655 | 0.0311 | 0.0517 |
| `[1+2+4, 3+5]` + 模态区分 | 原始 | -0.0616 | 0.1054 | 0.0052 |
| `[1+2+4, 3+5]` + 模态区分 | dimension soft | -0.0706 | 0.1006 | -0.0021 |
| `[1+2+4, 3+5]` + 模态区分 | 去 few-shot | -0.0095 | 0.1475 | 0.0533 |
| `[1+2+4, 3+5]` + 模态区分 | 去 few-shot + dimension soft | 0.0577 | 0.0697 | 0.0625 |
| `[1+2, 3+5, 4]` | 原始 | -0.0904 | 0.1027 | -0.0132 |
| `[1+2, 3+5, 4]` | dimension soft | -0.1079 | 0.1071 | -0.0219 |
| `[1+2, 3+5, 4]` | 去 few-shot | 0.0104 | 0.0201 | 0.0142 |
| `[1+2, 3+5, 4]` | 去 few-shot + dimension soft | -0.0302 | 0.0280 | -0.0069 |
| `[1+4, 3+5, 2]` + 模态区分 | 原始 | -0.0426 | 0.0344 | -0.0118 |
| `[1+4, 3+5, 2]` + 模态区分 | dimension soft | -0.0920 | 0.0770 | -0.0244 |
| `[1+4, 3+5, 2]` + 模态区分 | 去 few-shot | 0.0002 | 0.0105 | 0.0043 |
| `[1+4, 3+5, 2]` + 模态区分 | 去 few-shot + dimension soft | -0.0070 | 0.0885 | 0.0312 |

### 2.3 阶段结论

- 5D 最好结果来自最简单的两组拆分 `[1+2+4, 3+5]` 且去掉 few-shot，Final 为 `0.1239`。
- 模态区分没有稳定带来收益；最好模态区分结果为 `0.0625`，明显低于不区分模态的最好结果。
- 进一步拆为 3 组也没有改善，说明增加调用次数或更细分组并不自动产生更好的 ECR 排序能力。
- dimension soft 在 5D 分组中没有稳定增益；它有时提高 PLCC，但会损害 SRCC。
- 这些零样本分治结果仍远低于约 `0.693` 的后训练回归基线，因此后续重点转向 12D 结构化信号和强基线融合，而不是单独用零样本模型替代后训练模型。

## 3. 12D 分治实验（v1.6 prompt）

### 3.1 十二个维度

v1.6 将 ECR 判断拆为：

1. `First_Frame_Hook_Strength`
2. `Visual_Pacing_Camera_Dynamism`
3. `Facial_Expression_Intensity_Synchronization`
4. `Text_Overlay_Timing_Emotional_Charge`
5. `Audio_Visual_Sync_Impact`
6. `Color_Contrast_Saturation`
7. `Production_Quality_Technical_Clarity`
8. `Physical_Motion_Intensity_Consequence`
9. `Narrative_Arc_Completion`
10. `Topic_Relatability_Concept_Accessibility`
11. `Direct_Viewer_Engagement_CTA`
12. `Interpersonal_Interaction_Density`

### 3.2 分组方案

**普通 2 组**

- 组 1：前八个偏感知和制作层面的维度；
- 组 2：`Narrative_Arc_Completion`、`Topic_Relatability_Concept_Accessibility`、`Direct_Viewer_Engagement_CTA`、`Interpersonal_Interaction_Density`。

**按模态 2 组**

- 视频组：`Visual_Pacing_Camera_Dynamism`、`Color_Contrast_Saturation`、`Production_Quality_Technical_Clarity`、`Physical_Motion_Intensity_Consequence`、`Facial_Expression_Intensity_Synchronization`、`Text_Overlay_Timing_Emotional_Charge`；
- 音频 + 视频组：`First_Frame_Hook_Strength`、`Audio_Visual_Sync_Impact`、`Narrative_Arc_Completion`、`Topic_Relatability_Concept_Accessibility`、`Direct_Viewer_Engagement_CTA`、`Interpersonal_Interaction_Density`。

**普通 3 组**

- 组 1：`Visual_Pacing_Camera_Dynamism`、`Color_Contrast_Saturation`、`Production_Quality_Technical_Clarity`、`Physical_Motion_Intensity_Consequence`；
- 组 2：`First_Frame_Hook_Strength`、`Facial_Expression_Intensity_Synchronization`、`Text_Overlay_Timing_Emotional_Charge`、`Audio_Visual_Sync_Impact`；
- 组 3：`Narrative_Arc_Completion`、`Topic_Relatability_Concept_Accessibility`、`Direct_Viewer_Engagement_CTA`、`Interpersonal_Interaction_Density`。

**按模态 3 组**

- 视频组：`Visual_Pacing_Camera_Dynamism`、`Color_Contrast_Saturation`、`Production_Quality_Technical_Clarity`、`Physical_Motion_Intensity_Consequence`、`Facial_Expression_Intensity_Synchronization`；
- 音频 + 视频组 1：`First_Frame_Hook_Strength`、`Text_Overlay_Timing_Emotional_Charge`、`Audio_Visual_Sync_Impact`；
- 音频 + 视频组 2：`Narrative_Arc_Completion`、`Topic_Relatability_Concept_Accessibility`、`Direct_Viewer_Engagement_CTA`、`Interpersonal_Interaction_Density`。

### 3.3 结果

| 分组 | prompt 变体 | SRCC | PLCC | Final |
|---|---|---:|---:|---:|
| 普通 2 组 | 原始 | 0.0377 | 0.1105 | 0.0668 |
| 普通 2 组 | dimension soft | 0.1026 | 0.1424 | **0.1185** |
| 普通 2 组 | 去 few-shot | 0.0894 | 0.1126 | 0.0987 |
| 普通 2 组 | 去 few-shot + dimension soft | 0.0263 | 0.1892 | 0.0915 |
| 按模态 2 组 | 原始 | 0.0154 | 0.0307 | 0.0216 |
| 按模态 2 组 | dimension soft | 0.0057 | 0.0759 | 0.0338 |
| 按模态 2 组 | 去 few-shot | -0.1139 | 0.1194 | -0.0206 |
| 按模态 2 组 | 去 few-shot + dimension soft | -0.0397 | 0.1822 | 0.0491 |
| 普通 3 组 | 原始 | 0.2069 | 0.2057 | **0.2064** |
| 普通 3 组 | dimension soft | 0.0333 | 0.0181 | 0.0272 |
| 普通 3 组 | 去 few-shot | -0.0031 | 0.1532 | 0.0594 |
| 普通 3 组 | 去 few-shot + dimension soft | 0.0971 | 0.1484 | 0.1176 |
| 按模态 3 组 | 原始 | 0.1952 | 0.2674 | **0.2241** |
| 按模态 3 组 | dimension soft | 0.2044 | 0.1582 | 0.1859 |
| 按模态 3 组 | 去 few-shot | 0.0784 | 0.1236 | 0.0964 |
| 按模态 3 组 | 去 few-shot + dimension soft | 0.0075 | 0.0887 | 0.0400 |

在按模态 3 组的最好分治结果基础上，还尝试让 LLM 聚合：

| LLM 聚合输入 | SRCC | PLCC | Final |
|---|---:|---:|---:|
| score + evidence + formula | -0.0893 | 0.1056 | -0.0113 |
| 完整维度定义 + score + evidence + formula | -0.0461 | 0.0261 | -0.0172 |

### 3.4 阶段结论

- 12D 最好零样本分治结果是“按模态 3 组 + 原始 v1.6 prompt”，Final 为 `0.2241`，优于 5D 最好结果 `0.1239`，说明更细的维度结构具有诊断价值。
- 普通 3 组原始 prompt 也达到 `0.2064`，表明收益不完全依赖模态区分。
- soft 与去 few-shot 的效果依赖分组，不能作为稳定改进：最好结果反而来自原始离散评分和保留 few-shot。
- 将已有分组结果再次交给 7B LLM 聚合会严重破坏排序；Python 聚合比让模型自由做连续数值计算更可靠。

## 4. 在后训练 prompt 中注入 12D 信息

在强后训练链路中加入分治结果或维度名称，得到：

| 输入方式 | SRCC | PLCC | Final | 相对 Baseline |
|---|---:|---:|---:|---:|
| Baseline | 0.687303 | 0.701452 | **0.692963** | — |
| Correct 12D | 0.686821 | 0.701131 | 0.692545 | -0.000418 |
| Constant 12D | 0.686792 | 0.701107 | 0.692518 | -0.000445 |
| Shuffled 12D | 0.686654 | 0.701036 | 0.692407 | -0.000556 |
| 仅 12D Names | 0.686663 | 0.700915 | 0.692364 | -0.000599 |

结论：直接把 12D 信息塞进后训练 prompt 没有超过基线。Correct 虽略高于 Constant / Shuffled，但整体仍下降，说明融合位置和机制比“是否提供 12D”更关键。

> 注：源 CSV 这里记录的基线 Final 为 `0.692963`，详细融合说明按更高精度 PLCC `0.701470` 计算为 `0.692970`。两者只存在 `7e-6` 的记录精度差异，后续融合章节统一采用详细说明中的 `0.692970`。

## 5. 后训练 ECR 与 12D 分治融合

### 5.1 目标、符号与严格控制组

该阶段不只追求 Final 提升，而是验证“和当前视频正确对应的 12D 信息”是否提供真实增量。

- `F`：后训练回归模型输出的 ECR；
- `D`：12D grouped 分治结果经 Python weighted 聚合得到的 ECR；
- `d1...d12`：12 个维度分数；
- `D_z`：对齐到 `F` 均值和标准差后的 12D ECR；
- `Y`：真实 ECR。

控制组：

- **Correct**：`F_i` 与视频 `i` 自己的 `D_i / 12D_i` 融合；
- **Shuffled**：`F_i` 与其他视频 `j` 的 `D_j / 12D_j` 融合，且 `j != i`；
- **Constant**：`F_i` 与固定 `D=50`、12 个维度均为 `0.5` 的结果融合。

只有同时满足 `Correct > Baseline`、`Correct > Shuffled`、`Correct > Constant`，才是支持正确分治信息有效的理想证据。Shuffled 也提升时，提升可能来自集成、尺度校准或预测收缩，而非视频对应的 12D 内容。

本节统一基线：SRCC `0.687303`、PLCC `0.701470`、Final `0.692970`。

### 5.2 LLM 直接输出最终 ECR

输入视频、`F`、`D` 和 `d1...d12`，由原始 VideoLLaMA2 自由生成最终连续 ECR。主要结果：

| 模式 | SRCC | PLCC | Final | 相对 Baseline |
|---|---:|---:|---:|---:|
| 只输入 `F` | 0.658799 | 0.677332 | 0.666212 | -0.026758 |
| `F + Shuffled 12D` | 0.658167 | 0.670927 | 0.663271 | -0.029699 |
| `F + Correct 12D` | 0.638720 | 0.651704 | 0.643914 | -0.049056 |
| `F + Constant 12D` | 0.618056 | 0.628848 | 0.622373 | -0.070597 |
| `F + Correct D` | 0.579280 | 0.588731 | 0.583060 | -0.109910 |

即使只让 LLM 复述/处理 `F` 也会损害强基线；Correct 还低于 Shuffled。7B 模型不适合承担最终连续数值聚合。

### 5.3 固定公式融合

**Raw Linear**

```text
Final = αF + (1-α)D
```

测试 `α = 0.95 / 0.90 / 0.85 / 0.80 / 0.75 / 0.70`，最好参数 `α=0.70`：Correct `0.696626`，Shuffled `0.695616`，Constant `0.692767`，Correct-Baseline `+0.003656`，Correct-Shuffled `+0.001010`。

**Z-score Aligned Linear**

```text
D_z = mean(F) + std(F) × (D-mean(D)) / std(D)
Final = αF + (1-α)D_z
```

最好参数 `α=0.90`：Correct `0.696662`，Shuffled `0.695534`，Constant `0.692767`，Correct-Baseline `+0.003692`，Correct-Shuffled `+0.001129`。

**Robust Aligned Linear**

```text
D_robust = median(F) + IQR(F) × (D-median(D)) / IQR(D)
Final = αF + (1-α)D_robust
```

最好参数 `α=0.90`：Correct `0.696721`，Shuffled `0.695742`，Constant `0.692767`，Correct-Baseline `+0.003751`，Correct-Shuffled `+0.000979`。

三类线性融合都有约 `+0.0037` 的表面收益，但正确视频对应关系的净贡献仅约 `+0.001`。

### 5.4 Bounded Residual

12D 只做有限修正：

```text
gap = D_z - F
correction = clip(β × gap, -cap, +cap)
Final = F + correction
```

测试 `β = 0.1 / 0.2 / 0.3`、`cap = 2 / 5`。最好参数 `β=0.1, cap=5`：Correct `0.696648`，Shuffled `0.695730`，Constant `0.692767`，Correct-Baseline `+0.003678`，Correct-Shuffled `+0.000918`。方案安全，但正确 12D 的净贡献偏小。

### 5.5 Confidence-aware（当前最强证据）

原离散维度分数为 `0 / 0.5 / 1`，用远离 `0.5` 的程度表示分治确定性：

```text
confidence = mean(abs(d_i - 0.5)) / 0.5
w = λ × confidence
Final = (1-w)F + wD_z
```

| λ | Correct | Shuffled | Constant | Correct-Baseline | Correct-Shuffled |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.695965 | 0.694060 | 0.692970 | +0.002995 | +0.001905 |
| 0.2 | 0.697276 | 0.694354 | 0.692970 | +0.004306 | +0.002922 |
| 0.3 | **0.697487** | 0.693012 | 0.692970 | **+0.004517** | **+0.004475** |

`λ=0.3` 时：SRCC 从 `0.687303` 提升到 `0.693342`（`+0.006039`），PLCC 从 `0.701470` 提升到 `0.703705`（`+0.002235`）。三个相邻参数均满足 Correct 同时高于 Baseline / Shuffled / Constant，是当前最符合研究目标的结果。

但 Correct 相对 Shuffled 的配对 bootstrap 95% 区间仍跨零，因此这是较强正向证据，不是严格统计证明。

### 5.6 Agreement、Disagreement 与 Rank Tie-break

**Agreement gating**：后训练和 12D 越接近，权重越大。

```text
gap = abs(D_z-F)
w = λ × exp(-gap/τ)
Final = (1-w)F + wD_z
```

最好参数 `λ=0.1, τ=20`：Correct `0.693038`，Shuffled `0.693649`，Constant `0.692728`，Correct-Shuffled `-0.000612`。不支持正确分治有效。

**Disagreement gating**：分歧越大，12D 权重越大。

```text
w = λ × min(gap/τ, 1)
Final = (1-w)F + wD_z
```

最好参数 `λ=0.1, τ=20`：Correct `0.697054`，Shuffled `0.694883`，Constant `0.692843`，Correct-Baseline `+0.004083`，Correct-Shuffled `+0.002171`。测试的 9 组参数全部满足 Correct > Shuffled，说明 12D 的价值可能主要出现在它与后训练模型冲突的样本上。

**Rank tie-break**：

```text
Final = F + ε × centered_rank(D)
```

最好参数 `ε=1`：Correct `0.693882`，Shuffled `0.694089`，Constant `0.692970`，Correct-Shuffled `-0.000207`。不支持正确分治有效。

### 5.7 固定多公式集成

固定平均 Raw Linear、Z-score Linear、Bounded Residual、Confidence-aware、Rank Tie-break 五个成员：Correct `0.695938`，Shuffled `0.696068`，Constant `0.692615`，Correct-Shuffled `-0.000129`。虽高于 baseline，但只能归因于集成收益，不能归因于正确 12D。

### 5.8 LLM 选择融合权重

让 LLM 在 `NONE / LOW / MEDIUM / HIGH` 中选择，再映射为 `0 / 0.1 / 0.2 / 0.3`，由 Python 计算：

```text
Final = (1-w)F + wD
```

结果为 Correct `0.696163`、Shuffled `0.696273`、Constant `0.692733`。LLM 在 Correct 中 499/500 输出 `w=0.2`，在 Shuffled 中 500/500 输出 `w=0.2`，基本退化为固定权重，且 Shuffled 高于 Correct。

### 5.9 融合方案总表

| 方案 | 最佳 Correct | 同参数 Shuffled | Correct-Shuffled | 判断 |
|---|---:|---:|---:|---|
| LLM 直接融合 | 0.643914 | 0.663271 | -0.019357 | 失败 |
| Raw Linear | 0.696626 | 0.695616 | +0.001010 | 少量正确分治贡献 |
| Z Linear | 0.696662 | 0.695534 | +0.001129 | 少量贡献 |
| Robust Linear | 0.696721 | 0.695742 | +0.000979 | 少量贡献 |
| Bounded Residual | 0.696648 | 0.695730 | +0.000918 | 安全但净贡献小 |
| Confidence-aware | **0.697487** | 0.693012 | **+0.004475** | 当前最强证据 |
| Agreement | 0.693038 | 0.693649 | -0.000612 | 不支持 |
| Disagreement | 0.697054 | 0.694883 | +0.002171 | 次优且方向一致 |
| Rank tie-break | 0.693882 | 0.694089 | -0.000207 | 不支持 |
| 固定公式集成 | 0.695938 | 0.696068 | -0.000129 | 只有集成收益 |
| LLM 选权重 | 0.696163 | 0.696273 | -0.000110 | 退化成固定权重 |

总体 50 对同参数公式实验中：Correct > Shuffled 为 `29/50`，Correct > Constant 为 `28/50`，Correct 同时超过两者为 `17/50`。

## 6. `VideoLLaMA2-audio_visual` 中保留的补充实验

本章来自仓库内现存代码、运行脚本和结果文件，补充前述两个外部文档没有完整展开的实验代际。这里区分三种状态：

- **有完整汇总**：存在对应 `comparison_*.csv`，可以核对样本数、失败数、唯一预测数和指标；
- **有逐样本结果但汇总缺失**：可以计算局部结果，但不能把脚本中规划的整个矩阵视为已完成；
- **只有脚本设计**：只记录原计划，不记为已验证结论。

### 6.1 原始零样本基线：三种数值产生方式

`zero_shot_ecr/` 与原微调回归链路隔离：它使用原始开源 `VideoLLaMA2.1-7B-AV`，临时把模型 `forward` 切回语言模型词表 logits 通路，不使用微调 checkpoint 中的 `lm_head_reg`。原因是开源权重没有回归头，直接沿用原测试脚本会随机初始化回归头，结果没有意义。

尝试了三种 ECR 输出方式：

1. `generate`：要求模型自由生成一个 `0~100` 数字；
2. `rubric`：输入中文 5D rubric，生成 `ECR + reason`；
3. `soft`：不生成数字，读取下一个 token 位置上 `excellent / good / fair / poor / bad` 的概率并加权，得到连续值。

完整汇总如下：

| 方案 | SRCC | PLCC | Final | 唯一预测数 |
|---|---:|---:|---:|---:|
| 微调模型 + 回归头（参照） | 0.6866 | 0.7007 | **0.6922** | 445 |
| soft 评级词概率 | -0.0039 | 0.1217 | **0.0463** | 500 |
| rubric + few-shot + 蒸馏 rubric | 0.0287 | 0.0691 | 0.0449 | 17 |
| rubric + few-shot | 0.0166 | 0.0396 | 0.0258 | 7 |
| generate 裸数字 | -0.0255 | 0.0780 | 0.0159 | 11 |
| soft + 蒸馏 rubric | -0.0632 | 0.0639 | -0.0123 | 500 |
| generate + 蒸馏 rubric | -0.0779 | 0.0742 | -0.0171 | 10 |
| rubric 去 few-shot + 蒸馏 rubric | -0.1682 | 0.1998 | -0.0210 | 8 |
| rubric 去 few-shot | -0.1768 | 0.2033 | -0.0247 | 12 |

诊断结论：

- 自由生成会坍缩到少数圆整数值；`soft` 能产生 500 个不同值，但相关性仍很弱。
- few-shot 版本中 344/500 条恰好输出 `0.5`、121/500 条输出 `0.85`，且 reason 会复制锚点文本，说明模型倾向于照抄示例而非执行公式。
- 去掉 few-shot 后输出更连续，也开始描述真实画面，PLCC 从 `0.0396` 增至 `0.2033`；但 SRCC 从 `0.0166` 降至 `-0.1768`，暴露出系统性反向排序。
- 原分析发现模型偏好制作精良、构图好、光线充足的视频，而真实平台 ECR 可能更偏好粗糙但钩子强的原生内容。对预测取反可把 SRCC 变为正值，但这是观察测试集结果后的操作，属于信息泄漏，不能作为合法提交方案。

### 6.2 从微调模型蒸馏 rubric

`videollama2/distill_ecr_rubric.py` 尝试让强微调模型自述其判断逻辑，再把该文本注入零样本模型。流程为：

1. **Self-report**：让微调模型描述评分维度；
2. **Contrastive**：对比高/低 ECR 视频总结差异；
3. **Final rubric**：合并前两步，生成可复用 rubric。

Self-report 得到了 11 个维度：Opening-second hook、Visual novelty、Motion/pace、Subject clarity、Cognitive payoff/curiosity gap、Aesthetic quality、Text overlay strategy、Title-content congruence、Emotional valence/arousal、Familiar-vs-novel framing、Platform fit。Contrastive 阶段失败，只返回拒答文本，因此最终 rubric 主要来自 self-report 和模型给出的权重。

这条链路修复过两个关键问题：

- 该 fork 的 `generate()` 返回值只包含新 token，旧实现再次按 prompt 长度切片，会把答案开头截掉；
- JSON 解析失败时，旧渲染器会静默生成只有标题的空 rubric。修复后改为回退保存原始文本。

仓库中的 `rubric_out/04_reusable_prompt.md` 只有约 374 字节，是旧的近空产物；`rubric_out_v3/` 与 `zero_shot_ecr/rubric_distilled/` 中约 3.8KB 的版本才是修复后保留的原始 rubric。尽管蒸馏文本包含可解释维度，它没有稳定提高零样本指标，说明“微调模型可自述部分线索”不等于“这些自然语言线索足以复现回归头能力”。

### 6.3 5D/12D 直出与逐维分治 a/b/c/d

在后续第一套结构化实验中，每个维度先独立输出 `0 / 0.5 / 1 + reason`，再由额外一次 LLM 调用生成最终 ECR。a/b/c/d 的开关语义为：

| 情况 | 维度调用 few-shot | 聚合调用 few-shot | 聚合时重复完整维度定义 |
|---|---:|---:|---:|
| a | 有 | 有 | 否 |
| b | 无 | 无 | 否 |
| c | 有 | 有 | 是 |
| d | 无 | 无 | 是 |

完整汇总结果：

| 实验 | 最好配置 | SRCC | PLCC | Final | 完成度 |
|---|---|---:|---:|---:|---|
| 5D 逐维分治 | a：few-shot，聚合不重复定义 | 0.0448 | 0.1005 | **0.0671** | 500/500 |
| 12D 一次直出 | 保留 few-shot | 0.0210 | 0.0643 | **0.0383** | 500/500 |
| 12D 逐维分治 | c：few-shot，聚合重复维度定义 | 0.0845 | 0.0710 | **0.0791** | 499/500，1 条失败 |

这一代分治仍由 LLM 做最终聚合，只有 6–18 个唯一预测值，输出坍缩明显。5D 中重复完整定义反而下降；12D 中 c 高于 a，但总体差异较小。最重要的工程结论是：逐维推理本身可保留结构化证据，最终连续数值不应再交给 7B LLM 自由计算。

> 本节是仓库中 `comparison_requirement*.csv` 对“每维独立调用 + LLM 聚合”的记录，与第 2、3 章外部 CSV 中“将维度划成 2/3 个组 + Python 合并”的分组实验不是同一套实验，不能直接横向当作同参数比较。

### 6.4 soft 维度分数与时间窗口

为减少硬离散值和自由生成聚合的问题，后续实现了两个 soft 方案：

- `continuous`：prompt 要求模型生成 `[0,1]` 内至少三位小数；
- `ordinal-prob`：读取 `low / medium / high` 下一个 token 的 logits，在三者内部 softmax 后计算：

```text
score = P(low) × 0 + P(medium) × 0.5 + P(high) × 1
```

逐维结果支持两类确定性 Python 聚合：

- `weighted`：只做维度线性加权；
- `strict`：线性加权后继续应用 rubric 中的扣分/奖励规则；对于 `ordinal-prob`，规则触发使用概率最高的 hard label，连续期望仍用于加权。

时间窗口按 rubric 对每个维度裁剪视听证据，AV 模式中画面和音频使用相同 `[0,end]`：

- 5D：Hook `3s`，其余维度 `5s`；
- 12D：首帧 `0.5s`，文字 `1s`，表情/音画同步/主题 `2s`，节奏/色彩/互动/动作 `3s`，叙事/制作质量/CTA `5s`；
- 完整 rubric 一次直出统一使用前 `5s`。

实现不会创建临时视频，而是直接从原视频的目标时间范围均匀采样固定帧数，并让 FFmpeg 从同一范围读取音频。

### 6.5 soft 全视频消融（实验 1，已完整汇总）

实验 1 保持完整视频和完整音频，只改变 soft 评分方式，用于隔离 soft 自身的影响。38 个已汇总配置中，最好结果为：

| 配置 | SRCC | PLCC | Final | 唯一预测数 |
|---|---:|---:|---:|---:|
| 12D + few-shot + 逐维 ordinal-prob + Python weighted | 0.0794 | 0.0712 | **0.0761** | 500 |
| 5D + few-shot + 逐维 ordinal-prob + LLM 聚合（无定义） | -0.0123 | 0.1871 | 0.0674 | 104 |
| 5D + few-shot + 逐维 ordinal-prob + LLM 聚合（带定义） | 0.0547 | 0.0846 | 0.0667 | 59 |
| 12D + few-shot + 逐维 ordinal-prob + Python strict | 0.0343 | 0.1027 | 0.0617 | 490 |

`ordinal-prob + Python` 整体优于要求模型自由生成连续小数；后者仍会聚集在少量值，且 12D 某些组合出现失败样本。soft 本身能消除硬三档输出，但在不裁剪时间窗口时，最好 Final 仅 `0.0761`。

### 6.6 soft + 时间窗口（实验 3，已完整汇总）

`comparison_stage1.csv` 保留了 32 个配置。最好结果集中在逐维 `ordinal-prob + Python`：

| 配置 | SRCC | PLCC | Final | 唯一预测数 |
|---|---:|---:|---:|---:|
| 12D + few-shot + ordinal-prob + Python weighted | 0.1204 | 0.1279 | **0.1234** | 500 |
| 12D + few-shot + ordinal-prob + Python strict | 0.0952 | 0.1169 | 0.1039 | 488 |
| 5D + few-shot + ordinal-prob + Python strict | 0.0516 | 0.1550 | 0.0929 | 488 |
| 5D + few-shot + ordinal-prob + Python weighted | 0.0516 | 0.1451 | 0.0890 | 500 |
| 12D + 去 few-shot + ordinal-prob + Python weighted | 0.0666 | 0.0830 | 0.0732 | 500 |

在排除 LLM 聚合、只比较语义一致的 direct/Python 配置后，全视频与时间窗口共有 24 对可比设置，其中时间窗口版本 `18/24` 更好、全视频版本 `6/24` 更好。关键同参数变化为：

| 配置 | 全视频 | 时间窗口 | 差值 |
|---|---:|---:|---:|
| 12D + few-shot + ordinal-prob + Python weighted | 0.0761 | **0.1234** | +0.0473 |
| 12D + few-shot + ordinal-prob + Python strict | 0.0617 | **0.1039** | +0.0422 |
| 5D + few-shot + ordinal-prob + Python weighted | 0.0449 | **0.0890** | +0.0441 |
| 5D + few-shot + ordinal-prob + Python strict | 0.0573 | **0.0929** | +0.0356 |

这表明按维度限制证据范围与 `ordinal-prob + Python` 组合有明显正向信号。但它仍不能单独证明时间窗口的因果收益，因为原计划用于隔离时间窗口作用的实验 2 没有完整跑完。

此外，`continuous` 路线可靠性较差：例如 12D few-shot + Python strict 只有 286 条有效、214 条失败。比较不同配置时必须同时看 `n_failed`，不能只按 Final 排序。

### 6.7 离散评分 + 时间窗口（实验 2，未完成）

`run_experiment2_discrete_temporal.sh` 原计划覆盖：5D/12D × 有/无 few-shot × 直出/逐维 × Python weighted/strict/LLM 聚合。但仓库中只留下：

```text
experiment2_discrete_temporal_5d_fewshot_direct.csv
experiment2_discrete_temporal_5d_fewshot_direct_detail.csv
```

`comparison_experiment2_discrete_temporal.csv` 和 `comparison_experiment1_and_2.csv` 均不存在，因此实验矩阵没有完成。对唯一残留结果重新计算得到：500 条有效样本、仅 2 个唯一预测值，SRCC `-0.0250`、PLCC `-0.0274`、Final `-0.0260`。该结果只说明“5D few-shot 离散直出 + 时间窗口”仍严重坍缩，不能代表整个实验 2，也不足以单独估计时间窗口贡献。

### 6.8 当前代码可复现范围

仓库当前可直接追溯和复现的内容包括：

- 原始零样本 `generate / rubric / soft`；
- 5D/12D 一次直出和逐维 a/b/c/d；
- continuous / ordinal-prob soft；
- 完整输入与按维度时间窗口；
- Python weighted / strict 与 LLM 聚合；
- rubric 蒸馏及其 bug 修复；
- 对应逐样本 CSV 和汇总表。

第 5 章“后训练 ECR 与 12D 融合”的实现代码仍未在仓库中找到，目前只有恢复后的方案和结果说明。

## 7. 当前可靠结论

1. **零样本直接预测 ECR 远弱于后训练回归。** 原始零样本最好 Final 仅 `0.0463`；结构化分治虽有改善，仍不能替代约 `0.693` 的后训练模型。
2. **输出连续不等于学会排序。** rating-token soft 可产生 500 个唯一值，但基础版本 SRCC 仍接近零；自由生成连续小数也经常再次坍缩。
3. **few-shot 既可能造成锚点复制，也可能为分治提供校准。** 直接 rubric 中去除 few-shot 暴露反向排序；而 12D `ordinal-prob + Python` 最好结果保留了 few-shot，因此不能笼统认定应该全部删除。
4. **12D 比 5D 更有诊断价值。** 外部分组实验中最好零样本 Final 从 5D 的 `0.1239` 提高到 12D 的 `0.2241`；仓库内 soft 时间窗口实验中最好配置也来自 12D。
5. **逐维 ordinal-prob + 确定性 Python 聚合是保留代码中最稳定的零样本路线。** 它避免自由数值生成和 LLM 心算，且输出唯一性、完成度更好。
6. **时间窗口有正向相关证据，但独立消融不完整。** 24 对可比 soft 配置中 18 对提升，最好同参数提升 `+0.0473`；不过离散窗口实验只完成一项，尚不能做严格因果归因。
7. **7B LLM 不适合做最终连续数值聚合。** 无论是再次聚合分治结果、处理强后训练分数还是选择融合权重，都出现明显退化、坍缩或近似常数策略。
8. **简单线性融合的表面提升不能全部归因于正确 12D。** Shuffled 也经常提高，表明集成、尺度对齐和预测收缩贡献很大。
9. **动态门控比固定融合更有希望。** Confidence-aware 是当前最强结果；Disagreement 的多组参数方向一致，提示 12D 更适合在自身判断明确或与后训练模型明显冲突时参与修正。
10. **当前最优融合证据仍需复验。** `0.697487` 值得继续验证，但 bootstrap 区间仍跨零，不能把全部 `+0.004517` 都解释为正确 12D 的贡献。

## 8. 后续迭代建议

按优先级建议：

1. 在独立验证集或多个随机划分上复验 Confidence-aware `λ=0.1/0.2/0.3` 和 Disagreement 全参数网格，报告均值、方差与配对 bootstrap 区间；
2. 优先恢复第 5 章融合代码，以及固定随机种子、Correct/Shuffled/Constant 映射和参数清单；
3. 补完实验 2 的离散评分 + 时间窗口矩阵，和无窗口离散基线做严格配对，以隔离时间窗口的独立贡献；
4. 固定参数后再评估测试集，避免基于同一测试集挑选最优参数造成信息泄漏；
5. 分析 Confidence-aware / Disagreement 真正修正的样本，定位哪些维度、内容类型和时间窗口提供增量；
6. 对比 hard `0/0.5/1` confidence 与概率熵、维度方差、模型 logit margin 等更连续的置信度定义；
7. 使用 out-of-fold 预测学习轻量融合器，并保留 Correct / Shuffled / Constant 三组控制，防止把普通集成收益误判为分治有效；
8. 所有后续汇总必须同时报告 `n`、`n_failed`、`n_unique`、SRCC、PLCC 和 Final，避免失败样本或输出坍缩造成误判。

## 9. 可追溯文件

| 内容 | 文件/目录 |
|---|---|
| 零样本、5D/12D 分治总说明 | `VideoLLaMA2-audio_visual/zero_shot_ecr/README.md` |
| 原始零样本入口 | `VideoLLaMA2-audio_visual/zero_shot_ecr/run_all.sh` |
| 5D/12D 直出与 a/b/c/d | `VideoLLaMA2-audio_visual/zero_shot_ecr/run_divide_and_12d.sh` |
| 5D prompt | `VideoLLaMA2-audio_visual/zero_shot_ecr/ecr_rubric_prompt.py` |
| 12D prompt | `VideoLLaMA2-audio_visual/zero_shot_ecr/ecr_rubric_12_prompt.py` |
| soft 与时间窗口 prompt | `VideoLLaMA2-audio_visual/zero_shot_ecr/ecr_stage1_prompts.py` |
| soft 与时间窗口推理实现 | `VideoLLaMA2-audio_visual/zero_shot_ecr/infer_ecr_stage1.py` |
| soft 全视频实验 | `VideoLLaMA2-audio_visual/zero_shot_ecr/run_experiment1_soft_full.sh` |
| 离散时间窗口实验（未完成） | `VideoLLaMA2-audio_visual/zero_shot_ecr/run_experiment2_discrete_temporal.sh` |
| soft + 时间窗口实验 | `VideoLLaMA2-audio_visual/zero_shot_ecr/run_stage1_soft_temporal.sh` |
| 原始零样本汇总 | `VideoLLaMA2-audio_visual/zero_shot_ecr/results/comparison_500.csv` |
| 5D/12D a/b/c/d 汇总 | `VideoLLaMA2-audio_visual/zero_shot_ecr/results/comparison_requirement*.csv` |
| soft 全视频汇总 | `VideoLLaMA2-audio_visual/zero_shot_ecr/results/comparison_experiment1_soft_full.csv` |
| soft + 时间窗口汇总 | `VideoLLaMA2-audio_visual/zero_shot_ecr/results/comparison_stage1.csv` |
| rubric 蒸馏实现 | `VideoLLaMA2-audio_visual/videollama2/distill_ecr_rubric.py` |
| 修复后的蒸馏 rubric | `VideoLLaMA2-audio_visual/zero_shot_ecr/rubric_distilled/` |

后训练与 12D 融合部分目前仅恢复了实验设计和结果说明；对应实现代码在机器故障后未保留，本文不声称这些方案可由当前仓库直接复现。
