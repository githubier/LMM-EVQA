# SnapUGC ECR 实验方案与阶段性结果

> 本文用于恢复机器故障前已经完成、但尚未系统沉淀到仓库中的实验背景。当前只记录方案、结果和结论，不补写已经丢失的实验代码，也不重新运行实验。
>
> 信息来源：工作区根目录的 `SnapUGC数据集DBDE结果.csv`（重点为第 29–86 行）和 `后训练ECR与12D分治融合实验说明.md`（对应 CSV 第 76–86 行的详细记录）。指标均在 500 条 SnapUGC 测试样本上计算。

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

## 6. 当前可靠结论

1. **零样本直接预测 ECR 远弱于后训练回归。** 5D/12D 分治改善了零样本结构化判断，但仍不能替代后训练模型。
2. **12D 比 5D 更有诊断价值。** 最好零样本分治 Final 从 5D 的 `0.1239` 提高到 12D 的 `0.2241`。
3. **7B LLM 不适合做最终连续数值聚合。** 无论是直接输出最终 ECR、再次聚合分治结果还是选择融合权重，都出现明显退化或近似常数策略。
4. **简单线性融合的表面提升不能全部归因于正确 12D。** Shuffled 也经常提高，表明集成、尺度对齐和预测收缩贡献很大。
5. **动态门控比固定融合更有希望。** Confidence-aware 是当前最强结果；Disagreement 的多组参数方向一致，提示 12D 更适合在自身判断明确、或与后训练模型明显冲突时参与修正。
6. **当前证据仍需复验。** `0.697487` 是值得继续验证的候选，但 bootstrap 区间仍跨零，不能把全部 `+0.004517` 都解释为正确 12D 带来的收益。

## 7. 后续迭代建议

按优先级建议：

1. 在独立验证集或多个随机划分上复验 Confidence-aware `λ=0.1/0.2/0.3` 和 Disagreement 全参数网格，报告均值、方差与配对 bootstrap 区间；
2. 固定参数后再评估测试集，避免基于同一测试集挑选最优参数造成信息泄漏；
3. 分析 Confidence-aware / Disagreement 真正修正的样本，定位哪些维度、哪些内容类型提供增量；
4. 对比 hard `0/0.5/1` confidence 与概率熵、维度方差、模型 logit margin 等更连续的置信度定义；
5. 使用 out-of-fold 预测学习轻量融合器，并保留 Correct / Shuffled / Constant 三组控制，防止把普通集成收益误判为分治有效；
6. 若恢复实验代码，优先恢复可重复的指标计算、随机种子、shuffle 映射和参数清单，而不是先扩展更多 prompt 变体。

## 8. 可追溯文件

- 零样本、5D/12D 分治代码与结果：`VideoLLaMA2-audio_visual/zero_shot_ecr/`
- 5D prompt：`VideoLLaMA2-audio_visual/zero_shot_ecr/ecr_rubric_prompt.py`
- 12D prompt：`VideoLLaMA2-audio_visual/zero_shot_ecr/ecr_rubric_12_prompt.py`
- soft / 时间窗口实验：`VideoLLaMA2-audio_visual/zero_shot_ecr/infer_ecr_stage1.py`
- 已有运行与结果说明：`VideoLLaMA2-audio_visual/zero_shot_ecr/README.md`

后训练与 12D 融合部分目前仅恢复了实验设计和结果说明；对应实现代码在机器故障后未保留，本文不声称这些方案可由当前仓库直接复现。
