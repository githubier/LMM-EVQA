# zero_shot_ecr — 用原始开源 VideoLLaMA2 直接 prompt 出 ECR

这个目录里的所有东西都是**本轮新增的**，和仓库原有的「微调 + 回归头」链路完全隔离。

- **原仓库链路**（`../videollama2/test.py`、`../prepare_dataset.py` 等）：用微调权重
  `videollama2_EVQA_weights_mse`，靠回归头 `lm_head_reg` 直接吐一个标量，**不经过 prompt**。
- **本目录链路**：用原始开源权重 `DAMO-NLP-SG/VideoLLaMA2.1-7B-AV`，**不做任何训练**，
  纯靠 prompt 让大模型用自然语言写出 ECR。

唯一动到目录外的文件是 `../videollama2/distill_ecr_rubric.py`（修了两个 bug，见文末）。

---

## 为什么需要单独写一套推理代码

`../videollama2/model/videollama2_qwen2.py` 里的 `forward` 被写死成回归通路：

```
hidden_states -> mean-pool -> lm_head_reg (MLP) -> 标量
```

开源权重里**根本没有** `lm_head_reg`（对比过两份 checkpoint 的 index：微调版 1160 个
key，开源版 1156 个，差的正好是 `lm_head_reg.fc1/fc2` 的 weight+bias）。所以直接拿开源
权重跑原脚本，`lm_head_reg` 会被随机初始化，输出是噪声；`generate()` 也会崩，因为它拿到
的是标量而不是词表 logits。

好在原始的 `lm_head` 完好保留着。`infer_ecr_prompt.py` 用一个上下文管理器在推理期间把
`forward` 临时切回标准语言模型通路（`hidden -> lm_head -> vocab logits`），推理结束再还原。
**权重本身一个字节都没改。**

---

## 文件说明

### 脚本

| 文件 | 作用 |
|---|---|
| `prepare_dataset_500.py` | 把 `SnapUGC/test_data_500.csv` 转成推理用的 `test_500.json`。**注意这个 csv 没有表头**，第一行就是数据，直接 `read_csv` 会丢一条并把 ID 当列名，这里用 `header=None` 处理，再从 `test_truth.csv` join 出 ECR 真值。 |
| `ecr_rubric_prompt.py` | 你提供的那套中文 rubric prompt。拆成 `ROLE_PLAY` / `TASK_DESC` / `FEWSHOT_ANCHORS` / `CAVEATS` / `OUTPUT_LIMITATION` 几块，方便单独消融。`build_rubric_prompt(title, desc, use_fewshot=)` 组装成最终 prompt。 |
| `infer_ecr_prompt.py` | 推理主脚本。三种打分模式（见下），支持批量 / 单视频，跑完直接打印 SRCC/PLCC/final_score。 |
| `compare_results_500.py` | 读 `results/*_detail.csv`，输出各方案的 SRCC / PLCC / final_score 对比表。 |
| `run_all.sh` | 一键复现全部实验的命令清单。 |

### 三种打分模式（`--score-mode`）

| 模式 | prompt | 怎么得到数值 | 适用 |
|---|---|---|---|
| `rubric` | 你的中文 rubric | 模型写 `ECR: 0.56 ; reason: ...`，正则抠 | 有可解释的 reason |
| `soft` | 一句简短英文提问 | **不生成**，读 5 个评级词（excellent/good/fair/poor/bad）在下一个 token 位置的概率做加权平均 | 需要连续分数排序 |
| `generate` | 一句英文"只输出 0-100 数字" | 模型写个数字，正则抠 | 基线对照 |

`soft` 用的就是仓库里已有的 `wa5()` 思路，只是原作者没在推理链路上用起来。它的好处是输出
**连续**（500 条视频 500 个不同值），而生成式打分会collapse到 0.5 / 0.85 这种整数。

常用参数：
- `--no-fewshot`：rubric 模式下去掉 few-shot 锚点（原因见下）
- `--rubric-file`：把一份额外的 rubric markdown 拼到 prompt 最前面
- `--limit N`：只跑前 N 条，调试用

### 数据与产物

| 路径 | 内容 |
|---|---|
| `test_500.json` | 500 条测试视频（路径 + 标题 + 描述 + ECR 真值） |
| `results/*_detail.csv` | 每条视频的 `Id, ECR(预测), raw_answer(模型原话), ECR_gt(真值)`。**想看模型到底说了什么就翻这个文件的 `raw_answer` 列。** |
| `results/*.csv`（不带 _detail） | 只有 `Id, ECR` 两列，提交格式 |
| `results/comparison_500.csv` | 汇总对比表 |
| `rubric_distilled/` | 蒸馏 rubric，见下节 |

---

## `rubric_distilled/` 是什么

这**不是**你给的 prompt，来源完全不同。

思路：微调模型 SRCC 有 0.69，说明它确实学到了东西。那能不能让**微调模型**把「我到底看哪些
维度」用自然语言讲出来，再把这段文字喂给零样本的开源模型当小抄？

`../videollama2/distill_ecr_rubric.py` 干的就是这件事，分三步：

| 文件 | 是什么 |
|---|---|
| `01_self_report.*` | 让微调模型自述评分维度。**这一步成功了**，吐出 11 个维度（Opening-second hook / Visual novelty / Motion-pace / Subject clarity / Cognitive payoff / Aesthetic quality / Text overlay / Title-content congruence / Emotional valence / Familiar-vs-novel / Platform fit）。 |
| `02_contrastive.*` | 想让它对比高 ECR / 低 ECR 两组视频找规律。**这一步失败了**，模型只回了一句 `I'm sorry, I cannot assist with this request.` |
| `03_final_rubric.*` | 把前两步合并。内容基本等于 01 加上了数值权重。 |
| `04_reusable_prompt.md` | 渲染成可直接注入的 markdown，就是 `--rubric-file` 用的那个。 |

`.raw.txt` 是模型原始输出，`.json` 是解析后的。**建议直接看 `.raw.txt`**，因为 7B 模型
输出的 JSON 有不少格式错误（用单引号收尾、key 里的下划线丢了、键名前多空格），解析器只能
尽力修。

### 修的两个 bug

**Bug 1 — 生成结果被错误切片**（致命）

```python
gen = out[0][input_ids.shape[1]:]   # 原代码
```

这个 fork 的 `generate()` 内部把 `input_ids` 转成 `inputs_embeds` 再调父类，返回值**只含
新生成的 token，不带 prompt 前缀**。再切掉 `input_ids.shape[1]` 就把答案开头几百个 token
砍没了。所以旧的 `../rubric_out/` 里所有文件都是从半截 JSON 开始的，解析必然失败，最终
`04_reusable_prompt.md` 只剩一个空壳标题。

修完后蒸馏内容从 250 字节涨到 3.5 KB。

**Bug 2 — markdown 渲染静默产出空文件**

解析失败时 `render_markdown_prompt` 会输出一个只有标题、没有任何维度的 markdown。这意味着
`--rubric-file` 注入的是个 no-op，跑了等于没跑，而且不报错。现在加了兜底：解析失败就输出
原始文本。

---

## 结论

在 500 条测试集上（`final_score = 0.6*SROCC + 0.4*PLCC`）：

| 方案 | SRCC | PLCC | final_score | 唯一值 |
|---|---|---|---|---|
| **[参照] 微调 + 回归头** | **0.6866** | **0.7007** | **0.6922** | 445 |
| soft（评级词概率） | -0.0039 | 0.1217 | 0.0463 | 500 |
| rubric 带锚点 + 蒸馏 | 0.0287 | 0.0691 | 0.0449 | 17 |
| rubric 带锚点（原样） | 0.0166 | 0.0396 | 0.0258 | 7 |
| generate（裸数字） | -0.0255 | 0.0780 | 0.0159 | 11 |
| soft + 蒸馏 | -0.0632 | 0.0639 | -0.0123 | 500 |
| generate + 蒸馏 | -0.0779 | 0.0742 | -0.0171 | 10 |
| rubric 去锚点 + 蒸馏 | -0.1682 | 0.1998 | -0.0210 | 8 |
| **rubric 去锚点** | **-0.1768** | **0.2033** | **-0.0247** | 12 |

零样本的所有变体 final_score 都在 **-0.03 ~ 0.05** 之间，而同一份测试集上微调模型是
**0.69**（和论文报的 0.71 吻合，说明测试集和评测代码本身没问题）。

差距不是 pipeline bug，是任务性质决定的：ECR 是平台侧的行为标签，取决于推荐分发、用户画像、
发布时间等视频内容之外的因素，光看画面和标题推不出来。这个结果本身量化了论文里微调增益的来源。

### few-shot 锚点消融：症状治好了，分数反而更低

带锚点跑 500 条，**344 条答案恰好是 0.5**，121 条是 0.85，只有 7 个不同取值。看 reason：

```
ECR: 0.5; reason: 无钩子+静态画面
```

`无钩子+静态画面` 是 few-shot 里第一个锚点的 title 原文——模型没执行加权公式，直接抄了
最近的锚点。

去掉锚点后（`--no-fewshot`）确实治好了这个症状：

```
ECR: 0.62 ; reason: The video shows a view of the water from Pier 56 in Seattle...
ECR: 0.47 ; reason: 1. blue tshirt 2. throwing object 3. glass table...
```

数值变成 0.62 / 0.47 这类非整数，reason 也变成对画面的实际描述。**但 final_score 反而从
0.0258 掉到 -0.0247。**

我原本预期去掉锚点会涨，结果是错的。拆开看两个指标方向相反：

- **PLCC 从 0.0396 涨到 0.2033**（涨了 5 倍）— 线性相关变好了，说明模型确实在"看视频"，
  预测值和真值有了微弱的线性关系
- **SRCC 从 0.0166 掉到 -0.1768** — 排序能力变成了负相关

这个组合说明：模型开始基于画面内容打分了，但它的**打分方向和真实 ECR 是反的**。按预测值
分箱看真值均值，趋势是单调递减的：

| 预测 ECR 分箱 | 真实 ECR 均值 | 样本数 |
|---|---|---|
| 12.0 ~ 45.0（模型认为最差） | **55.76** | 160 |
| 45.0 ~ 47.0 | 59.20 | 68 |
| 47.0 ~ 56.0 | 46.25 | 182 |
| 56.0 ~ 85.0（模型认为最好） | **42.94** | 90 |

**模型打分最低的那批视频，真实完播率反而最高。** 把预测值取反，SRCC 就从 -0.1768 变成
+0.1768。

原因大概是：模型倾向于给"制作精良、构图好看、光线充足"的视频高分（它的 rubric 里
Production Quality 就占 10%），而短视频平台上高完播的往往是粗糙但有强钩子的原生内容。
带锚点时因为几乎不看视频（全输出 0.5），反而没暴露这个系统性偏差，SRCC 才勉强在 0 附近。

所以真实结论是：**7B 模型对"什么视频让人看得下去"的先验判断本身是反的，不是 prompt 工程
能修的**。锚点只是掩盖了问题，去掉锚点让问题显形了——从诊断角度这是好事。

### 如果还想往上试

1. **最省事的**：既然方向是反的且已验证（取反后 SRCC = +0.1768），直接对预测值取反就能拿到
   final_score ≈ 0.19。但这是在测试集上看到答案后做的决定，**属于信息泄漏，不能这么交**，
   必须先在独立的验证集上确认方向确实稳定为负。
2. 让模型**分别输出 5 个维度分**，在代码里算加权和，而不是让 7B 心算带惩罚项和非线性增强
   的公式。这样还能看出到底是哪个维度在带偏。
3. 在 prompt 里显式对抗这个偏差，比如加一句"制作粗糙但有强钩子的内容往往完播率更高，不要
   因为画质好就给高分"。
4. 换更强的模型（如 Qwen2.5-VL-72B）验证是「7B 先验错误」还是「任务零样本不可解」。
5. 混合方案：用 `soft` 拿连续排序分，用 `rubric` 拿可解释的 reason。

---

## 新增：分治评分与 12 维 rubric

### 新增文件与模式

| 项目 | 说明 |
|---|---|
| `ecr_rubric_12_prompt.py` | 用户提供的 12 维 rubric，包含原始权重、时间窗口、扣分规则和 six-shot 锚点。 |
| `run_divide_and_12d.sh` | 只运行本轮新增的 10 个实验，不重复旧的 soft/generate/5D 直出实验。 |
| `--score-mode rubric5-divide` | 每条视频先做 5 次独立维度评分，再做 1 次 LLM 聚合，共 6 次调用。 |
| `--score-mode rubric12` | 将完整 12 维 rubric 一次输入，直接生成 ECR。 |
| `--score-mode rubric12-divide` | 每条视频先做 12 次独立维度评分，再做 1 次 LLM 聚合，共 13 次调用。 |

分维度调用固定输出：

```text
dimension: <维度键> ; score: 0/0.5/1 ; reason: <实际视频依据>
```

最终聚合调用固定输出：

```text
ECR: xxx ; reason: xxx
```

最终 `ECR` 来自最后一次 LLM 调用，不是 Python 直接代算。detail CSV 额外保存：

- 每个维度的 `*_score`、`*_reason`、`*_raw_answer`；
- `weighted_base_before_penalty`：Python 按维度权重计算的未扣分基线，仅用于审计和排错。

### a / b / c / d 的参数对应

5 维和 12 维分治使用完全相同的开关语义：

| 情况 | 维度调用 few-shot | 聚合调用 few-shot | 聚合调用重复完整维度说明 | 参数 |
|---|---:|---:|---:|---|
| a | 有 | 有 | 无 | 默认 |
| b | 无 | 无 | 无 | `--no-fewshot` |
| c | 有 | 有 | 有 | `--aggregate-with-dimensions` |
| d | 无 | 无 | 有 | `--no-fewshot --aggregate-with-dimensions` |

`--no-fewshot` 会同时关闭前置维度调用和最终聚合调用中的 few-shot，不会出现前后阶段不一致。

### 按三个要求分别一键运行

三个要求已经拆成三个互相独立的 Shell 脚本。你可以从 `/workspace` 依次执行：

```bash
cd /workspace

# 要求 1：5 维分治 a/b/c/d
sh LMM-EVQA/VideoLLaMA2-audio_visual/zero_shot_ecr/run_requirement1_5d_divide.sh

# 要求 2：12 维一次性直出，带/不带 few-shot
sh LMM-EVQA/VideoLLaMA2-audio_visual/zero_shot_ecr/run_requirement2_12d_direct.sh

# 要求 3：12 维分治 a/b/c/d
sh LMM-EVQA/VideoLLaMA2-audio_visual/zero_shot_ecr/run_requirement3_12d_divide.sh
```

每个脚本结束后会分别生成本组的对比表：

```text
results/comparison_requirement1_5d_divide.csv
results/comparison_requirement2_12d_direct.csv
results/comparison_requirement3_12d_divide.csv
```

三个脚本都默认带 `--resume`，每完成一条视频都会更新 CSV。中断后重新执行对应脚本，会保留已有成功结果并重试未完成或失败的样本。

### 一次运行全部新增实验（可选）

原来的总入口仍然保留：

```bash
cd /workspace
sh LMM-EVQA/VideoLLaMA2-audio_visual/zero_shot_ecr/run_divide_and_12d.sh
```

总脚本默认带 `--resume`，每完成一条视频都会更新 CSV。中断后重新执行同一命令，会保留已有成功结果并重试未完成或失败的样本。

调用量较大：

- 5 维分治每种：`500 × 6 = 3000` 次生成；4 种共 12000 次；
- 12 维直出两种：`500 × 1 × 2 = 1000` 次生成；
- 12 维分治每种：`500 × 13 = 6500` 次生成；4 种共 26000 次；
- 全部新实验合计约 39000 次生成。

建议先用单条或小样本验证输出格式：

```bash
cd /workspace/LMM-EVQA/VideoLLaMA2-audio_visual/zero_shot_ecr
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/root/anaconda3/envs/videollama2/bin/python infer_ecr_prompt.py \
  --model-path /root/workspace/cvuaggk7v38s73dgjft0/videollama2weights \
  --modal-type av --json-file test_500.json --limit 1 \
  --score-mode rubric12-divide --no-fewshot --aggregate-with-dimensions \
  --verbose --save-csv results/smoke_rubric12_divide_d.csv
```

12 个输入权重按用户给出的四位小数原样保留，数值合计为 `0.9996`（四舍五入误差），代码不会暗中改写或重新归一化。

---

## 第一阶段优化：soft 维度分数 + 时间窗口

新增统一执行入口：

```bash
cd /workspace
sh LMM-EVQA/VideoLLaMA2-audio_visual/zero_shot_ecr/run_stage1_soft_temporal.sh
```

脚本会运行 32 个结果组合：

| 变量 | 取值 |
|---|---|
| rubric | 5D / 12D |
| few-shot | 有 / 无 |
| 流程 | 完整 rubric 一次直出 / 逐维评分 |
| soft 方法 | prompt 生成连续 `0~1` / `low-medium-high` next-token 概率期望 |
| 逐维聚合 | Python 纯加权 / Python 严格公式 / LLM 最终聚合 |

其中一次直出不存在“Python/LLM 聚合”之分，因为它没有前置维度结果：完整 rubric 本身就是唯一一次 LLM 调用。

### 两种 soft 方法

1. `continuous`：修改 prompt，要求模型生成至少三位小数的 `score: 0.xxx` 或 `ECR: 0.xxx`，解析时不再吸附到 `0/0.5/1`。
2. `ordinal-prob`：不生成分数，读取下一个 token 对 `low/medium/high` 的 logits，三者内部 softmax 后计算：

```text
score = P(low) × 0 + P(medium) × 0.5 + P(high) × 1
```

当前 tokenizer 中三个候选均为单 token，已做启动校验。逐维严格公式的离散触发条件使用概率最高的 hard label，连续期望值仍用于线性加权。

### 时间窗口

AV 模式下画面和音频使用完全相同的 `[0, end]` 范围：

- 12D：首帧 `0.5s`；文字 `1s`；表情/音画同步/主题 `2s`；节奏/色彩/互动/动作 `3s`；叙事/制作质量/CTA `5s`。
- 5D：Hook `3s`；其余未明确规定或需要完整观察的维度使用前 `5s`。
- 完整 rubric 一次直出统一使用前 `5s`。

### 结果复用与断点续跑

最昂贵的逐维推理只有 8 组：

```text
5D/12D × few-shot开关 × continuous/ordinal-prob
```

每组先生成严格 Python 聚合及 detail CSV；纯加权 Python 和 LLM 聚合直接用 `--dimensions-from` 复用同一组维度特征，不会重复 5/12 次维度推理。所有任务均使用 `--resume`，每条视频后写 checkpoint。

最终汇总：

```text
results/comparison_stage1.csv
```

相关文件：

| 文件 | 用途 |
|---|---|
| `ecr_stage1_prompts.py` | 第一阶段 prompt 与 5D/12D 时间窗口映射 |
| `infer_ecr_stage1.py` | 独立推理器，连续生成、等级概率、时间窗口、三类聚合 |
| `compare_stage1.py` | 自动扫描并汇总所有 `stage1_*_detail.csv` |
| `run_stage1_soft_temporal.sh` | 32 个结果组合的一键断点续跑入口 |

### 三项消融实验入口

三项实验严格分开，避免把 soft 与时间窗口的收益混在一起：

| 实验 | 维度评分 | 视频/音频输入 | 执行脚本 |
|---|---|---|---|
| 实验1 | `continuous` 或 `ordinal-prob` soft | 完整视频 + 完整音频 | `run_experiment1_soft_full.sh` |
| 实验2 | 原始离散 `0/0.5/1` | 按维度裁到前0.5/1/2/3/5秒 | `run_experiment2_discrete_temporal.sh` |
| 实验3 | `continuous` 或 `ordinal-prob` soft | 按维度裁到前0.5/1/2/3/5秒 | `run_stage1_soft_temporal.sh`（已运行） |

实验1：

```bash
cd /workspace
sh LMM-EVQA/VideoLLaMA2-audio_visual/zero_shot_ecr/run_experiment1_soft_full.sh
```

输出汇总：

```text
results/comparison_experiment1_soft_full.csv
```

实验2：

```bash
cd /workspace
sh LMM-EVQA/VideoLLaMA2-audio_visual/zero_shot_ecr/run_experiment2_discrete_temporal.sh
```

输出汇总：

```text
results/comparison_experiment2_discrete_temporal.csv
```

两个脚本均覆盖 5D/12D、有/无 few-shot、完整 rubric 直出、分维度 Python 纯加权/严格公式，以及 LLM 聚合不带/带完整维度说明。逐维结果只计算一次，其他聚合方式从 detail CSV 复用。

时间窗口不会创建临时 MP4，也不是每秒固定取一帧。视频处理器直接在原文件中计算 `[0,end]` 的帧范围，再在范围内按原 VideoLLaMA2 `uniform` 规则均匀采样固定帧数；AV 音频由 FFmpeg 从原视频直接读取同一个 `[0,end]` 范围到内存。

