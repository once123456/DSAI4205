# Big Data 期末项目 — GPT 交接文档（满分导向）

> 用途：把本文件（或按任务切分后）交给 GPT，让它接手期末交付物。
> 交付人：庞智瀚（Phelan）。截止：**2026-11-22 23:55**。
> 仓库：`group project/`（本文件所在目录）。GitHub：`once123456/DSAI4205`。

---

## 0. 给 GPT 的总指令（每个任务前先读这段）

```
你是一名顶尖的数据科学课程助教兼技术写作专家，正在协助一支 5 人本科生小组完成
PolyU「DSAI4205 Big Data Analytics」的期末 Group Project，目标是拿到满分。

硬性规则（违反任何一条都会导致重做）：
1. 【禁止编造】所有数字、超参数、结论必须来自本文件「事实数据卡」或你读到的仓库文件，
   不得自行发明。没有的数字就留空或标注【待补充】。
2. 【对比口径】本组神经模型（BERT+MLP、Qwen）用的是 473 训练 / 118 验证 / 397 测试
   的固定划分，必须和「473-split 基线 macro-F1 = 0.4461」比较，绝不能拿「591-full 基线
   0.4662」去比——后者训练数据更多，不可直接对比（详见事实数据卡第 2 条）。
3. 【占位符】另外 3 位成员的模型尚未提供结果，报告里必须用【待 XX 成员填充】占位，
   不要替他们编模型和数字。
4. 【诚实写作】本项目的亮点之一就是"诚实的负结果"（Kai 的蒸馏/反事实增强未带来显著
   增益），这要作为加分点如实呈现，而不是抹掉。
5. 【结构对齐】报告严格按评分细则的章节组织（见第 2 节），因为评分是逐项按 rubric 打的。
6. 【语言】默认用【英文】写正式报告和 PPT（课程为英语授课、rubric 为英文，更利于拿分）。
   若最终决定与已交 interim 保持中文一致，把所有"英文"改回"中文"即可，其余不变。
```

---

## 1. 项目背景与现状速览

**任务**：Twitter 四分类情感分析（数据 `Data 5 - Twitter`），做一个文本分类决策支持系统。

| 项 | 值 |
|---|---|
| 标签 | 0=Negative, 1=Positive, 2=Irrelevant, 3=Neutral |
| 原始数据 | 训练 596 条 / 测试 397 条 |
| 清洗后 | 训练 591 条（去重+去跨集重叠）/ 测试 397 条 |
| 固定划分（神经模型用） | 473 训练 / 118 验证 / 397 测试 |
| 组员 | 5 人（庞智瀚 Phelan、Kai、另 3 位尚未提供模型） |

**已经完成、可直接用于报告的东西：**

- EDA + 数据清洗 + 基线：`reports/Twitter情感数据技术分析报告.md`（371 行，中文，含摘要/六类数据问题/EDA/基线），`reports/数据清理与分析.md`（239 行）。
- 14 张 EDA/基线图：`figures/01_class_distribution.png` … `14_tfidf_heatmap.png`。
- Phelan 的 BERT(冻结)+MLP：`models/phelan/`、`runs/phelan/`（结果见事实数据卡）。
- Kai 的 Qwen3-0.6B+LoRA 三组实验（+ 新增两组等量对照）：`models/kai/`、`runs/kai/`。
- 共享基线 notebook：`notebooks/TFIDF_LR_Baseline.ipynb`。

**还没做（交给你的）：** ① 期末报告 .docx　② PPT　③ 整合成一个 notebook。

---

## 2. 评分细则全解（满分拆解，逐条对齐）

### 期末报告 = 60%（内部权重再分 6 块）

| 板块 | 占比 | 满分要点（GPT 必须逐条覆盖） |
|---|---|---|
| **EDA & Data Cleaning** | 20% | 数据粒度（一条推文→一个实体级标签）；分布与类别不平衡（Irrelevant 最少，最多类≈最少类 1.6 倍，轻度不平衡）；缺失/无效/离群（编码损坏、`<unk>`、`RhandlerR`、拉长词、网址、`@`）；标准化（lowercase、去标点）与 tokenization；**文档化修改日志**（每条改动可追溯）。 |
| **Baseline Model** | 10% | TF-IDF(1-2 元词)+逻辑回归；报告 accuracy/precision/recall/F1 全四项；作为后续对比的基准锚点。 |
| **Advance Modelling** | 30% | N=5 个「词嵌入模型 + 独立分析分类器」；**明确写"如何组合嵌入与分类器"的高层方案**；深度（模型复杂度、是否课堂外、是否从零训练）。⚠️ 当前只有 Phelan 1 个合规 + Kai 1 个 LLM 消融，另外 3 人待补。 |
| **Refinement** | 10% | 超参调优证据（本组有 8 组搜索 + class_weight 消融）；不平衡处理（class_weight=balanced + 用 macro-F1 而非只看 acc）。 |
| **Uniqueness & Creativity** | 10% | 创新特征工程（EDA 发现的 4 个表层特征）；新颖方法（教师-学生蒸馏、反事实增强）；深度业务洞察。 |
| **Development & Documentation** | 20% | 开发文档完整度；详细进度表；个人分工清晰；迭代/协作证据（git 历史、消融实验）。 |

### PPT = 20%（12 分钟演讲，blackboard 交 .pptx）

| 评分维度 | 满分要点 |
|---|---|
| Engagement | 用真实推文例子讲故事，别只念数字 |
| Clarity & coherence | 逻辑线：问题→数据→方法→结果→业务洞察 |
| Delivery | 每页标注谁讲 |
| Teamwork | 分工页 + 交接顺畅 |
| Response to questions | 附一页"预判问答" |

### 交付物（11/22 23:55）
1. 报告 `.docx`
2. PPT `.pptx`（blackboard）
3. 单个整合 notebook `.ipynb`（"all source code + preprocessing + model impl in a single well-annotated notebook"）

---

## 3. 事实数据卡（禁止改动这些数字）

### 3.1 基线（TF-IDF + Logistic Regression）
- **473-split（与神经模型同划分，对比就用这个）**：macro-F1 **0.4461**、accuracy **0.4912**。
- 591-full（训练数据更多，仅供参考，**不可**与神经模型直接对比）：macro-F1 0.4662、accuracy 0.5063。
- 591-full 逐类 F1：Negative 0.616 / Positive 0.447 / Irrelevant 0.302 / Neutral 0.500。

### 3.2 Phelan 的 BERT(冻结)+MLP（`runs/phelan/bert_mlp_summary.md`）
- 结构：`google-bert/bert-base-uncased`（**冻结**，mean-pooling 768 维，revision `86b5e093`）→ 独立 MLP。
- 输入 772 维 = 768 句向量 + 4 个标准化表层特征（`exclaim_count`、`question_count`、`url_count`、`mention_count`）。
- 验证选超参：`hidden=[512,256]`, `dropout=0.5`, `lr=1e-3`, `class_weight=balanced`，**验证 macro-F1 0.5140**。
- **测试（397）**：macro-F1 **0.4843**、accuracy **0.5038** → 较 473-split 基线 **+3.8pp**。
- 逐类 F1：Negative 0.592 / Positive 0.518 / **Irrelevant 0.308** / Neutral 0.520。
- Irrelevant 召回 **0.333**（基线仅 0.197，显著改善）。
- 敏感性：去掉 1 条历史跨集重叠样本（396 条）→ macro-F1 0.4856、acc 0.5051。
- **class_weight 消融**（关键加分点）：同超参无 class_weight 时验证分 0.4767 → 加 class_weight 后 0.5140。

### 3.3 Kai 的 Qwen3-0.6B + LoRA（`runs/kai/*.md`）
- label-only（主结果）：测试 macro-F1 **0.4850**、accuracy 0.5088 → 较 473-split 基线 **+3.9pp**。
- 短解释蒸馏（教师=DeepSeek `deepseek-v4-pro`，946 请求 / ~326k tokens）：unfiltered 最好 **0.4886**（较同协议 label_only 0.4745 约 +1.4pp，但 bootstrap 区间跨 0，不显著）。
- 反事实增强（46/80 样本通过核验）：0.4873，较仅解释组 +0.17pp，同样不显著。
- 新增两组等量对照（rationale-matched / counterfactual-matched）：结果未跑，报告留【待 Kai 填充】。
- 定位：三组/五组是**同一模型的消融对照**，算 1 个模型；适合写进"创新方法"而非"独立嵌入+分类器"。

---

## 4. 任务清单与执行顺序

| 顺序 | 任务 | 产物 | 依赖 |
|---|---|---|---|
| A | 期末报告 .docx | `final_report.docx` | 事实数据卡 + 已有 markdown |
| B | PPT | `final_presentation.pptx` | 报告定稿 |
| C | 整合 notebook | `integrated_notebook.ipynb` | 三个 notebook 都在 |

> 建议 A 先行（报告是 PPT 和 notebook 的内容来源）。

---

## 5. 任务 A：期末报告 .docx

### A1 提示词（先写，产出报告骨架与可写实章节）

```
【任务】为一门大数据课程期末项目写正式报告的初稿，输出 Markdown（之后我会转 .docx）。

输入材料（请先通读，若无法读文件我会逐段粘贴）：
- reports/Twitter情感数据技术分析报告.md  ← EDA+清洗+基线，直接翻译改写进正式报告
- reports/数据清理与分析.md               ← 清洗修改日志
- runs/phelan/bert_mlp_summary.md         ← Phelan 模型
- runs/kai/*_summary.md                   ← Kai 模型
- 本交接文档的"事实数据卡"               ← 所有准确数字的唯一来源

报告结构（严格按此顺序，标题可用英文）：
1. Executive Summary（一页摘要，含关键结论 + 最佳模型宏 F1 0.4850 + 业务价值一句话）
2. Introduction & Data Overview（任务定义、实体级情感、粒度、数据规模）
3. Exploratory Data Analysis（分布、类别不平衡、表层特征相关性、可视化引用 figures/01-14）
4. Data Cleaning（六类问题逐条：编码损坏/网址/@/占位符/拉长词/重复；每条=发现问题→处理→影响；
   附"修改日志"表；说明为什么神经模型保留原文不做 lemmatize，而基线做标准化——这是设计决策）
5. Baseline Model（TF-IDF+LR；报告 acc/macro-F1/逐类 P-R-F1；明确写"473-split 基线 0.4461"与
   "591-full 基线 0.4662"的区别和各自用途）
6. Advanced Modelling（重点，占 30%）
   6.1 组合方案总述（高层方法：冻结嵌入→独立分类器，明确回应 rubric 的"integrate embedding
       + independent classifier"）
   6.2 模型 1：BERT(冻结)+MLP（Phelan）—— 完整写实
   6.3 模型 2：Qwen3-0.6B+LoRA（Kai）—— 完整写实，含三组消融 + 诚实的负结果
   6.4 模型 3/4/5：【待 XX 成员填充】—— 留空 + 一句话说明占位
7. Model Comparison（N 个模型 vs 基线，横向对比表；解释差异原因：BERT 上下文语义 vs 词袋、
   class_weight 对少数类的作用、Irrelevant 为何最弱）
8. Refinement（超参搜索表 8 组、class_weight 消融 0.477→0.514、不平衡处理策略）
9. Uniqueness & Creativity（4 个表层特征、教师-学生蒸馏、反事实增强、跨集重叠检测）
10. Business Insights（实体级情感对品牌/舆情监控的价值；Irrelevant 类=话题不匹配的实务含义）
11. Project Development & Documentation（进度表、5 人分工、git/消融迭代证据）

写作要求：
- 全英文（若要求中文，整篇改中文）。
- 每个数字都来自事实数据卡，逐类 F1 用表格呈现。
- "对比口径"错误是硬伤：BERT/Qwen 只和 0.4461 比，写清楚"same 473-train split"。
- 诚实呈现负结果：Kai 的蒸馏 +1.4pp、反事实 +0.17pp 都"区间跨 0、不显著"，这是严谨加分项，
  措辞用 "no statistically reliable gain" 而非 "improved"。
- 缺失的 3 个模型用【待 XX 填充】占位，绝不虚构。
```

### A2 提示词（写完初稿后，做满分自查与润色）

```
【任务】对上一步产出的报告初稿做满分质检，逐条对照 rubric 打勾/补漏：

- EDA 是否覆盖：粒度、分布、类别不平衡、缺失/无效/离群？
- 清洗是否覆盖 rubric 明列项：missing/outliers/invalid/duplicates/standardize/
  normalization(lowercase,punctuation)/tokenization(lemmatization/stemming)？缺哪条补哪条。
- Baseline 是否四项指标齐全、是否作为基准被反复引用？
- Advanced Modelling 是否显式写了一句"high-level approach for combining embedding + analytics"？
- Refinement 是否有超参证据 + 不平衡处理证据？
- Uniqueness 是否逐条点出"innovative feature / novel methodology / deep insights"三个词对应的内容？
- Development 是否有进度表 + 分工表 + 迭代证据？
- 全文数字是否与事实数据卡 100% 一致（尤其 0.4461 vs 0.4662、0.4843、0.4850、0.5140）？

输出：一份"已覆盖/待补充"清单 + 修订后的完整报告。
```

---

## 6. 任务 B：PPT（12 分钟，20%）

### B1 提示词

```
【任务】基于已定稿的报告，做一份 12 分钟演讲用的 PPT，输出逐页大纲 + 每页讲稿要点 + 每页配图
（引用 figures/ 里已有的 14 张图，不够再标注"需要新图"）。

结构（12 分钟 ≈ 12-15 页）：
1. 封面（题目+组员名）
2. 问题定义（Twitter 四分类、实体级情感、为什么重要）
3. 数据与挑战（596/397、类别不平衡、编码损坏等隐藏坑）
4. EDA 亮点（1-2 张图讲故事）
5. 清洗方法（六类问题一张表）
6. 基线（TF-IDF+LR 0.4461，作为锚点）
7. 进阶模型总览（5 个模型的组合方案图：嵌入→分类器）
8. BERT+MLP 详解（结构图 + 0.4843 +3.8pp）
9. Qwen+LoRA 详解（含诚实的负结果：蒸馏/反事实不显著）
10. 模型横向对比（一张大表：基线 vs N 模型，逐类 F1）
11. 精调与不平衡处理（class_weight 消融 0.477→0.514）
12. 业务洞察（决策支持系统的落地价值）
13. 团队分工与迭代（5 人分工 + git/消融证据）
14. 结论 + 未来工作
15. 附录：预判 Q&A（老师可能问：为什么 Irrelevant 弱？蒸馏为什么没用？N 模型哪个最好？）

要求：每页标注 presenter 姓名占位；口语化讲稿；数字用大字突出；配色简洁。
```

---

## 7. 任务 C：整合 notebook（技术交付物）

### C1 提示词

```
【任务】把三个 notebook 合并成一个"single, well-annotated Jupyter Notebook"的合并方案 +
每个 cell 的中英文 Markdown 注释文案。

三个来源：notebooks/TFIDF_LR_Baseline.ipynb、notebooks/kai/Kai_Sentiment_Experiments.ipynb、
notebooks/phelan/BERT_MLP_Sentiment.ipynb。

要求：
- 顺序：数据加载→清洗→EDA→固定划分→基线→BERT+MLP→Qwen（各节加 Markdown 说明"这节做什么、
  关键结果、为什么这么设计"）。
- 每个模型节开头写清：模型结构、超参、复现命令、结果（引用事实数据卡数字）。
- 注明固定划分 split_manifest.csv 是唯一真相，训练/验证/测试绝不互相泄漏。
- 不编代码，只给"合并方案 + 注释文案"；实际 cell 拼接由我在本地完成。
```

---

## 8. 满分自查 checklist（交之前逐条打勾）

- [ ] 报告/PPT/notebook 三件套齐全，且 notebook 是"单个"文件
- [ ] 对比口径全部是 473-split 基线 0.4461，没有误用 0.4662
- [ ] 显式写了"如何组合嵌入模型与独立分类器"的高层方案
- [ ] 5 个模型要么写实、要么【待 XX 填充】占位，无虚构
- [ ] 负结果如实呈现（蒸馏/反事实"不显著"）
- [ ] Refinement 有超参证据 + 不平衡处理证据
- [ ] ≥5 个数据统计图（figures/ 已有 14 张，够）
- [ ] 开发文档：进度表 + 5 人分工 + 迭代证据
- [ ] PPT ≤12 分钟、每页有 presenter、附 Q&A 页
- [ ] 所有数字与事实数据卡一致

---

## 9. 风险与待确认项（需真人处理，GPT 帮不了）

1. **另外 3 位成员还没交模型** → Advance Modelling 30% 的"5 个模型"满分依赖他们。建议催：
   每人 1 个「嵌入+独立分类器」模型（推荐 Word2Vec/GloVe + XGBoost / 随机森林 / SVM，正好补上
   "多样性"，避免全神经网络扣分）。
2. **Kai 的 Qwen 合规性**：端到端 LLM 可能不满足"独立嵌入+独立分类器"——已交 Kai 自己判断，
   若不合规，报告里把它归入"创新方法"而非 N 个模型之一。
3. **报告语言**：默认英文（更利于拿分），若要与 interim 中文一致，整篇切换。
4. **PPT 是否需要真人录视频/露脸**：Big Data 无此要求（那是 HCI 的），12 分钟现场讲即可。
