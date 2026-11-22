# Kai 的情感分类模型

**无需教师缓存的 A、E 两组入口：** 使用 `python models/kai/run_frozen_ae.py --arms A E --seeds 42`，比较冻结 Qwen3-0.6B 与冻结 Qwen3-Embedding-0.6B，分别训练相同结构的独立 MLP。仅此入口不依赖教师缓存，也不执行或改写统一 notebook。[安装、运行、续跑与 E 盘报告说明](frozen-ae/README.md)。输出位于 `runs/kai/qwen_frozen_ae_runs/`。

本实验研究在只有几百条标注推文时，教师模型提供的短解释和情感翻转样本能否改善 Qwen3-0.6B 的四分类表现。标签编码为 0（Negative）、1（Positive）、2（Irrelevant）、3（Neutral）。

原来的三组实验用 Qwen 解码器对四个标签代码打分，改变训练监督。新增 `embedding-mlp` 模式将表示模型与分析模型分开：先适配并冻结 Qwen，提取上下文文本向量，再单独训练 MLP 分类器。五种表示设置共用同一种 MLP，属于一个分析模型的对照实验，不能计作五种不同分析模型。课程建模要求见课程平台上的项目说明。

## 在 Colab 运行

训练方法、等量对照和冻结表示实验共用一个 notebook。在 Colab 选择 GPU，在首格的 `RUN_MODE` 下拉框选择实验，再点「全部运行」。[打开 Colab](https://colab.research.google.com/github/once123456/DSAI4205/blob/main/group%20project/notebooks/kai/Kai_Sentiment_Experiments.ipynb) · [查看 notebook](../../notebooks/kai/Kai_Sentiment_Experiments.ipynb)

| RUN_MODE | 模型与实验 |
|---|---|
| `label-only` | [原始标签 LoRA 微调](label-only/README.md)，按验证集 macro-F1 早停 |
| `rationale-distillation` | [短解释蒸馏](rationale-distillation/README.md)，比较原始标签、未筛选解释和筛选解释三种训练方式 |
| `counterfactual` | [反事实增强](counterfactual/README.md)，比较仅用筛选解释训练与加入情感翻转样本后的表现 |
| `rationale-matched` | 固定解释条数和各类数量，比较严格筛选与随机抽取 |
| `counterfactual-matched` | 固定改写条数，比较平衡和原比例的翻转方向 |
| `embedding-mlp` | 四种冻结 Qwen 表示＋独立 MLP，并加入 Qwen3-Embedding-0.6B＋MLP |

首次生成解释时，在 Colab Secrets 中设置 `TEACHER_API_KEY`，运行解释蒸馏并保留导出的 ZIP。接着运行反事实实验，将 `DISTILLATION_EXPORT` 指向该 ZIP，即可沿用同一份数据划分和教师解释缓存。首次生成反事实样本也会调用教师 API。

### 在单张 5090 上跑等量对照

在之前训练成功的 Python 环境中更新仓库，保留原来的 `runs/kai/`。在同一 notebook 首格选择 `rationale-matched` 并运行全部单元，然后重启内核，选择 `counterfactual-matched` 再运行。两个模式都读取既有教师记录，每次只训练一个学生。也可以在 `group project` 目录执行以下命令。

```bash
python models/kai/run_local.py --mode rationale-matched
python models/kai/run_local.py --mode counterfactual-matched
```

默认从 `runs/kai/last_rationale_distillation.json` 和 `last_counterfactual.json` 定位各自的完整导出。迁移过文件时，在配置格填写 `PRIOR_EXPORT`，或在命令末尾加 `--prior-export "实际路径"`。解释对照选择上次的 `qwen3_rationale_distillation.zip`，翻转对照选择 `qwen3_counterfactual.zip`。也支持对应的解压目录或运行目录。导出需含原划分、协议和教师解释记录，反事实导出还需含 `accepted_counterfactuals.csv`。公开仓库里的摘要和运行指针不足以还原逐条教师结果。

解释组先应用学生的 token 长度限制，再按人工类别确定配额。如果上一轮的 193 条严格解释全部可用，两组各取 Negative 84、Positive 78、Irrelevant 6、Neutral 25。随机池包含通过严格筛选的行，两组允许重合。所有原始训练行都保留分类监督。

翻转组沿用同一组原始标签与严格解释。每组的改写数等于较少方向样本数的两倍。上一轮共有 14 条负转正、32 条正转负，对应平衡组 14/14、原比例组 9/19（方向依次为负转正、正转负）。同方向共享随机顺序，以尽量复用同一批父样本。

结果分别写入 `runs/kai/qwen_rationale_matched_runs/` 和 `runs/kai/qwen_cf_matched_runs/`。先看 `matched_counts.csv` 和 `matched_selection.csv` 核对实际配额与样本，再看 `comparison.csv` 和 `paired_bootstrap.csv`。差值方向分别为严格减随机、平衡减原比例。默认抽样种子 `MATCH_SEED=42` 与学生训练种子 `SEEDS=[42, 43, 44]` 分开记录。复查抽样敏感性可更改 `MATCH_SEED`（命令行 `--match-seed`）并开启新运行。

### 在单张 5090 上跑五组冻结表示＋MLP

使用之前训练成功的 CUDA/PyTorch 环境，从 `group project` 目录执行：

```bash
python models/kai/run_local.py --mode embedding-mlp
```

默认读取 `runs/kai/last_counterfactual.json` 所指的完整反事实结果。文件迁移过，或者在 Colab 上运行时，明确指定原 `counterfactual` 导出的 ZIP、`export` 文件夹或运行目录：

```bash
python models/kai/run_local.py --mode embedding-mlp --prior-export "/path/to/qwen3_counterfactual.zip"
```

直接运行 notebook 时，首格选择 `RUN_MODE="embedding-mlp"`，配置格填写 `PRIOR_EXPORT`，重启内核后运行全部单元。导出必须包含原 `experiment_protocol.json`、`split_manifest.csv`、教师缓存分片或 `teacher_responses.json`，以及 `accepted_counterfactuals.csv`。公开仓库中的运行摘要不足以启动此实验。缓存缺失会停止，**不会补发教师请求，也不需要教师 API 密钥**。初次运行仍需联网下载 Hugging Face 模型。

| 输出组名 | 表示模型 | 表示适配时使用的监督 | MLP 训练行 |
|---|---|---|---|
| `qwen_base` | Qwen3-0.6B | 无任务适配 | 原始训练行 |
| `qwen_label` | Qwen3-0.6B＋LoRA | 原始标签 | 同上 |
| `qwen_rationale` | Qwen3-0.6B＋LoRA | 原始标签＋严格筛选短解释 | 同上 |
| `qwen_rationale_cf` | Qwen3-0.6B＋LoRA | 原始标签＋相同短解释＋缓存中全部已核验改写 | 同上 |
| `qwen_embedding` | Qwen3-Embedding-0.6B | 不做本任务微调 | 同上 |

每个 seed 从共同的基础 revision 重新训练三个 adapter，保持 LoRA 配置、原始样本曝光次数和优化步数一致。三个 adapter 都按生成式标签验证 macro-F1 选择 checkpoint；旧 standalone `label-only` 的早停结果不直接混入本次比较。为固定优化步数，此模式的三个 adapter 不使用早停，均训练到预设 epoch，再恢复验证 macro-F1 最佳 checkpoint。反事实组增加辅助前向计算，不声称计算量相同。这里使用完整的已接受反事实集合；固定数量的方向平衡研究仍在 `counterfactual-matched` 模式中。

选好 adapter 后冻结全部参数并关闭 dropout。前四组使用相同的分类 chat 提示，输入只有任务说明与推文，取最后有效 token 的最终层隐藏状态。它处于生成答案之前，不包含真实标签、教师解释或模型生成的答案。专用 Embedding 组使用官方 `Instruct/Query` 格式与最后有效 token pooling。所有向量都做 L2 归一化；默认两种 0.6B 模型均为 1024 维，代码会检查维度相同。第五组的模板不同，属于现成嵌入模型的参考，不是只替换权重的严格消融。

独立分类器采用 scikit-learn `MLPClassifier`，结构为 `1024 → 128（ReLU）→ 4`，Adam 优化，batch 32，学习率 0.001，L2 正则 `alpha=0.01`，最多 200 epoch，验证 macro-F1 连续 20 epoch 不改善则停止，并恢复最佳 epoch。`StandardScaler` 只拟合原始训练向量，验证/测试只做变换。MLP 在 CPU 上训练，不向 Qwen 反传梯度；五组都只使用同一批 473 条原始训练行，反事实只参与 Qwen 适配。全部 MLP 选型结束后才统一读取测试标签和评估；此前仅读取测试文本做重复检查。

一次仅加载一个 Qwen，默认使用 BF16（不支持时 FP16）、SDPA，无需 FlashAttention 或量化。向量提取 batch 为 8，显存不足可在新运行中调低 `EMBED_BATCH_SIZE`；LoRA 沿用 batch 2、梯度累积 8。保留已经能运行 5090 的 PyTorch，运行器默认跳过安装单元。`SMOKE_TEST=True` 每个 adapter 只训练两步，MLP 只运行两轮，跳过测试评估，不能用作正式结果。

默认单 seed 42。正式比较建议开启新运行，使用三个种子：

```bash
python models/kai/run_local.py --mode embedding-mlp --prior-export "/path/to/qwen3_counterfactual.zip" --seeds 42 43 44
```

中断后使用相同配置和种子续跑：

```bash
python models/kai/run_local.py --mode embedding-mlp --prior-export "/path/to/qwen3_counterfactual.zip" --resume-dir "runs/kai/qwen_embedding_mlp_runs/实际运行目录"
```

已完成的 adapter、向量缓存和 MLP 会复用；未完成的单组训练从头重做。续跑检查数据、教师内容、反事实内容、模型 revision、软件版本、实现指纹和超参数；若不同则要求开启新运行。向量缓存另含输入顺序、token 序列、adapter 权重指纹和数值校验。原始 Qwen 与专用 Embedding 的向量可以在同次运行的多个 seed 间复用，MLP 仍按各自 seed 独立训练。

输出位于 `runs/kai/qwen_embedding_mlp_runs/<run_id>/`，并导出 `qwen3_embedding_mlp.zip`：

| 文件 | 用途 |
|---|---|
| `comparison.csv` | 五组测试 accuracy、macro-F1，以及相对原始 Qwen＋MLP 的差值 |
| `comparison_aggregate.csv` | 多 seed 均值和标准差；单 seed 的标准差留空 |
| `paired_bootstrap.csv` | 标签适配减原始、解释减标签、反事实减解释、专用 Embedding 减原始、完整适配减专用 Embedding 的配对区间 |
| `per_class_metrics.csv` | 四类 precision、recall、F1，检查正负类收益是否伴随 Neutral/Irrelevant 下降 |
| `mlp/<组名>_seed*/` | 独立 MLP 与 scaler、选型日志、验证/测试指标、逐条预测和混淆矩阵 |
| `embeddings/*.npz` | 带行 ID、截断标记和配置指纹的向量缓存 |
| `adaptation_coverage.csv` | 三组实际使用的原始样本、解释和反事实数量 |
| `experiment_protocol.json` | 固定划分指纹、模型 revision、提示、损失权重和全部实验设置 |

五组 MLP 的正式结果需要在 5090 上运行后产生，不沿用旧生成式分类成绩。配对 bootstrap 只衡量固定 seed 下的测试样本不确定性，不能替代多 seed 重复实验。

方法依据：[SetFit](https://arxiv.org/abs/2209.11055) 支持先适配表示再训练分类器；[LLM2Vec](https://arxiv.org/abs/2404.05961) 支持 decoder 模型用于文本表示，但本实现不复现其双向注意力和对比学习；[Qwen3-Embedding 官方模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) 给出专用嵌入模型的提取方式；[Distilling Step-by-Step](https://aclanthology.org/2023.findings-acl.507/) 和[反事实增强研究](https://arxiv.org/abs/1909.12434) 分别支持两类辅助监督。这里检验它们能否改善冻结表示，不预先声称提升。

## 数据与评估

原始训练池有 596 条推文，处理重复文本和跨集重叠后保留 591 条。神经模型使用保留大小写、标点和表情的轻度清洗文本，默认划分为 **473 条训练、118 条验证、397 条测试**。比较方法时使用相同划分，按验证集 macro-F1 选取模型，在测试集报告 macro-F1、准确率和各类别指标。[原始数据](../../Data%205%20-%20Twitter/) · [处理后数据](../../data/processed/) · [清理记录](../../reports/数据清理与分析.md)

Qwen 的测试结果由 Colab 运行后导出。要与小组的线性基线比较，可将原始标签微调导出的 ZIP 提供给[共享基线 notebook](../../notebooks/TFIDF_LR_Baseline.ipynb)，按同一训练划分和测试样本比较。

## 对照实验与结果分析

| 问题 | 如何比较与解读 |
|---|---|
| 原始标签微调能否超过共享基线？ | 在相同的 473 条训练、397 条测试推文上，比较原始标签 LoRA 与共享基线的 macro-F1、准确率和各类召回率。该划分的基线 macro-F1 为 **0.4461**，准确率为 **0.4912**。 |
| 教师短解释是否有帮助？筛选起了什么作用？ | 在蒸馏模式内比较同一随机种子的三组结果。格式有效解释组与原始标签组之差衡量加入解释后的变化。筛选组与格式有效组之差衡量采用筛选规则后的整体变化。结合 `teacher_filter_summary.csv` 的覆盖率解读后一个差值，因为筛选会同时改变解释的内容和数量。 |
| 情感翻转样本能否在解释蒸馏之上带来增益？ | 两组共用原始标签、筛选解释缓存和训练划分，增强组额外学习通过核验的正负向改写。查看 `comparison.csv` 中相对仅解释组的 `delta_macro_f1`，并用 `cf_audit.csv` 统计两种翻转方向的接受数。分别检查四类 F1，以判断正负类变化是否伴随 Neutral 或 Irrelevant 的性能变化。 |

**已验证的结果。** 本地用缩小的 Qwen3 在 CPU 上跑通了三种模式。原始标签流程检查覆盖数据划分、LoRA 参数更新和 adapter 重载。蒸馏与反事实流程用模拟教师或缓存检查筛选、对照训练和结果导出。[原始标签检查记录](../../reports/qwen_lora_notebook_validation.json) · [蒸馏检查记录](../../reports/qwen_distillation_notebook_validation.json) · [反事实检查脚本](counterfactual/validate.py)

**模型选择。** 原有生成式训练模式的学生训练臂统一：`eval/save` 按 epoch、`load_best_model_at_end`、`metric_for_best_model=macro_f1`、早停 patience=3；只在验证集选型，测试集只评估一次。

**正式测试结果。** 固定划分上的独立原始标签 LoRA 测试 macro-F1 为 0.4850，共享 TF-IDF 基线为 0.4461。蒸馏实验中，格式有效解释组相对该次原始标签对照提高 0.0141。反事实实验中，加入 46 条核验通过的改写相对该次仅解释对照提高 0.0017。两项差值的 95% 配对区间都跨过零。历史单次实验为 seed 42；多种子报告见 `comparison_aggregate.csv`（mean±std、最佳单 seed）与 `mean_delta_bootstrap.csv`。[三组运行摘要](../../runs/kai/)

本地多种子示例：

```bash
python models/kai/run_local.py --mode rationale-distillation --seeds 42,43,44 --rebuild
python models/kai/run_local.py --mode counterfactual --seeds 42,43,44 --distillation-export path/to/qwen3_rationale_distillation.zip
```

## 方法汇报


## 修改和构建

三个 Qwen 模型目录中的 `notebook.py` 提供对应实验单元，`validate.py` 用于小模型流程检查。解释蒸馏复用原始标签微调的分类逻辑，反事实实验复用解释蒸馏的筛选和训练逻辑。

`embedding-mlp/notebook.py` 组合五组实验；`embedding-mlp/pipeline.py` 实现冻结向量、缓存检查和独立 MLP；构建时全部内嵌到同一个 notebook，无需在 Colab 额外上传辅助代码。`embedding-mlp/validate.py` 用随机初始化的小型 Qwen3 和模拟教师缓存检查完整流程，不产生可用于报告的性能结论。

在 `group project` 目录运行以下命令，将这些单元写入 `notebooks/kai/Kai_Sentiment_Experiments.ipynb`。提交修改时一并提交更新后的 notebook。

```bash
python models/kai/build_notebook.py
```
