# 庞智瀚（Phelan）· 冻结 BERT + XGBoost

**22 November model decision.** XGBoost is the adopted classifier (test macro-F1 0.5217, accuracy 0.5441). The MLP is the historical comparator. [Final method, parameters and metrics](../../runs/phelan/20261122_boosted_trees/TEAM_RESULTS_EN.md).


完整的选型记录、逐类结果与历史比较见 [TEAM_REPORT_EN.md](TEAM_REPORT_EN.md)。原始贡献来自 Phelan 的 [PR #1，314d846](https://github.com/once123456/DSAI4205/pull/1)。

## 最终模型与复现入口

冻结的 BERT 输出 768 维 masked mean 向量。拼接四个按训练集标准化的表层特征后，独立的 XGBoost 使用 772 维输入。XGBoost 和 LightGBM 各在 118 条验证数据上比较 432 个配置；随后用此前查看过的 397 条测试数据决定是否采用 XGBoost。该组测试 Macro-F1 为 0.5217，MLP v4 为 0.4971，配对差值的 95% 区间为 [−0.0203, 0.0694]。

- **查看归档**：直接打开[测试结果表](../../runs/phelan/20261122_boosted_trees/test_summary.csv)、[验证网格](../../runs/phelan/20261122_boosted_trees/validation_grid.csv)或[按 ID 保存的预测](../../runs/phelan/20261122_boosted_trees/test_predictions.csv)，无需训练或模型文件。
- **复现树模型**：在独立的实验副本中，将授权的 train_cleaned.csv、test_cleaned.csv 和仓库中的 split_manifest.csv 放在 data/processed 下。以 group project 为工作目录，运行已有的 seed_ensemble.py（命令为 `python models/phelan/seed_ensemble.py`）。该脚本也会训练历史 MLP 集成并写入固定运行目录。然后打开[精炼实验 Notebook](../../notebooks/phelan/BERT_MLP_Refinement.ipynb)，从 “Frozen BERT with XGBoost and LightGBM” 一节独立运行。脚本使用的固定 BERT revision、XGBoost 3.0.2、LightGBM 4.6.0 及特征缓存的历史指纹见代码和 protocol.json。
- **其他实验**：finetune_bert.py 是历史解冻对照。迁移实验的独立 tuned-baseline 预测导出未随 PR 提供；该文件缺失时，Notebook 使用[归档比较表](../../runs/phelan/20261121_transfer_study/test_predictions.csv)中同一批行的 tuned_baseline 列，按 ID 和真实标签对齐。翻译筛选记录只存行 ID、筛选结果或人工观察。
- **迭代依据**：11 组表示实验中，去掉表层特征使训练折外 Macro-F1 从 0.4494 升至 0.4672，但锁定的文本模型在测试上的 Macro-F1 为 0.4686，低于 MLP v4。121 条 Neutral 训练样本中，76 条回译通过筛选；作者在锁定模型后的检查记录中报告，它们与验证或测试文本没有规范化后的完全匹配。两次 v4 运行的计时分别保留，重复的逐条预测保留一份。

## 早期冻结 BERT + MLP 对照

以下说明对应早期 MLP Notebook。它与最终树模型共用四类标签。

| 标签 | 含义 |
|---|---|
| 0 | Negative |
| 1 | Positive |
| 2 | Irrelevant |
| 3 | Neutral |

## 设计要点

- **嵌入层**：`google-bert/bert-base-uncased`，冻结。输入 `Phrase` 原文轻度修复编码（保留大小写、标点、emoji、@ 账号），不做词形还原——BERT 是上下文敏感模型，lemmatize 反而丢信号。对 `last_hidden_state` 按 attention mask 做 mean-pooling（默认），也可切到 `[CLS]` 池化。
- **分类器**：独立 MLP（如 `768 → 512 → 256 → 4`，ReLU + Dropout + softmax）。因为嵌入一次性预计算并缓存，MLP 训练在几秒内完成，可廉价地做超参搜索。
- **特征增强（可选）**：把 EDA 发现的四个表层特征（`exclaim_count`、`question_count`、`url_count`、`mention_count`）标准化后拼到 MLP 输入，对应小组 Next Steps 的「加入表层特征」。默认开启。
- **类别不平衡**：Irrelevant 是少数类，本模型用 `class_weight="balanced"` 处理；原始 473 条训练集的 TF-IDF 基线没有使用类别权重。

## 评估协议

沿用小组固定划分（`data/processed/split_manifest.csv`）：**473 训练 / 118 验证 / 397 测试**。按验证集 macro-F1 选超参，测试集报告 macro-F1、准确率、逐类 P/R/F1、混淆矩阵，并额外输出去掉 1 条历史跨集重叠样本的敏感性结果。主比较对象是共享基线（同划分 macro-F1 **0.4461**）。

## 在 Colab 运行早期 MLP

1. 打开 [BERT_MLP_Sentiment.ipynb](../../notebooks/phelan/BERT_MLP_Sentiment.ipynb)（或上传到 [Google Colab](https://colab.research.google.com/)）。
2. 「运行时 → 更改运行时类型 → T4 GPU」。
3. 将 DSAI4205_DATA_DIR 环境变量设为上述三个 CSV 所在的目录。从安装依赖开始顺序运行，首次会下载公开的 BERT 权重，无需 Hugging Face token。
4. 最后下载 ZIP，内含 `tuning.json`（超参搜索）、`test_metrics.json`（测试指标）、`confusion_matrix.png` 和 `test_predictions.csv`。

`SMOKE_TEST=True` 时只跑一组最小超参，用于快速验证代码可运行，不作正式结果。

## 修改与构建

`notebook.py` 是 notebook 的构建脚本，生成读取本地授权 CSV 并校验 SHA-256 的单元。修改后，在 `group project` 目录运行：

```bash
python models/phelan/build_notebook.py
```

提交修改时一并提交更新后的 notebook。

## 文件

| 文件 | 内容 |
|---|---|
| `notebook.py` | 生成早期 MLP Notebook 的代码与数据指纹检查 |
| `build_notebook.py` | 把 `notebook.py` 写成 `notebooks/phelan/BERT_MLP_Sentiment.ipynb` |
| `README.md` | 本说明 |
