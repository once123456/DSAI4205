# Phelan（庞智瀚）· BERT（冻结嵌入）+ MLP 分类器

英文团队说明与报告可用内容：[Frozen BERT and MLP model documentation](TEAM_REPORT_EN.md)。

本模型是课程要求「**词嵌入模型 + 独立分析分类器**」的直接实现：预训练 `bert-base-uncased` **冻结权重**、只做特征提取，对每条推文做 mean-pooling 得到 768 维句向量；再训练一个**独立的 MLP** 做四分类。BERT 不参与分类器的梯度更新，两者严格分离。

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

## 在 Colab 运行

1. 打开 [BERT_MLP_Sentiment.ipynb](../../notebooks/phelan/BERT_MLP_Sentiment.ipynb)（或上传到 [Google Colab](https://colab.research.google.com/)）。
2. 「运行时 → 更改运行时类型 → T4 GPU」。
3. 从安装依赖开始顺序运行，首次会下载公开的 BERT 权重，无需 Hugging Face token。
4. 最后下载 ZIP，内含 `tuning.json`（超参搜索）、`test_metrics.json`（测试指标）、`confusion_matrix.png` 和 `test_predictions.csv`。

`SMOKE_TEST=True` 时只跑一组最小超参，用于快速验证代码可运行，不作正式结果。

## 修改与构建

`notebook.py` 是 notebook 的构建脚本，内嵌项目数据快照（base64+gzip+SHA-256）和固定划分。修改后，在 `group project` 目录运行：

```bash
python models/phelan/build_notebook.py
```

提交修改时一并提交更新后的 notebook。

## 文件

| 文件 | 内容 |
|---|---|
| `notebook.py` | 构建自包含 Colab notebook（数据快照 + 全部单元） |
| `build_notebook.py` | 把 `notebook.py` 写成 `notebooks/phelan/BERT_MLP_Sentiment.ipynb` |
| `README.md` | 本说明 |
