# BERT（冻结嵌入）+ MLP 结果汇总

- Run: `runs/phelan/20261115T060046Z`
- 方法：`google-bert/bert-base-uncased`（冻结，mean-pooling 768 维，revision `86b5e093`）+ 独立 MLP（772 维输入 = 768 句向量 + 4 个标准化表层特征）
- 硬件：RTX 4070 Laptop，torch 2.6.0+cu124
- 验证选超参：`hidden=[512,256] dropout=0.5 lr=1e-3 class_weight=balanced`，验证 macro-F1 **0.5140**

## 测试集（397）

| 指标 | 本模型 | TF-IDF+LR 基线 | 差值 |
|---|---:|---:|---:|
| macro-F1 | **0.4843** | 0.4461 | +0.0382 |
| accuracy | **0.5038** | 0.4912 | +0.0126 |

### 各类别 F1

| 类别 | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| Negative | 0.569 | 0.617 | 0.592 | 120 |
| Positive | 0.560 | 0.481 | 0.518 | 106 |
| Irrelevant | 0.286 | 0.333 | 0.308 | 66 |
| Neutral | 0.535 | 0.505 | 0.520 | 105 |

## 敏感性

去掉 1 条历史跨集重叠样本后（396 条），macro-F1 **0.4856**、accuracy 0.5051（主指标仍用全部 397 条）。

## 结论

冻结 BERT 只做特征提取、独立 MLP 分类，在同划分测试集上超过共享基线约 **+3.8 个百分点** macro-F1。这是一组「词嵌入模型 + 独立分析分类器」的实现（BERT 不参与分类器训练）。Irrelevant 仍是四类里最弱的一类；与 473 条训练集的基线相比，召回从 **0.1515** 升至 **0.3333**，但 precision 从 **0.8333** 降至 **0.2857**。0.197 是使用 591 条训练数据的另一组基线召回率，不应混用。相同其余超参下，有类别权重的验证 macro-F1 为 0.5140，无类别权重为 0.4767；这是单次验证集对照，不能证明权重在测试集上的因果作用。四个表层特征未做独立消融，不能断言其带来的增益。

## 文件

- `test_metrics.json` / `tuning.json` / `run_config.json` / `confusion_matrix.png` / `test_predictions.csv`
- 复现：`python models/phelan/run_local.py`（`group project` 目录；需 torch+transformers，HF 走 `HF_ENDPOINT=https://hf-mirror.com`）
