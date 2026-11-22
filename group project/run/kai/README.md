# Kai 实验记录

数据划分为训练 473 条、验证 118 条、测试 397 条。主方法 B_mean 与 A_mean 的测试宏 F1 分别为 0.4961 和 0.4838（五个种子的平均值）。配对差值为 +0.0123，bootstrap 区间为 [−0.0215, +0.0475]。测试集在项目过程中被多次查看。

| 实验 | 记录数 | 主要记录 |
|---|---:|---|
| [frozen_ae_20261121](frozen_ae_20261121/) | 13 | [首次冻结编码器对照](frozen_ae_20261121/comparison.csv) |
| [audit_20261121](audit_20261121/) | 173 | [A/E 与逻辑回归对照：五种子、校准、少样本学习曲线、延迟与成本](audit_20261121/tables/test_summary.csv) |
| [pooling_20261121](pooling_20261121/) | 77 | [模板末尾、评论末 token 和评论 mean pooling 对照](pooling_20261121/tables/test_summary.csv) |
| [b_mean_preflight_20261121](b_mean_preflight_20261121/) | 5 | [正式运行前的调试记录，单独列示，不纳入五种子汇总](b_mean_preflight_20261121/logs/lora_epochs.jsonl) |
| [b_mean_20261121_deterministic](b_mean_20261121_deterministic/) | 96 | [固定评论 mean pooling，用分类损失训练 LoRA，冻结后另训 MLP](b_mean_20261121_deterministic/tables/test_summary.csv) |
| [validation_20261122](validation_20261122/) | 10 | [5090 独立复跑、预测重放和未通过的检查](validation_20261122/kai_verification.json) |

[来源与 SHA-256 清单](record_manifest.csv)列出 374 份记录，包括逐轮日志、指标、预测与轻量结果图。预测表用行 ID 对齐。source_sha256 对应实例上的原始文件，archive_sha256 对应此处发布的文件；部分日志只保留相关字段，文本文件统一采用 LF 换行。

[Hugging Face 固定版本](https://huggingface.co/kingkaikai/DSAI4205-Qwen3-0.6B-B-mean/tree/e4b0be461caa04873b319fa4f5a44d664edf9d8b) 保存五个种子的 LoRA 适配器、分类头及 A/B 重放特征。数值训练历史也覆盖未通过重载检查的验证运行；完整状态和失败原因见验证目录。

发布核验：以匿名身份下载固定版本的 50 个文件并校验哈希。用归档分类头、特征和原版 NumPy / scikit-learn 在 CPU 上重放 A_mean 和 B_mean，五种子 × 三个划分 × 两种方法共 9,880 个预测标签及行 ID 全部匹配。这项检查复用了已提取的特征。

[早期标签微调、解释蒸馏与反事实实验摘要](../../runs/kai/) · [全组实验日志 Notebook](../../../submission/02_Experiment_Log.ipynb)
