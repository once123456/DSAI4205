# 小组共享基线 · TF-IDF + 逻辑回归

基线使用清洗和词形还原后的推文 `text_clean`。TF-IDF 提取一元和二元词组，设置 `min_df=2`、`sublinear_tf=True`，逻辑回归使用 `C=1`。

## 在 Colab 运行

选择 CPU 运行时，点击「全部运行」。将 DSAI4205_DATA_DIR 指向两份清洗 CSV 和 split_manifest.csv 所在目录。Notebook 读取项目数据和固定划分，分别用 473 条训练子集和 591 条完整训练池拟合模型，在同一批 397 条测试推文上报告准确率、macro-F1、分类报告和逐条预测。[打开 Colab](https://colab.research.google.com/github/once123456/DSAI4205/blob/main/group%20project/notebooks/TFIDF_LR_Baseline.ipynb) · [查看 notebook](../../notebooks/TFIDF_LR_Baseline.ipynb) · [默认划分](../../data/processed/split_manifest.csv)

要与 Qwen 比较，将原始标签微调导出的 ZIP 上传到 Colab，并把 `QWEN_EXPORT` 设为文件路径。基线会读取 ZIP 中的训练划分，核对测试推文和标签，并把 Qwen 与两组基线的结果写入 `tfidf_comparison.csv`。比较相同训练规模时，使用 `same_training_subset` 行。

## 复现和修改

在 `group project` 下运行 `python models/tfidf-lr/baseline_tfidf_lr.py` 可重建完整训练池的报告与图表。已有报告中，591 条训练样本对应的测试准确率为 **0.5063**，macro-F1 为 **0.4662**。[训练脚本](baseline_tfidf_lr.py) · [基线报告](../../reports/baseline_metrics.json)

Colab 中的评估逻辑位于 [experiment.py](experiment.py)。修改后，在 `group project` 下运行 `python models/tfidf-lr/build_notebook.py` 更新共享 notebook。
