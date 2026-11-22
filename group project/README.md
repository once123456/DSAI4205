# Twitter 情感分类小组项目

本目录存放英文推文四分类的数据、代码、notebook 和分析报告。完整项目说明、课程交付要求、固定数据划分和共享基线结果见[仓库首页](../README.md)。

## 共享 baseline

TF-IDF + 逻辑回归在 397 条测试推文上取得的 macro-F1 为 0.4461（473 条训练子集）和 0.4662（591 条完整训练池）。在 Colab 的 CPU 运行时运行共享 notebook 即可复现。[模型说明](models/tfidf-lr/) · [基线 notebook](notebooks/TFIDF_LR_Baseline.ipynb) · [完整训练池指标](reports/baseline_metrics.json)

## 项目文件

| 内容 | 位置 |
|---|---|

| 处理后的数据与固定划分 | [data/processed/](data/processed/) |
| 共享清理和分析脚本 | [scripts/](scripts/) |
| 分析报告与图表 | [reports/](reports/) · [figures/](figures/) |
| Kai | [models/kai/](models/kai/) · [notebooks/kai/Kai_Sentiment_Experiments.ipynb](notebooks/kai/Kai_Sentiment_Experiments.ipynb) |
| Huakun Huang | [models/kun/](models/kun/) | [Notebook](notebooks/kun/Huakun_Huang_ELMo_ABCD_Report.ipynb) |
| Jingwen Yang | [models/jingwen/](models/jingwen/) | [Notebook](notebooks/jingwen/Jingwen_MultiView_SVM.ipynb) |
| Yuchen Gu | [models/yuchen/](models/yuchen/) · [Notebook](notebooks/yuchen/GloVe_BiLSTM.ipynb) |

课程数据由组员从课程平台取得。在 group project 目录下将 traindata.csv 和 testdata.csv 放入 Data 5 - Twitter，再运行 python scripts/clean_and_analyze.py。也可以直接将组内授权的 train_cleaned.csv 和 test_cleaned.csv 放入 data/processed，配合仓库中的 split_manifest.csv 使用。在 Colab 单独打开 Kai 或共享基线 Notebook 时，将 DSAI4205_DATA_DIR 环境变量设为这些 CSV 所在目录。Kun 的归档结果展示可离线运行。
