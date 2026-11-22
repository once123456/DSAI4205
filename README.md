# DSAI4205 · Twitter 情感分类小组项目

本项目为 DSAI4205 Big Data Analytics 课程开发英文推文分类系统。输入是一条推文，输出是 Negative（0）、Positive（1）、Irrelevant（2）或 Neutral（3）。原始文件提供推文正文和人工标签，涉及的品牌或账号出现在正文中。小组使用同一份测试集比较模型，分析哪些类别容易混淆，以及分类结果能为解读用户反馈提供什么依据。

## 项目范围与交付

| 阶段 | 小组任务 |
|---|---|
| 数据探索与清理 | 检查类别分布、缺失值、异常长度、乱码和重复文本，记录处理前后的差异。用至少 N 项数据统计及相应图表呈现发现，N 为组员人数。 |
| 建立共享基线 | 在清洗后的推文上训练 TF-IDF + 逻辑回归，报告准确率、精确率、召回率和 F1。 |
| 进阶建模与改进 | 比较 N 种不同的分析模型。每种方法将词嵌入模型与独立的分析分类器组合，并说明如何连接两者。课程将 TF-IDF 限用于基线，逻辑回归不计入 N。改进基线后再次比较，解释性能差异。 |
| 汇报与提交 | 期中报告汇总数据探索、清理和基线结果。期末交付一份整合预处理和全部模型的带注释 Jupyter notebook、一份汇总结果与业务解读的 Word 报告，以及用于 12 分钟展示的幻灯片。 |

## 作业提交版 Notebook

| 文件 | 内容与使用方式 |
|---|---|
| [全组代码 Notebook](submission/01_Group_Implementation.ipynb) | 汇集数据清洗、探索分析、共享基线和五位成员的模型实现。默认执行可检查代码与运行入口；复现实验时，在开头选择对应模型并配置数据路径。 |
| [全组实验日志 Notebook](submission/02_Experiment_Log.ipynb) | 汇总实验设置、指标、图表和复跑记录。打开即可查看已保存的结果；训练集交叉验证与测试集结果分别标注。 |

各成员继续在 [原始 Notebook 目录](group%20project/notebooks/) 维护自己的实现。课程要求的 Word 正式报告和展示幻灯片另行提交。

## 数据与评估

原始数据含 596 条训练推文和 397 条测试推文。训练池中有一句重复五次，按多数标签保留一条。另有一句同时出现在训练集和测试集，从训练池移除后保留带标记的测试行。清理后训练池有 **591 条**，测试集有 **397 条**。[数据清理记录](group%20project/reports/数据清理与分析.md)

固定划分从训练池中取 **473 条训练、118 条验证**，相同的标准化文本归入同一子集。需要选参数的模型使用验证集，最终在同一批 397 条推文上评估。主要比较 macro-F1，使四个类别的 F1 各占相同权重，同时报告准确率和逐类指标。[划分清单](group%20project/data/processed/split_manifest.csv) · [探索性分析报告](group%20project/reports/Twitter情感数据技术分析报告.md) · [图表](group%20project/figures/)

## 共享 baseline · TF-IDF + 逻辑回归

基线用清洗后的 `text_clean` 提取单词和相邻词对的 TF-IDF 特征，仅在各自的训练样本上拟合词表，再用逻辑回归预测类别。下表两行均在同一批 397 条测试推文上计算。

| 拟合数据 | 训练条数 | 测试准确率 | 测试 macro-F1 |
|---|---:|---:|---:|
| 固定训练子集 | 473 | 0.4912 | 0.4461 |
| 完整训练池 | 591 | 0.5063 | 0.4662 |

比较使用固定训练划分的方法时，参照第一行。完整训练池模型对 Negative 的召回率为 0.7417，对 Irrelevant 的召回率为 0.1970，即 66 条 Irrelevant 推文中仅识别出 13 条。[逐类测试报告](group%20project/reports/baseline_classification_report.txt)

在 Colab 选择 CPU 运行时，点击「全部运行」即可复现两行结果，并导出每条推文的预测。[打开共享 baseline notebook](https://colab.research.google.com/github/once123456/DSAI4205/blob/main/group%20project/notebooks/TFIDF_LR_Baseline.ipynb) · [基线代码与运行说明](group%20project/models/tfidf-lr/)

## 成员文件位置

| 成员 | 模型目录与 README | Notebook |
|---|---|---|
| Kai | [models/kai/](group%20project/models/kai/) | [notebooks/kai/Kai_Sentiment_Experiments.ipynb](group%20project/notebooks/kai/Kai_Sentiment_Experiments.ipynb) |
| Huakun Huang | [models/kun/](group%20project/models/kun/) | [Notebook](group%20project/notebooks/kun/Huakun_Huang_ELMo_ABCD_Report.ipynb) |
| Jingwen Yang | [models/jingwen/](group%20project/models/jingwen/) | [Notebook](group%20project/notebooks/jingwen/Jingwen_MultiView_SVM.ipynb) |
| Yuchen Gu | [models/yuchen/](group%20project/models/yuchen/) | [Notebook](group%20project/notebooks/yuchen/GloVe_BiLSTM.ipynb) |
| 庞智瀚（Phelan） | [models/phelan/](group%20project/models/phelan/) | [Notebook](group%20project/notebooks/phelan/BERT_MLP_Refinement.ipynb) |

课程数据由组员从课程平台取得。在 group project 目录下将 traindata.csv 和 testdata.csv 放入 Data 5 - Twitter，再运行 python scripts/clean_and_analyze.py。也可以直接将组内授权的 train_cleaned.csv 和 test_cleaned.csv 放入 data/processed，配合仓库中的 split_manifest.csv 使用。在 Colab 单独打开 Kai 或共享基线 Notebook 时，将 DSAI4205_DATA_DIR 环境变量设为这些 CSV 所在目录。Kun 的归档结果展示可离线运行。
