# Frozen BERT Embeddings with an Independent MLP Classifier

**Contributor:** Phelan (庞智瀚)  
**Project:** DSAI4205 Big Data Analytics, Twitter sentiment classification  
**Purpose:** Team reference and report-ready description of the contributor's model  
**Evidence:** Run `20261115T060046Z`; the comparison uses the shared 473-row training split and 397-row test set.

> **Latest study, 21 November.** We compared 11 configurations. The locked challenger used base BERT with text alone. On the shared 397-message test set it scored 0.4686 macro-F1 and 0.4786 accuracy, against 0.4971 and 0.5113 for the previous class-weight model. No finalist met all three validation thresholds. The class-weight configuration (v4) remains the selected submission model and the challenger is an exploratory result. On official validation the challenger achieved macro-F1 0.4679, Irrelevant F1 0.3200 and Neutral F1 0.4651. On the project test set its Irrelevant F1 was 0.3273 and Neutral F1 0.4914. The macro-F1 difference was +0.0225 versus the original baseline and +0.0318 versus the tuned baseline. The challenger-minus-v4 paired 95% interval was [-0.0512, -0.0065]. It describes test-row variability for fixed predictions on this previously used set. [Full study](../../reports/phelan_overleaf/transfer_study.tex) · [Saved comparisons](../../runs/phelan/20261121_transfer_study/)

> **Earlier class-weight study, 21 November.** An exploratory class-weight candidate produced test macro-F1 **0.4971**, accuracy **0.5113**, Irrelevant F1 **0.3399**, and Neutral F1 **0.5246**. The candidate was selected on validation after introducing a post-hoc 0.005 tolerance in macro-F1; the project test set had already been examined in prior work. A 5,000-sample paired bootstrap interval for candidate-minus-original macro-F1 is **[−0.0130, 0.0383]** and does not establish a reliable gain. Neutral recall falls from 0.5048 to 0.4571. The original model remains a valid reference; do not describe the candidate as a statistically confirmed improvement. The complete iteration history and updated English report are in [`reports/phelan_overleaf/`](../../reports/phelan_overleaf/), with code in [`BERT_MLP_Refinement.ipynb`](../../notebooks/phelan/BERT_MLP_Refinement.ipynb) and recorded results in [`results.json`](../../runs/phelan/20261121_refinement_v4/results.json). The original-run sections below have been retained for traceability.

## 1. Summary for the team

This contribution combines a frozen, pretrained BERT encoder with a separately trained multilayer perceptron (MLP) for four-class tweet classification. BERT supplies contextual text representations; the MLP is the independent analytics classifier. Four standardized surface features are added to the BERT representation. The classifier is trained on 473 examples, its configuration is selected using 118 validation examples, and its performance is reported on the 397-example test set.

The model achieves **test macro-F1 of 0.4843** and **accuracy of 0.5038**. The shared TF-IDF plus logistic regression baseline, fitted on the same 473 training rows, achieves macro-F1 of **0.4461** and accuracy of **0.4912**. The observed differences are **+3.82 percentage points in macro-F1** and **+1.26 percentage points in accuracy**. These are descriptive results from a single recorded model run; they do not by themselves establish statistical significance.

The structure directly implements the assignment's embedding-plus-separate-classifier approach: BERT is used as the embedding model, while the MLP is trained separately for classification. BERT is pretrained, not trained from scratch for this project; the MLP is trained on the project data. This is one model contribution, not the entire set of models required from the group.

## 2. Task and evaluation protocol

The classifier predicts one of four labels for each message:

| Label ID | Class |
|---:|---|
| 0 | Negative |
| 1 | Positive |
| 2 | Irrelevant |
| 3 | Neutral |

The project's cleaned training pool contains 591 rows. The fixed [`split_manifest.csv`](../../data/processed/split_manifest.csv) assigns **473** to training and **118** to validation; the original test set contains **397** rows. Both the BERT model and the main comparison baseline are fitted using the 473-row training partition. The validation partition is used for early stopping and model selection. Test metrics are calculated for the selected model.

The source files have no separate target-entity column. Although the task is described as entity-level sentiment, the model receives the message text rather than an explicitly supplied target entity. Predictions should therefore not be interpreted as reliable attribution to a particular brand in a multi-entity message.

**Comparison rule:** A different baseline run trained on all 591 training-pool rows reports macro-F1 **0.4662**. It uses more labelled data, so it should be identified as a separate training-size reference, not substituted for the **0.4461** same-training-subset baseline in the main comparison.

## 3. Method

### 3.1 Text representation

The model uses `google-bert/bert-base-uncased`, with the encoder frozen during classifier training. The recorded run resolved the model to revision `86b5e0934494bd15c9632b12f734a8a67f723594`.

Input is the original `Phrase` after light encoding repair with `ftfy`, Unicode NFC normalization, and whitespace normalization. Unlike the TF-IDF baseline's cleaned text, this path does not lemmatize or remove punctuation from the input. The selected BERT checkpoint is **uncased**, so its tokenizer applies its own uncased processing even though the light normalization step does not explicitly lowercase the source text. Tokenization truncates sequences to a maximum of 128 tokens.

The encoder produces a 768-dimensional hidden representation for each token. Attention-mask-aware mean pooling averages the representations of non-padding tokens, including tokenizer-added special tokens, into one 768-dimensional vector per message. BERT is run in evaluation/inference mode, its parameters have gradients disabled, and its weights do not change while the MLP is trained.

### 3.2 Additional features and classifier

Four counts from the processed data are concatenated with the BERT vector:

1. `exclaim_count`
2. `question_count`
3. `url_count`
4. `mention_count`

A `StandardScaler` is fitted on the training partition only and then applied to the validation and test partitions. The resulting classifier input has **772 dimensions**: 768 BERT features plus 4 standardized counts. The model's surface-feature design is motivated by exploratory analysis, but no with-versus-without feature ablation has been recorded; their individual contribution is therefore unknown.

The independent MLP selected in the recorded run has the architecture **772 → 512 → 256 → 4**. Its hidden layers use ReLU and dropout; the output layer produces four logits. Cross-entropy is applied to logits during training, and softmax is used to produce prediction probabilities at evaluation. The training code uses Adam with learning rate `0.001` and weight decay `0.0001`, balanced class weights in the loss, a maximum of 300 epochs, and early stopping after 15 epochs without validation macro-F1 improvement.

## 4. Validation and refinement

The recorded search evaluated eight combinations: two hidden-layer configurations, two dropout rates, one learning rate, and the presence or absence of balanced class weights. All configurations used random seed 42. The following values are **validation** macro-F1 scores, not test scores.

| Hidden layers | Dropout | Learning rate | Balanced weights | Validation macro-F1 |
|---|---:|---:|---|---:|
| 512, 256 | 0.5 | 0.001 | Yes | **0.5140** |
| 512, 256 | 0.3 | 0.001 | Yes | 0.5093 |
| 256, 128 | 0.5 | 0.001 | Yes | 0.5030 |
| 256, 128 | 0.3 | 0.001 | Yes | 0.4939 |
| 512, 256 | 0.3 | 0.001 | No | 0.4903 |
| 512, 256 | 0.5 | 0.001 | No | 0.4767 |
| 256, 128 | 0.5 | 0.001 | No | 0.4326 |
| 256, 128 | 0.3 | 0.001 | No | 0.4276 |

For the `512, 256`, dropout `0.5` setting, the weighted configuration scored **0.5140** versus **0.4767** without weights on the validation set. This is a useful single-split comparison, but it is not a statistically established effect and does not isolate the effect of class weighting on test-set minority-class recall. The original 473-row TF-IDF baseline uses no class weights; its configuration differs from the BERT model in more than one respect.

## 5. Test results

### 5.1 Overall comparison

| Model | Training rows | Test rows | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| TF-IDF (unigrams and bigrams) + logistic regression | 473 | 397 | 0.4912 | 0.4461 |
| Frozen BERT + surface features + independent MLP | 473 | 397 | **0.5038** | **0.4843** |
| Observed difference, BERT minus baseline | — | — | +0.0126 | +0.0382 |

The BERT model's test macro-precision is **0.4877** and macro-recall is **0.4840**. Macro-F1 is the arithmetic mean of the four class F1 scores, so each class has equal weight regardless of its support.

### 5.2 Per-class BERT results

| Class | Precision | Recall | F1 | Test support |
|---|---:|---:|---:|---:|
| Negative | 0.5692 | 0.6167 | 0.5920 | 120 |
| Positive | 0.5604 | 0.4811 | 0.5178 | 106 |
| Irrelevant | 0.2857 | 0.3333 | 0.3077 | 66 |
| Neutral | 0.5354 | 0.5048 | 0.5196 | 105 |
| **Macro average** | **0.4877** | **0.4840** | **0.4843** | **397** |

Irrelevant is the weakest class by F1. The same-subset baseline recognizes **10 of 66** Irrelevant messages (recall **0.1515**) and assigns the Irrelevant label 12 times (precision **0.8333**). BERT recognizes **22 of 66** (recall **0.3333**) but assigns that label 77 times (precision **0.2857**). Thus, it detects more Irrelevant cases while also producing more false-positive Irrelevant predictions. These models differ in representation, classifier, input features, and weighting, so the comparison does not identify which design choice caused the change.

### 5.3 Confusion matrix

Rows are the true labels; columns are the predicted labels. The entries below were independently recomputed from the saved 397-row prediction file.

| True / predicted | Negative | Positive | Irrelevant | Neutral |
|---|---:|---:|---:|---:|
| Negative | 74 | 15 | 21 | 10 |
| Positive | 16 | 51 | 17 | 22 |
| Irrelevant | 15 | 15 | 22 | 14 |
| Neutral | 25 | 10 | 17 | 53 |

The corresponding saved figure is [BERT confusion matrix](../../runs/phelan/20261115T060046Z/confusion_matrix.png). An example of an operational consequence is that 25 Neutral messages were predicted as Negative, which could inflate a complaint-review queue. This is a result on the present sample, not an estimate of production performance.

### 5.4 Sensitivity to the historical overlap row

One message originally appeared in both source partitions with conflicting labels. Its training copy was removed during cleaning; the retained test row was flagged. Excluding that flagged test row leaves **396** examples and gives macro-F1 **0.4856** and accuracy **0.5051**. The primary results above retain all **397** test rows. This sensitivity check concerns the disputed retained row; the training copy had already been removed before model fitting.

## 6. Text ready to adapt for the final report

> We combined a frozen pretrained `bert-base-uncased` encoder with a separately trained multilayer perceptron for four-class Twitter sentiment classification. After light text normalization, attention-mask-aware mean pooling converted BERT token representations into a 768-dimensional message vector. Four surface counts—exclamation marks, question marks, URLs, and mentions—were standardized using the training partition and concatenated with the BERT vector. The selected classifier used two hidden layers of 512 and 256 units, dropout of 0.5, and balanced class weights. Hyperparameters were selected by validation macro-F1 on a fixed 473/118 training–validation split.
>
> On 397 test messages, the BERT–MLP model achieved macro-F1 of 0.4843 and accuracy of 0.5038, compared with 0.4461 and 0.4912 for a TF-IDF–logistic regression baseline fitted on the same 473 training examples. The observed macro-F1 difference was 3.82 percentage points. Irrelevant-class recall increased from 0.1515 to 0.3333, while precision decreased from 0.8333 to 0.2857. These results describe one recorded run on a small test set; the difference should not be presented as statistically significant without further analysis.

## 7. Reproduction instructions

### Option A: standalone Colab notebook

1. Open [`BERT_MLP_Sentiment.ipynb`](../../notebooks/phelan/BERT_MLP_Sentiment.ipynb) in Google Colab and select a GPU runtime such as T4. CPU execution is possible in principle but will be slower.
2. Execute cells in order, including dependency installation. The notebook embeds snapshots of the relevant processed data and split manifest with SHA-256 checksums. The first model load requires network access to the public model files.
3. Keep `SMOKE_TEST = False` to reproduce the full eight-configuration search. A smoke test runs a reduced configuration and must not be reported as the official result.
4. The notebook exports a ZIP containing the configuration, tuning results, metrics, confusion matrix, and per-row predictions. Its embedded data snapshot can be used without separately uploading the project CSVs.

### Option B: local runner

From the `group project` directory, after installing the dependencies listed in the notebook and setting up a compatible PyTorch environment:

```bash
python models/phelan/run_local.py
```

The runner skips the notebook's installation cell and writes a new timestamped directory under `runs/phelan/`. It defaults `HF_ENDPOINT` to `https://hf-mirror.com` when that environment variable is not already set; availability depends on the network environment. `python models/phelan/run_local.py --smoke` checks the pipeline with a reduced search and does not reproduce the reported score. The local runner builds its cells from the current generator and local processed data, so the project data and manifest must be present.

For the recorded run, the local summary reports an RTX 4070 Laptop GPU and `torch 2.6.0+cu124`. The notebook declares `transformers==4.57.1`, `numpy==2.0.2`, `pandas==2.2.3`, `scikit-learn==1.6.1`, `matplotlib==3.10.6`, and `ftfy==6.3.1`; the saved run configuration does not independently record every installed package version.

**Version and checkpoint limitations:** The notebook currently sets `MODEL_REVISION = "main"`, resolves it through the model hub at runtime, and records the resolved revision. To target the recorded model, set `MODEL_REVISION` to the full SHA listed in Section 3 before rerunning. Resolution currently calls the model hub, so a cached model alone does not guarantee offline execution. The exported ZIP does **not** contain the trained MLP weights or fitted scaler: to predict new messages, rerun training or extend the implementation to persist and reload these objects. GPU and library differences may also prevent exact numerical reproduction.

## 8. Limitations and team integration

- The available recorded result uses a single seed and a small test set. Repeat runs and uncertainty estimates would make comparisons more robust.
- The group has already inspected the test set during several experiments. A fresh external holdout would provide a stronger final estimate after all model and feature decisions are fixed.
- The four surface features were not separately ablated. Their usefulness is a hypothesis, not a measured causal contribution.
- Class weighting and model selection were assessed on one validation partition; the validation comparison alone does not prove a robust test improvement.
- The missing target-entity field and potential label ambiguity limit interpretation of entity-specific sentiment and Irrelevant predictions.
- The existing model notebook is a **standalone contribution**. The course also requires one integrated, well-annotated notebook containing the group's complete source code and preprocessing; that integration is still pending.
- Final comparisons should add other members' actual results using their verified splits, training sizes and per-row predictions. Do not infer those results or claim the best overall model before they are available.

## 9. Evidence and files

| Item | File |
|---|---|
| Standalone notebook | [`notebooks/phelan/BERT_MLP_Sentiment.ipynb`](../../notebooks/phelan/BERT_MLP_Sentiment.ipynb) |
| Notebook source and builder | [`notebook.py`](notebook.py), [`build_notebook.py`](build_notebook.py) |
| Local execution | [`run_local.py`](run_local.py) |
| Training/validation split | [`data/processed/split_manifest.csv`](../../data/processed/split_manifest.csv) |
| Run configuration | [`run_config.json`](../../runs/phelan/20261115T060046Z/run_config.json) |
| Validation grid | [`tuning.json`](../../runs/phelan/20261115T060046Z/tuning.json) |
| Test metrics and individual predictions | [`test_metrics.json`](../../runs/phelan/20261115T060046Z/test_metrics.json), [`test_predictions.csv`](../../runs/phelan/20261115T060046Z/test_predictions.csv) |
| Confusion matrix | [`confusion_matrix.png`](../../runs/phelan/20261115T060046Z/confusion_matrix.png) |
| Baseline configuration | [`models/tfidf-lr/experiment.py`](../tfidf-lr/experiment.py) |

The baseline class metrics in Section 5.2 were recomputed from the processed data, fixed split, and documented baseline configuration using scikit-learn 1.6.1. The BERT metrics and confusion matrix were independently checked against the saved prediction CSV. The BERT model was **not retrained** for this documentation review.
