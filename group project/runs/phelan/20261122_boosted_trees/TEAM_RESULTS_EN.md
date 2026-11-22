# Frozen BERT with boosted-tree classifiers

XGBoost reached test macro-F1 0.5217, a change of +0.0247 from the frozen-BERT MLP. LightGBM scored 0.5076. The pre-specified 0.4971 threshold selects XGBoost.

We replaced the independent MLP with gradient-boosted decision trees. Both libraries received identical cached 772-dimensional inputs from the pinned frozen BERT encoder (768 mean-pooled values plus four training-standardized counts). We reused the feature producer's row order and checked the last four columns against the shared files for all three partitions. The 473 original training rows alone supplied the class frequencies. We multiplied inverse-frequency balanced weights by (1, 1, 1.25, 0.75) and passed the resulting per-row weights to both learners. Effective weights in Negative, Positive, Irrelevant, Neutral order were 0.8569, 0.8825, 1.8477, 0.7330.

Each family was evaluated on 432 configurations. For every combination of depth, learning rate, row and column subsampling, minimum child weight and L2 regularization, we assessed 100, 200 and 300 rounds from one boosting path. The maximum number of histogram bins was fixed at 32 and seed at 42. Selection maximized unweighted macro-F1 on the fixed 118-row validation set. Ties favored fewer rounds, then the declared grid order. Each winner was refitted on the same training rows and its validation score verified before test evaluation.

## Validation-selected settings

| Parameter | XGBoost | LightGBM |
| --- | --- | --- |
| max_depth | 3 | 5 |
| learning_rate | 0.2 | 0.1 |
| subsample | 1.0 | 0.8 |
| colsample_bytree | 1.0 | 0.8 |
| min_child_weight | 1 | 3 |
| reg_lambda | 10 | 1 |
| n_estimators | 100 | 300 |

## Shared test results

| Model | Validation macro-F1 | Accuracy | Macro P | Macro R | Macro-F1 | F1 change vs. v4 |
| --- | --- | --- | --- | --- | --- | --- |
| MLP v4 | 0.5118 | 0.5113 | 0.5059 | 0.4975 | 0.4971 | +0.0000 |
| XGBoost | 0.5295 | 0.5441 | 0.5313 | 0.5195 | 0.5217 | +0.0247 |
| LightGBM | 0.5393 | 0.5315 | 0.5142 | 0.5066 | 0.5076 | +0.0106 |

| Class | Support | MLP v4 F1 | XGBoost F1 | LightGBM F1 |
| --- | --- | --- | --- | --- |
| Negative | 120 | 0.5917 | 0.5856 | 0.5779 |
| Positive | 106 | 0.5321 | 0.5545 | 0.5607 |
| Irrelevant | 66 | 0.3399 | 0.3500 | 0.3390 |
| Neutral | 105 | 0.5246 | 0.5969 | 0.5528 |

The adoption rule uses the existing test set to choose between model families. The figures therefore describe this project sample. A fresh holdout would be needed to estimate the performance of the resulting choice. LightGBM serves as a comparator within the same tree family. The paired, 5,000-resample 95% interval for the XGBoost-minus-MLP test macro-F1 difference was [-0.0203, 0.0694], conditional on these trained models.

The recorded predictions and per-class scores are available in this directory. The serialized boosters are excluded. See the [model README](../../../models/phelan/README.md) for the local feature-cache producer and the notebook entry point.
