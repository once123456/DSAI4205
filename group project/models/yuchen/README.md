# Yuchen Gu · GloVe-Twitter + BiLSTM

[Open the notebook](../../notebooks/yuchen/GloVe_BiLSTM.ipynb) or [browse the archived experiment](../../runs/yuchen/glove_bilstm/). Both include saved results that can be read offline. The notebook contains the full model implementation.

## Reproduction

1. Follow the [shared data instructions](../../README.md) to obtain train_cleaned.csv and test_cleaned.csv. Put them alongside the repository’s split_manifest.csv in data/processed. When running outside the checkout, set the DSAI4205_DATA_DIR environment variable to the folder containing all three files. The loader checks the original column order, row order and content hashes.
2. Open the notebook with a Python kernel containing torch, numpy, pandas, scikit-learn, matplotlib, IPython and gensim. The recorded run used PyTorch 2.11.0+cu128. The first training run downloads glove-twitter-50 through Gensim unless the vectors are already in its cache.
3. To train, restart the kernel and run all cells. The notebook uses CUDA when available and otherwise falls back to CPU. Each run writes its settings, logs, predictions and checkpoints under bilstm_outputs/ in the kernel working directory. This directory is ignored by Git.

## Experiment

A one-layer BiLSTM reads 50-dimensional pretrained Twitter word vectors. Its forward and backward states have 48 units each. Masked mean pooling combines the token states, followed by a 96-unit ReLU layer, dropout of 0.4 and four output logits. The embedding table is fine-tuned from the first epoch. Training uses class-weighted cross-entropy, AdamW, a batch size of 32 and seed 42.

Selection used 473 fitting rows and 118 validation rows. Epoch 12 achieved the highest validation macro-F1 of 0.4962. The vocabulary and model were then rebuilt on all 591 rows and trained for 12 epochs. The archived configuration used zero extra epochs.

| Evaluation | Rows | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| BiLSTM validation | 118 | 0.5169 | 0.4962 | 0.5007 |
| BiLSTM test | 397 | 0.4610 | 0.4371 | 0.4606 |

The shared TF-IDF + LR baseline fitted on 591 rows achieved test macro-F1 of 0.4662. The BiLSTM’s lowest class F1 was 0.2336 for Irrelevant. Its most frequent misclassification was Negative as Positive (26 rows). Development included repeated inspection of these 397 test rows. Small differences across the single-seed experiments therefore warrant caution.

## Reading the archive

The [records](../../runs/yuchen/glove_bilstm/) include 20 selection epochs and 12 refit epochs, validation and test metrics, confusion matrices, and six saved errors identified by row ID. The four validation tuning trials and nine historical configurations come from a separate source table. Those configurations also vary in training duration, vocabulary construction or fine-tuning schedule. The notebook documents those differences. The complete 397-row prediction file was not supplied.

[Record provenance and original cell numbers](../../runs/yuchen/glove_bilstm/provenance.json) · [Original notebook contributed by Yuchen](https://github.com/once123456/DSAI4205/blob/22eb62943cce41a5041660db7d0a117b7a393e3d/group%20project/notebooks/yuchen/GloVe_BiLSTM.ipynb)
