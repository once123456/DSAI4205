"""Build the frozen BERT + MLP notebook without reading course data."""
from __future__ import annotations

from textwrap import dedent

NOTEBOOK_NAME = "BERT_MLP_Sentiment.ipynb"
cells = []


def add(kind, source, name, **metadata):
    cell = dict(cell_type=kind, id=name, metadata=metadata,
                source=dedent(source).strip() + "\n")
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


def md(source, name):
    add("markdown", source, name)


def code(source, name, **metadata):
    add("code", source, name, **metadata)


# ---- cells ----
md('''
# BERT（冻结嵌入）+ MLP —— Twitter 四分类

**方法**：预训练 `bert-base-uncased` 只做特征提取（冻结权重），对每条推文做 mean-pooling 得到 768 维句向量；
再训练一个**独立的 MLP 分类器**。这是课程要求「词嵌入模型 + 独立分析分类器」的直接实现——BERT 不参与分类器训练。

**输入**：`Phrase` 原文轻度修复编码（保留大小写、标点、emoji、@ 账号），不做词形还原。
可选地把 EDA 发现的四个表层特征（感叹号、问号、网址、@ 数量）标准化后拼接到 MLP 输入。

**评估**：沿用小组固定划分（473 训练 / 118 验证 / 397 测试），按验证集 macro-F1 选超参，测试集报告 macro-F1、准确率、逐类指标和混淆矩阵。主比较对象是共享基线（同划分 macro-F1 **0.4461**）。

### Colab 使用方法
1. 将本文件上传到 [Google Colab](https://colab.research.google.com/)。
2. 「运行时 → 更改运行时类型 → T4 GPU」。
3. 从安装依赖单元开始按顺序运行。首次需联网下载公开模型，无需 Hugging Face token。
4. 最后下载 ZIP，内含超参搜索、测试指标、混淆矩阵和逐条预测。
''', "intro")

md('''
## 1. 安装依赖
固定主要框架版本。Colab 自带 CUDA PyTorch，无需重装。
''', "install-note")
code(r'''
%pip install -q "transformers==4.57.1" "numpy==2.0.2" "pandas==2.2.3" "scikit-learn==1.6.1" "matplotlib==3.10.6" "ftfy==6.3.1"
''', "install")

md('''
## 2. 配置与 GPU 检查
BERT 做嵌入提取需要 GPU（约 2GB 显存即可）；MLP 训练只需几十秒。
`SMOKE_TEST=True` 时只跑一组最小超参，用于验证代码可运行，不作正式结果。调整 `GRID` 可扩展超参搜索。
''', "config-note")
code(r'''
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv, hashlib, io, json, re, unicodedata, zipfile
from pathlib import Path
from datetime import datetime, timezone

import ftfy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import model_info

MODEL_ID = "google-bert/bert-base-uncased"
MODEL_REVISION = "main"          # 首次运行解析为不可变 commit SHA 并记录
SEED = 42
MAX_LENGTH = 128
EMBED_BATCH_SIZE = 64
POOLING = "mean"                 # "mean"（全 token 均值池化）或 "cls"
USE_SURFACE_FEATURES = True      # 是否拼接 EDA 表层特征
SURFACE_FEATURES = ["exclaim_count", "question_count", "url_count", "mention_count"]

# MLP 超参搜索网格（按验证集 macro-F1 选择）
GRID = {
    "hidden": [[512, 256], [256, 128]],
    "dropout": [0.3, 0.5],
    "lr": [1e-3],
    "class_weight": [True, False],
}

SMOKE_TEST = False
SAVE_TO_DRIVE = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
torch.manual_seed(SEED)
np.random.seed(SEED)

if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT_ROOT = Path("/content/drive/MyDrive/dsai4205_bert_mlp")
else:
    OUTPUT_ROOT = Path("/content/bert_mlp_runs")
RUN_DIR = OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR.mkdir(parents=True, exist_ok=False)
print("输出目录:", RUN_DIR)

if SMOKE_TEST:
    GRID = {"hidden": [[128]], "dropout": [0.3], "lr": [1e-3], "class_weight": [False]}
    print("SMOKE_TEST：仅跑一组最小超参，不作正式结果。")
''', "config")

md('''
## 3. 本地数据与固定划分
从项目的 data/processed 目录读取 train_cleaned.csv、test_cleaned.csv 和 split_manifest.csv。
在 Colab 设置 DSAI4205_DATA_DIR 指向这三个文件所在的目录。程序校验原实验使用的列和 SHA-256。
''', "data-note")
code(r'''
LABEL_NAMES = ["Negative", "Positive", "Irrelevant", "Neutral"]
LABEL_IDS = list(range(4))

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

DATA_MANIFEST = {'train': {'rows': 591, 'sha256': 'd4be1a7e6fa998aad57fed663465dc1c67d22e6c4c4aee42b3d700511247bc7e', 'source': 'data/processed/train_cleaned.csv'}, 'test': {'rows': 397, 'sha256': '75cccf17d4c48c7d8460f0e2fd52ff093c5527b7544a5f748acf559275e26e51', 'source': 'data/processed/test_cleaned.csv'}, 'split': {'rows': 591, 'sha256': 'e7fc1c5525dc8d9316e755c67b47dbe574115ef68959f0f33ece2e732f1f5bfb', 'source': 'data/processed/split_manifest.csv'}}

def data_folder():
    if os.environ.get("DSAI4205_DATA_DIR"):
        return Path(os.environ["DSAI4205_DATA_DIR"]).expanduser()
    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        for folder in (base / "data/processed", base / "group project/data/processed"):
            if (folder / "train_cleaned.csv").is_file():
                return folder
    raise FileNotFoundError("Set DSAI4205_DATA_DIR to the directory with the three course CSVs.")


def read_raw(name):
    record = DATA_MANIFEST[name]
    columns = (["row_id", "Sentiment", "group_id", "partition"] if name == "split" else
               ["row_id", "Phrase", "Sentiment", "cross_split_overlap", *SURFACE_FEATURES])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    with (data_folder() / Path(record["source"]).name).open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            writer.writerow({column: row[column] for column in columns})
    raw = buffer.getvalue().encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError(f"Course data fingerprint mismatch for {name}")
    return raw


def normalize_text(text):
    # 轻度清洗：修复 mojibake、Unicode 规范化、折叠空白。
    # 保留大小写、标点、emoji 和 @ 账号，不做词形还原。
    text = ftfy.fix_text(str(text))
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()

train_pool = pd.read_csv(io.BytesIO(read_raw("train")), keep_default_na=False)
test_df = pd.read_csv(io.BytesIO(read_raw("test")), keep_default_na=False)
manifest = pd.read_csv(io.BytesIO(read_raw("split")), keep_default_na=False)

for frame in (train_pool, test_df, manifest):
    frame["Sentiment"] = pd.to_numeric(frame["Sentiment"]).astype(int)

train_pool = train_pool.merge(manifest[["row_id", "partition"]], on="row_id", validate="one_to_one")
train_df = train_pool[train_pool.partition == "train"].reset_index(drop=True)
val_df = train_pool[train_pool.partition == "validation"].reset_index(drop=True)

for frame in (train_df, val_df, test_df):
    frame["text"] = frame["Phrase"].map(normalize_text)
    assert frame["text"].str.len().gt(0).all()

print("train:", len(train_df), " val:", len(val_df), " test:", len(test_df))
print(pd.crosstab(train_pool["partition"], train_pool["Sentiment"]).rename(columns=dict(enumerate(LABEL_NAMES))))
''', "data")

md('''
## 4. 冻结 BERT 提取句向量
只做一次前向，缓存 train/val/test 的嵌入。之后 MLP 在这些缓存向量上训练，BERT 权重始终不变。
''', "embed-note")
code(r'''
resolved_revision = model_info(MODEL_ID, revision=MODEL_REVISION).sha
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=resolved_revision)
bert = AutoModel.from_pretrained(MODEL_ID, revision=resolved_revision).to(device)
bert.eval()
for p in bert.parameters():
    p.requires_grad = False
print("BERT revision:", resolved_revision)
print("可训练参数(应为 0):", sum(p.numel() for p in bert.parameters() if p.requires_grad))

@torch.inference_mode()
def embed_texts(texts, batch_size=EMBED_BATCH_SIZE):
    chunks = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                          max_length=MAX_LENGTH, return_tensors="pt").to(device)
        hidden = bert(**batch).last_hidden_state  # (B, L, 768)
        mask = batch["attention_mask"].unsqueeze(-1).float()
        if POOLING == "mean":
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            pooled = hidden[:, 0, :]
        chunks.append(pooled.cpu().numpy())
    return np.concatenate(chunks, axis=0)

print("提取嵌入 ...")
X_train_emb = embed_texts(train_df.text.tolist())
X_val_emb = embed_texts(val_df.text.tolist())
X_test_emb = embed_texts(test_df.text.tolist())
print("嵌入维度:", X_train_emb.shape, X_val_emb.shape, X_test_emb.shape)
''', "embed")

md('''
## 5. 独立 MLP 分类器 + 超参搜索
MLP 在缓存的句向量上训练。可选拼接标准化后的四个表层特征。用 `class_weight="balanced"` 处理 Irrelevant 少数类。
网格搜索按验证集 macro-F1 选超参。
''', "mlp-note")
code(r'''
class MLP(nn.Module):
    def __init__(self, input_dim, hidden, dropout, n_classes=4):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

scaler = StandardScaler()
if USE_SURFACE_FEATURES:
    scaler.fit(train_df[SURFACE_FEATURES].astype(float))

def make_features(emb, frame):
    if USE_SURFACE_FEATURES:
        feats = scaler.transform(frame[SURFACE_FEATURES].astype(float))
        return np.concatenate([emb, feats], axis=1)
    return emb

X_tr = make_features(X_train_emb, train_df)
X_va = make_features(X_val_emb, val_df)
X_te = make_features(X_test_emb, test_df)
input_dim = X_tr.shape[1]
y_tr = train_df.Sentiment.to_numpy()
y_va = val_df.Sentiment.to_numpy()
y_te = test_df.Sentiment.to_numpy()
print("input_dim:", input_dim)

def train_mlp(hidden, dropout, lr, class_weight, seed=SEED, max_epochs=300, patience=15):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(input_dim, hidden, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    if class_weight:
        w = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        w = torch.tensor(w, dtype=torch.float32, device=device)
    else:
        w = None
    loss_fn = nn.CrossEntropyLoss(weight=w)
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.long, device=device)
    Xv = torch.tensor(X_va, dtype=torch.float32, device=device)
    best_val = -1.0
    best_state = None
    bad = 0
    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
        model.eval()
        with torch.inference_mode():
            val_pred = model(Xv).argmax(1).cpu().numpy()
        val_mf1 = f1_score(y_va, val_pred, labels=LABEL_IDS, average="macro", zero_division=0)
        if val_mf1 > best_val:
            best_val = val_mf1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_val

results = []
for hidden in GRID["hidden"]:
    for dropout in GRID["dropout"]:
        for lr in GRID["lr"]:
            for cw in GRID["class_weight"]:
                _, val_mf1 = train_mlp(hidden, dropout, lr, cw)
                results.append({"hidden": hidden, "dropout": dropout, "lr": lr,
                                "class_weight": cw, "val_macro_f1": round(val_mf1, 4)})
                print(f"hidden={hidden} dropout={dropout} lr={lr} cw={cw} -> val_macro_f1={val_mf1:.4f}")

tuning = pd.DataFrame(results).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
write_json(RUN_DIR / "tuning.json", tuning.to_dict(orient="records"))
print(tuning.to_string(index=False))
best = tuning.iloc[0]
print("最佳超参:", best.to_dict())

final_model, final_val = train_mlp(best["hidden"], best["dropout"], best["lr"], best["class_weight"])
final_model.eval()
with torch.inference_mode():
    test_logits = final_model(torch.tensor(X_te, dtype=torch.float32, device=device))
test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
test_pred = test_probs.argmax(1)
print("最终验证 macro-F1:", round(final_val, 4))
''', "mlp-train")

md('''
## 6. 最终测试
模型已按验证集选定，此处才读取测试标签。输出 accuracy、macro-F1、逐类指标、混淆矩阵和逐条预测，
并输出去掉 1 条历史跨集重叠样本的敏感性结果（主指标仍用全部 397 条）。
''', "test-note")
code(r'''
def metric_summary(y_true, y_pred):
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABEL_IDS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABEL_IDS, average="weighted", zero_division=0)),
        "irrelevant_recall": float(recall_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]),
        "classification_report": classification_report(
            y_true, y_pred, labels=LABEL_IDS, target_names=LABEL_NAMES, output_dict=True, zero_division=0),
    }

test_metrics = metric_summary(y_te, test_pred)
keep = test_df.cross_split_overlap.astype(int).eq(0).to_numpy()
test_metrics["without_historical_overlap"] = metric_summary(y_te[keep], test_pred[keep])
test_metrics["best_hyperparameters"] = best.to_dict()
write_json(RUN_DIR / "test_metrics.json", test_metrics)

print(classification_report(y_te, test_pred, labels=LABEL_IDS, target_names=LABEL_NAMES, digits=4, zero_division=0))
print("测试 macro-F1:", test_metrics["macro_f1"], " accuracy:", test_metrics["accuracy"])

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix(y_te, test_pred, labels=LABEL_IDS),
                       display_labels=LABEL_NAMES).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("BERT (frozen) + MLP — test set")
fig.tight_layout()
fig.savefig(RUN_DIR / "confusion_matrix.png", dpi=160)
plt.show()

predictions = test_df[["row_id", "Sentiment", "cross_split_overlap"]].copy()
predictions["predicted_id"] = test_pred
predictions["predicted_label"] = [LABEL_NAMES[i] for i in test_pred]
predictions["correct"] = predictions.Sentiment.eq(predictions.predicted_id)
for i, name in enumerate(LABEL_NAMES):
    predictions[f"prob_{name}"] = test_probs[:, i]
predictions.to_csv(RUN_DIR / "test_predictions.csv", index=False)
''', "test")

md('''
## 7. 保存与下载
导出超参搜索、测试指标、混淆矩阵和逐条预测。ZIP 不含 BERT 权重（公开下载即可），只存结果与配置。
''', "export-note")
code(r'''
write_json(RUN_DIR / "run_config.json", {
    "model_id": MODEL_ID, "model_revision": resolved_revision,
    "pooling": POOLING, "max_length": MAX_LENGTH, "seed": SEED,
    "use_surface_features": USE_SURFACE_FEATURES,
    "surface_features": SURFACE_FEATURES if USE_SURFACE_FEATURES else [],
    "train_rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df),
    "best_hyperparameters": best.to_dict(),
})

archive = RUN_DIR / "bert_mlp_sentiment.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
    for name in ("run_config.json", "tuning.json", "test_metrics.json", "test_predictions.csv", "confusion_matrix.png"):
        bundle.write(RUN_DIR / name, arcname=name)
print("导出:", archive)
try:
    from google.colab import files
    files.download(str(archive))
except ImportError:
    print("非 Colab 环境，请直接复制上面的 ZIP。")
''', "export")


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"name": NOTEBOOK_NAME, "provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
