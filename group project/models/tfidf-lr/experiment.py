"""Run the CPU comparison on the embedded project snapshot."""
import base64
import gzip
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline

LABEL_NAMES = ["Negative", "Positive", "Irrelevant", "Neutral"]
LABEL_IDS = list(range(len(LABEL_NAMES)))


def read_snapshot(name):
    names = {"train": "train_cleaned.csv", "test": "test_cleaned.csv", "split": "split_manifest.csv"}
    return pd.read_csv(DATA_DIR / names[name], keep_default_na=False)


pool = read_snapshot("train")
held_out = read_snapshot("test")
split = read_snapshot("split").set_index("row_id", verify_integrity=True)
pool = pool.set_index("row_id", verify_integrity=True)
baseline_rows = []
if QWEN_EXPORT:
    with ZipFile(QWEN_EXPORT) as archive:
        with archive.open("split_manifest.csv") as file:
            qwen_split = pd.read_csv(file).set_index("row_id", verify_integrity=True)
        with archive.open("test_predictions.csv") as file:
            qwen_test = pd.read_csv(file, keep_default_na=False).set_index("row_id", verify_integrity=True)
    pd.testing.assert_frame_equal(split[["Sentiment", "group_id"]].sort_index(),
                                  qwen_split[["Sentiment", "group_id"]].sort_index())
    pd.testing.assert_frame_equal(held_out.set_index("row_id")[["Phrase", "Sentiment"]].sort_index(),
                                  qwen_test[["Phrase", "Sentiment"]].sort_index())
    if not qwen_test.predicted_id.isin(LABEL_IDS).all():
        raise ValueError("Qwen 测试预测包含无效标签。")
    split = qwen_split
    baseline_rows.append({"variant": "Qwen3-0.6B + LoRA", "train_n": int(split.partition.eq("train").sum()),
                          "test_n": len(qwen_test), "accuracy": accuracy_score(qwen_test.Sentiment, qwen_test.predicted_id),
                          "macro_f1": f1_score(qwen_test.Sentiment, qwen_test.predicted_id, labels=LABEL_IDS, average="macro")})

pd.testing.assert_series_equal(pool.Sentiment.sort_index(), split.Sentiment.sort_index())
if set(split.partition) != {"train", "validation"} or split.groupby("group_id").partition.nunique().max() != 1:
    raise ValueError("划分清单须包含 train 和 validation，且同组推文须归入同一子集。")
train_df = pool.loc[split.index[split.partition.eq("train")]]
RUN_DIR = Path("tfidf_runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
RUN_DIR.mkdir(parents=True)
split.to_csv(RUN_DIR / "split_manifest.csv")
for variant, fit_data in [("same_training_subset", train_df), ("full_training_pool", pool)]:
    estimator = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                              LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=7))
    estimator.fit(fit_data.text_clean, fit_data.Sentiment)
    predicted = estimator.predict(held_out.text_clean)
    scores = classification_report(held_out.Sentiment, predicted, labels=LABEL_IDS,
                                   target_names=LABEL_NAMES, output_dict=True, zero_division=0)
    (RUN_DIR / f"tfidf_{variant}_metrics.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    held_out.assign(predicted=predicted)[["row_id", "Sentiment", "predicted"]].to_csv(
        RUN_DIR / f"tfidf_{variant}_predictions.csv", index=False)
    baseline_rows.append({"variant": variant, "train_n": len(fit_data), "test_n": len(held_out),
                          "accuracy": accuracy_score(held_out.Sentiment, predicted),
                          "macro_f1": scores["macro avg"]["f1-score"]})
baseline_table = pd.DataFrame(baseline_rows)
display(baseline_table)
baseline_table.to_csv(RUN_DIR / "tfidf_comparison.csv", index=False)
print("结果目录", RUN_DIR)
