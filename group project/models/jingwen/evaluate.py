"""Metrics and exports in the group's format (macro-F1 first, accuracy, per class, paired bootstrap)."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from data import LABEL_IDS, LABEL_NAMES


def macro_f1(gold, predicted):
    return f1_score(gold, predicted, labels=LABEL_IDS, average="macro", zero_division=0)


def metrics(gold, predicted):
    report = classification_report(gold, predicted, labels=LABEL_IDS, target_names=LABEL_NAMES,
                                   output_dict=True, zero_division=0)
    return {"accuracy": accuracy_score(gold, predicted), "macro_f1": macro_f1(gold, predicted),
            "per_class": {name: report[name] for name in LABEL_NAMES},
            "confusion_matrix": confusion_matrix(gold, predicted, labels=LABEL_IDS).tolist()}


def per_class_table(gold, predicted, model_name):
    rows = []
    for name, scores in metrics(gold, predicted)["per_class"].items():
        rows.append({"model": model_name, "label": name, "precision": scores["precision"],
                     "recall": scores["recall"], "f1": scores["f1-score"], "support": int(scores["support"])})
    return pd.DataFrame(rows)


def paired_bootstrap(gold, treatment, control, resamples=1000, seed=2026):
    """Macro-F1 difference (treatment minus control) with a 95% paired bootstrap interval."""
    gold, treatment, control = map(np.asarray, (gold, treatment, control))
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(resamples):
        idx = rng.integers(0, len(gold), len(gold))
        differences.append(macro_f1(gold[idx], treatment[idx]) - macro_f1(gold[idx], control[idx]))
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"delta_macro_f1": macro_f1(gold, treatment) - macro_f1(gold, control),
            "delta_ci_low": low, "delta_ci_high": high, "resamples": resamples}


def prediction_frame(test, predicted):
    frame = test[["row_id", "Phrase", "Sentiment", "cross_split_overlap"]].copy()
    frame["predicted_id"] = np.asarray(predicted)
    frame["correct"] = frame.Sentiment.eq(frame.predicted_id)
    return frame
