"""Step 3: feature engineering and kernel fusion for the SVM.

Each design is a list of views (encoder output + transform + weight). Every fitted transform (whitening,
scaling, prototype features) is fitted inside each fold. Designs are ranked by repeated 5-fold CV on the
473 training rows; the validation score is reported as an independent check. Test labels are never read.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the core modules
from data import EXPERIMENTS_DIR, load_splits, WORK_DIR
from encoders import cached_encode
from evaluate import macro_f1
from features import Whitener, mask_entities, prototype_knn_oof, scaled, surface_features


C_GRID = (0.5, 1, 2, 4, 8, 16)


def load_matrices(frames):
    texts = [t for f in frames for t in f.text]
    masked = [mask_entities(t) for t in texts]
    m = {}
    for name in ("bge_large", "e5_large", "mpnet", "bertweet", "e5i_sentiment", "e5i_topic", "e5i_emotion",
                 "e5i_relevance", "e5i_style", "e5i_sentiment_v2", "e5i_topic_v2", "e5i_style_v2",
                 "e5i_relevance_v2", "e5i_sentiment_last4", "e5i_topic_last4", "e5i_style_last4",
                 "e5i_sentiment_v2_last4", "e5i_topic_v2_last4", "e5i_style_v2_last4",
                 "modernbert_classification", "modernbert_clustering", "gte_modernbert"):
        m[name] = cached_encode(texts, name, WORK_DIR / "cache")
    for name in ("bge_large", "e5_large"):
        m[name + "_masked"] = cached_encode(masked, name, WORK_DIR / "cache", tag="masked")
        m[name + "_diff"] = normalize(m[name]) - normalize(m[name + "_masked"])
    import pandas as pd
    m["surface"] = surface_features(pd.concat(frames, ignore_index=True))
    return m


def transform(view, transform_name, x, fit_idx, other_idx, y):
    if transform_name == "norm":
        return normalize(x[fit_idx]), normalize(x[other_idx])
    if transform_name.startswith("whiten"):
        dim = int(transform_name[6:] or 128)
        w = Whitener(dim).fit(normalize(x[fit_idx]))
        return w.transform(normalize(x[fit_idx])), w.transform(normalize(x[other_idx]))
    if transform_name == "scale":
        a, (b,) = scaled(x[fit_idx], [x[other_idx]])
        return a, b
    if transform_name == "proto":
        a, (b,) = prototype_knn_oof(normalize(x[fit_idx]), y[fit_idx], [normalize(x[other_idx])])
        return a, b
    raise ValueError(transform_name)


def kernel(a, b, gamma):
    sq = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
    return np.exp(-gamma * np.maximum(sq, 0))


def design_kernels(design, m, fit_idx, other_idx, y):
    k_fit, k_other, total = 0, 0, sum(w for _, _, w in design)
    for view, tname, weight in design:
        a, b = transform(view, tname, m[view], fit_idx, other_idx, y)
        gamma = 1.0 / (a.shape[1] * a.var())
        k_fit = k_fit + weight / total * kernel(a, a, gamma)
        k_other = k_other + weight / total * kernel(b, a, gamma)
    return k_fit, k_other


def score_design(design, m, y, fit_idx, eval_idx):
    k_fit, k_eval = design_kernels(design, m, fit_idx, eval_idx, y)
    return {c: macro_f1(y[eval_idx], SVC(C=c, kernel="precomputed", class_weight="balanced")
                        .fit(k_fit, y[fit_idx]).predict(k_eval)) for c in C_GRID}


def evaluate(design, m, y, train_idx, val_idx):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
    fold_scores = [score_design(design, m, y, train_idx[a], train_idx[b])
                   for a, b in cv.split(train_idx, y[train_idx])]
    cv_by_c = {c: float(np.mean([s[c] for s in fold_scores])) for c in C_GRID}
    best_c = max(cv_by_c, key=cv_by_c.get)
    val = score_design(design, m, y, train_idx, val_idx)[best_c]
    return {"cv_macro_f1": cv_by_c[best_c], "C": best_c, "val_macro_f1": float(val)}


DESIGNS = {
    "bge": [("bge_large", "norm", 1)],
    "e5": [("e5_large", "norm", 1)],
    "mpnet": [("mpnet", "norm", 1)],
    "bertweet": [("bertweet", "norm", 1)],
    "bge_whiten": [("bge_large", "whiten", 1)],
    "e5_whiten": [("e5_large", "whiten", 1)],
    "bge+e5": [("bge_large", "norm", 1), ("e5_large", "norm", 1)],
    "bge+e5+mpnet": [("bge_large", "norm", 1), ("e5_large", "norm", 1), ("mpnet", "norm", 1)],
    "bge+e5+mpnet+bertweet": [("bge_large", "norm", 1), ("e5_large", "norm", 1), ("mpnet", "norm", 1),
                              ("bertweet", "norm", 1)],
    "bge+masked": [("bge_large", "norm", 1), ("bge_large_masked", "norm", 1)],
    "bge+diff": [("bge_large", "norm", 2), ("bge_large_diff", "scale", 1)],
    "bge+surface": [("bge_large", "norm", 4), ("surface", "scale", 1)],
    "bge+proto": [("bge_large", "norm", 3), ("bge_large", "proto", 1)],
    "bge+e5+surface": [("bge_large", "norm", 2), ("e5_large", "norm", 2), ("surface", "scale", 1)],
    "bge+e5+proto": [("bge_large", "norm", 2), ("e5_large", "norm", 2), ("bge_large", "proto", 1)],
    "bge_whiten64": [("bge_large", "whiten64", 1)],
    "bge_whiten256": [("bge_large", "whiten256", 1)],
    "e5_whiten128": [("e5_large", "whiten128", 1)],
    "e5i_sentiment": [("e5i_sentiment", "norm", 1)],
    "e5i_topic": [("e5i_topic", "norm", 1)],
    "e5i_emotion": [("e5i_emotion", "norm", 1)],
    "e5i_sent+topic": [("e5i_sentiment", "norm", 1), ("e5i_topic", "norm", 1)],
    "e5i_all3": [("e5i_sentiment", "norm", 1), ("e5i_topic", "norm", 1), ("e5i_emotion", "norm", 1)],
    "e5i_sent+bge+e5": [("e5i_sentiment", "norm", 1), ("bge_large", "norm", 1), ("e5_large", "norm", 1)],
    "e5i_all3+bge+e5": [("e5i_sentiment", "norm", 1), ("e5i_topic", "norm", 1), ("e5i_emotion", "norm", 1),
                        ("bge_large", "norm", 1), ("e5_large", "norm", 1)],
    "bge+e5+masked": [("bge_large", "norm", 1), ("e5_large", "norm", 1), ("bge_large_masked", "norm", 1),
                      ("e5_large_masked", "norm", 1)],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--designs", nargs="+", default=list(DESIGNS))
    parser.add_argument("--out", default="exp_features.json")
    args = parser.parse_args()
    train, validation, test = load_splits()
    m = load_matrices([train, validation, test])
    y = np.concatenate([train.Sentiment, validation.Sentiment, np.full(len(test), -1)])
    train_idx = np.arange(len(train))
    val_idx = np.arange(len(train), len(train) + len(validation))
    results = {}
    for name in args.designs:
        results[name] = evaluate(DESIGNS[name], m, y, train_idx, val_idx)
        print(f"{name:24s} cv {results[name]['cv_macro_f1']:.4f}  val {results[name]['val_macro_f1']:.4f}  C {results[name]['C']}",
              flush=True)
    path = EXPERIMENTS_DIR / args.out
    path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
