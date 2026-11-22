"""Robust design ranking: repeated stratified 5-fold CV over the whole 591-row pool (train + validation).

The validation rows exist for model selection, so pooling them gives a steadier estimate than 118 rows alone.
Test labels are never read.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the core modules
from data import EXPERIMENTS_DIR, load_splits
from exp_features import C_GRID, DESIGNS, load_matrices, score_design


EXTRA = {
    "sent+topic+bge+e5": [("e5i_sentiment", "norm", 1), ("e5i_topic", "norm", 1), ("bge_large", "norm", 1),
                          ("e5_large", "norm", 1)],
    "sent+topic+bge+e5+masked": [("e5i_sentiment", "norm", 1), ("e5i_topic", "norm", 1), ("bge_large", "norm", 1),
                                 ("e5_large", "norm", 1), ("bge_large_masked", "norm", 1), ("e5_large_masked", "norm", 1)],
    "topic+bge+e5": [("e5i_topic", "norm", 1), ("bge_large", "norm", 1), ("e5_large", "norm", 1)],
    "e5i_relevance": [("e5i_relevance", "norm", 1)],
    "e5i_style": [("e5i_style", "norm", 1)],
    "sent+topic+relevance": [("e5i_sentiment", "norm", 1), ("e5i_topic", "norm", 1), ("e5i_relevance", "norm", 1)],
    "sent+topic+style": [("e5i_sentiment", "norm", 1), ("e5i_topic", "norm", 1), ("e5i_style", "norm", 1)],
    "sent+topic+relevance+style": [("e5i_sentiment", "norm", 1), ("e5i_topic", "norm", 1),
                                   ("e5i_relevance", "norm", 1), ("e5i_style", "norm", 1)],
    "all5": [("e5i_sentiment", "norm", 1), ("e5i_topic", "norm", 1), ("e5i_emotion", "norm", 1),
             ("e5i_relevance", "norm", 1), ("e5i_style", "norm", 1)],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--designs", nargs="+", required=True)
    parser.add_argument("--out", default="pool_cv.json")
    args = parser.parse_args()
    train, validation, test = load_splits()
    m = load_matrices([train, validation, test])
    pool_n = len(train) + len(validation)
    y = np.concatenate([train.Sentiment, validation.Sentiment, np.full(len(test), -1)])
    pool_idx = np.arange(pool_n)
    cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1).split(pool_idx, y[pool_idx]))
    designs = DESIGNS | EXTRA
    results = {}
    for name in args.designs:
        folds = [score_design(designs[name], m, y, pool_idx[a], pool_idx[b]) for a, b in cv]
        by_c = {c: [f[c] for f in folds] for c in C_GRID}
        best_c = max(by_c, key=lambda c: np.mean(by_c[c]))
        results[name] = {"pool_cv_mean": float(np.mean(by_c[best_c])), "pool_cv_std": float(np.std(by_c[best_c])),
                         "C": best_c, "folds": [float(v) for v in by_c[best_c]]}
        print(f"{name:28s} {results[name]['pool_cv_mean']:.4f} +- {results[name]['pool_cv_std']:.4f}  C {best_c}", flush=True)
    (EXPERIMENTS_DIR / args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
