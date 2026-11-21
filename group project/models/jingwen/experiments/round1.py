"""Iteration round 1 for the SVM: relevance view in the prompt ensemble, and last-four-layer pooling."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the core modules
from data import EXPERIMENTS_DIR, load_splits
from exp_features import load_matrices
from tune import Folds, combine


SIX = ["e5i_sentiment", "e5i_topic", "e5i_style", "e5i_sentiment_v2", "e5i_topic_v2", "e5i_style_v2"]
CANDIDATES = {
    "six views (current final)": SIX,
    "six + relevance v1": SIX + ["e5i_relevance"],
    "six + relevance v1 v2": SIX + ["e5i_relevance", "e5i_relevance_v2"],
    "six, last-4 layers": [v + "_last4" for v in SIX],
    "six, last layer + last-4 layers": SIX + [v + "_last4" for v in SIX],
    "modernbert classification": ["modernbert_classification"],
    "modernbert clustering": ["modernbert_clustering"],
    "modernbert both prefixes": ["modernbert_classification", "modernbert_clustering"],
    "gte-modernbert": ["gte_modernbert"],
    "six + modernbert both": SIX + ["modernbert_classification", "modernbert_clustering"],
    "six + gte-modernbert": SIX + ["gte_modernbert"],
}


def main():
    train, validation, test = load_splits()
    m = load_matrices([train, validation, test])
    pool = np.arange(len(train) + len(validation))
    y = np.concatenate([train.Sentiment, validation.Sentiment, np.full(len(test), -1)])
    cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1).split(pool, y[pool]))
    f = Folds(m, y, pool, cv)
    cache, res = {}, {}
    for name, views in CANDIDATES.items():
        for v in views:
            cache.setdefault(v, f.view_kernels(m[v]))
        res[name] = f.score(combine([cache[v] for v in views], [1] * len(views)))
        print(f"{name:36s} {res[name]['mean']:.4f} +- {res[name]['std']:.4f}  C {res[name]['C']}", flush=True)
    (EXPERIMENTS_DIR / "round1.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
