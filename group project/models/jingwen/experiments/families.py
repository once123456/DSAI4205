"""Other classifier families on the same six instruction views, for the group (pooled CV, test never read).

Each family gets a small grid; the best setting by repeated 5-fold CV on the 591-row pool is reported.
Inputs: the six L2-normalized views concatenated (6 x 1024), optionally reduced by PCA fitted inside each fold.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the core modules
from data import EXPERIMENTS_DIR, load_splits
from evaluate import macro_f1
from exp_features import load_matrices


SIX = ["e5i_sentiment", "e5i_topic", "e5i_style", "e5i_sentiment_v2", "e5i_topic_v2", "e5i_style_v2"]


def grids():
    yield "kNN (cosine)", {f"k={k}": KNeighborsClassifier(k, metric="cosine", weights="distance") for k in (5, 10, 20, 40)}
    yield "Random forest", {f"trees=500 depth={d}": RandomForestClassifier(500, max_depth=d, class_weight="balanced",
                                                                            n_jobs=8, random_state=0) for d in (None, 12)}
    yield "Gradient boosting (PCA 64)", {f"lr={lr}": make_pipeline(PCA(64, random_state=0), HistGradientBoostingClassifier(
        learning_rate=lr, max_iter=300, class_weight="balanced", random_state=0)) for lr in (0.03, 0.1)}
    yield "Gaussian process (PCA 128)", {f"length={ls}": make_pipeline(PCA(128, random_state=0), GaussianProcessClassifier(
        RBF(ls), optimizer=None, random_state=0)) for ls in (1.0, 2.0)}
    yield "MLP", {f"hidden={h} alpha={a} epochs={e}": MLPClassifier((h,), alpha=a, max_iter=e, random_state=0)
                  for h in (128, 512) for a in (1e-2, 1.0) for e in (50, 200)}


def main():
    train, validation, test = load_splits()
    m = load_matrices([train, validation, test])
    x = np.hstack([normalize(m[v]) for v in SIX])
    y = np.concatenate([train.Sentiment, validation.Sentiment, np.full(len(test), -1)])
    pool = np.arange(len(train) + len(validation))
    cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1).split(pool, y[pool]))
    results = {}
    only = set(sys.argv[1:])
    for family, models in grids():
        if only and family not in only:
            continue
        best = None
        for setting, model in models.items():
            f1s = [macro_f1(y[pool[b]], model.fit(x[pool[a]], y[pool[a]]).predict(x[pool[b]])) for a, b in cv]
            if best is None or np.mean(f1s) > best["mean"]:
                best = {"setting": setting, "mean": float(np.mean(f1s)), "std": float(np.std(f1s))}
        results[family] = best
        print(f"{family:28s} {best['mean']:.4f} +- {best['std']:.4f}  ({best['setting']})", flush=True)
    path = EXPERIMENTS_DIR / "families.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(old | results, indent=2))


if __name__ == "__main__":
    main()
