"""Step 6: does removing likely mislabelled training rows help? (nested, test never read)

Inside each outer training part, out-of-fold SVM scores give every row a self-confidence: the margin of its
given label against the best other label. The rows with the lowest margins are dropped, the SVM is refitted,
and the outer fold is scored. The flagged rows of the full pool are saved for inspection.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the core modules
from data import EXPERIMENTS_DIR, load_splits
from evaluate import macro_f1
from exp_features import load_matrices
from views import rbf


VIEWS = ["e5i_sentiment", "e5i_topic", "e5i_style", "e5i_sentiment_v2", "e5i_topic_v2", "e5i_style_v2"]
C, GAMMA_SCALE = 4, 2.0


def fused(m, fit_idx, other_idx):
    kf, ko = 0, 0
    for v in VIEWS:
        a, b = normalize(m[v][fit_idx]), normalize(m[v][other_idx])
        g = GAMMA_SCALE / (a.shape[1] * a.var())
        kf, ko = kf + rbf(a, a, g) / len(VIEWS), ko + rbf(b, a, g) / len(VIEWS)
    return kf, ko


def margins(m, y, idx, seed):
    out = np.zeros(len(idx))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(idx, y[idx]):
        kf, ko = fused(m, idx[a], idx[b])
        s = SVC(C=C, kernel="precomputed", class_weight="balanced").fit(kf, y[idx[a]]).decision_function(ko)
        own = s[np.arange(len(b)), y[idx[b]]]
        s[np.arange(len(b)), y[idx[b]]] = -np.inf
        out[b] = own - s.max(1)
    return out


def main():
    train, validation, test = load_splits()
    m = load_matrices([train, validation, test])
    pool = np.arange(len(train) + len(validation))
    y = np.concatenate([train.Sentiment, validation.Sentiment, np.full(len(test), -1)])
    cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1).split(pool, y[pool]))
    results = {}
    for drop in (0.0, 0.03, 0.05, 0.1, 0.15):
        f1s = []
        for fold, (a, b) in enumerate(cv):
            fit_idx = pool[a]
            if drop:
                mg = margins(m, y, fit_idx, seed=fold)
                fit_idx = fit_idx[np.argsort(mg)[int(drop * len(fit_idx)):]]
            kf, ko = fused(m, fit_idx, pool[b])
            f1s.append(macro_f1(y[pool[b]], SVC(C=C, kernel="precomputed", class_weight="balanced").fit(kf, y[fit_idx]).predict(ko)))
        results[str(drop)] = {"mean": float(np.mean(f1s)), "std": float(np.std(f1s))}
        print(f"drop {drop:.2f}: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}", flush=True)
    mg = margins(m, y, pool, seed=0)
    pool_frame = __import__("pandas").concat([train, validation], ignore_index=True)
    flagged = pool_frame.assign(margin=mg).sort_values("margin").head(30)[["row_id", "Sentiment", "margin"]]
    flagged.to_csv(EXPERIMENTS_DIR / "suspect_labels.csv", index=False)
    (EXPERIMENTS_DIR / "noise.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
