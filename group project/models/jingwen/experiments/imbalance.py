"""Step 6: class weighting and C for the final six-view model, on the 591-row pool (test labels never read).

The same repeated 5-fold CV as final_eval.py. For every C and both class-weight settings it records macro-F1
and the per-class recall and F1, so the effect on the Irrelevant class can be read directly.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the core modules
from data import EXPERIMENTS_DIR, LABEL_NAMES, load_splits
from final_eval import C_GRID
from views import SIX_VIEWS, fused_kernel, load_views


def main():
    train, validation, _ = load_splits()
    m = load_views(SIX_VIEWS, [train, validation])
    y = np.concatenate([train.Sentiment, validation.Sentiment])
    pool = np.arange(len(y))
    cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1).split(pool, y))
    kernels = [fused_kernel(m, SIX_VIEWS, a, b) for a, b in cv]
    results = []
    for weight in ["balanced", None]:
        for c in C_GRID:
            folds = []
            for (a, b), (kf, ko) in zip(cv, kernels):
                pred = SVC(C=c, kernel="precomputed", class_weight=weight).fit(kf, y[a]).predict(ko)
                _, r, f, _ = precision_recall_fscore_support(y[b], pred, labels=range(4), zero_division=0)
                folds.append(np.concatenate([[f.mean()], r, f]))
            mean, std = np.mean(folds, 0), np.std(folds, 0)
            row = {"class_weight": weight or "none", "C": c, "macro_f1": float(mean[0]), "macro_f1_std": float(std[0])}
            row |= {f"{lab}_recall": float(mean[1 + i]) for i, lab in enumerate(LABEL_NAMES)}
            row |= {f"{lab}_f1": float(mean[5 + i]) for i, lab in enumerate(LABEL_NAMES)}
            results.append(row)
            print(row)
    (EXPERIMENTS_DIR / "imbalance.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
