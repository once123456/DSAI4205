"""Step 4: refinement of the chosen design on the 591-row pool (test labels never read).

1. SVM kernel width, C and class weighting, by repeated 5-fold CV.
2. Per-class decision offsets that maximise macro-F1. They are fitted on inner out-of-fold scores of each
   outer training part and scored on the outer fold (nested), so the reported gain is not optimistic.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the core modules
from data import EXPERIMENTS_DIR, load_splits
from evaluate import macro_f1
from exp_features import kernel, load_matrices, transform


DESIGN = [("e5i_sentiment", "norm", 1), ("e5i_topic", "norm", 1), ("e5i_style", "norm", 1)]
OFFSETS = np.round(np.arange(-0.6, 0.61, 0.1), 2)


def fused_kernel(m, fit_idx, other_idx, y, gamma_scale):
    k_fit, k_other, total = 0, 0, sum(w for _, _, w in DESIGN)
    for view, tname, weight in DESIGN:
        a, b = transform(view, tname, m[view], fit_idx, other_idx, y)
        gamma = gamma_scale / (a.shape[1] * a.var())
        k_fit = k_fit + weight / total * kernel(a, a, gamma)
        k_other = k_other + weight / total * kernel(b, a, gamma)
    return k_fit, k_other


def svm(c, class_weight):
    return SVC(C=c, kernel="precomputed", class_weight=class_weight, decision_function_shape="ovr")


def fit_offsets(scores, gold):
    """Grid search one additive offset per class (the first fixed at 0) for the best macro-F1."""
    best, best_f1 = np.zeros(4), macro_f1(gold, scores.argmax(1))
    for o in itertools.product(OFFSETS, repeat=3):
        offsets = np.array((0.0,) + o)
        f1 = macro_f1(gold, (scores + offsets).argmax(1))
        if f1 > best_f1 + 1e-9:
            best, best_f1 = offsets, f1
    return best


def inner_oof_scores(m, y, idx, c, cw, g, seed):
    scores = np.zeros((len(idx), 4))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(idx, y[idx]):
        k_fit, k_other = fused_kernel(m, idx[a], idx[b], y, g)
        scores[b] = svm(c, cw).fit(k_fit, y[idx[a]]).decision_function(k_other)
    return scores


def main():
    train, validation, test = load_splits()
    m = load_matrices([train, validation, test])
    pool_n = len(train) + len(validation)
    y = np.concatenate([train.Sentiment, validation.Sentiment, np.full(len(test), -1)])
    pool = np.arange(pool_n)
    cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1).split(pool, y[pool]))
    results = {"grid": []}
    for g, cw in itertools.product((0.5, 1, 2), ("balanced", None)):
        kernels = [fused_kernel(m, pool[a], pool[b], y, g) for a, b in cv]
        for c in (2, 4, 8, 16, 32):
            f1s = [macro_f1(y[pool[b]], svm(c, cw).fit(kf, y[pool[a]]).predict(ko)) for (a, b), (kf, ko) in zip(cv, kernels)]
            row = {"gamma_scale": g, "class_weight": cw or "none", "C": c, "mean": float(np.mean(f1s)), "std": float(np.std(f1s))}
            results["grid"].append(row)
    best = max(results["grid"], key=lambda r: r["mean"])
    results["best"] = best
    print("best svm setting", best, flush=True)
    g, cw, c = best["gamma_scale"], (None if best["class_weight"] == "none" else "balanced"), best["C"]
    plain, offset = [], []
    for fold, (a, b) in enumerate(cv):
        fit_idx, eval_idx = pool[a], pool[b]
        offsets = fit_offsets(inner_oof_scores(m, y, fit_idx, c, cw, g, seed=fold), y[fit_idx])
        k_fit, k_eval = fused_kernel(m, fit_idx, eval_idx, y, g)
        scores = svm(c, cw).fit(k_fit, y[fit_idx]).decision_function(k_eval)
        plain.append(macro_f1(y[eval_idx], scores.argmax(1)))
        offset.append(macro_f1(y[eval_idx], (scores + offsets).argmax(1)))
    results["nested_offsets"] = {"plain_mean": float(np.mean(plain)), "offset_mean": float(np.mean(offset)),
                                 "mean_gain": float(np.mean(np.subtract(offset, plain))),
                                 "folds_improved": int(np.sum(np.subtract(offset, plain) > 0)), "folds": len(cv)}
    print("nested offsets", results["nested_offsets"], flush=True)
    (EXPERIMENTS_DIR / "refine.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
