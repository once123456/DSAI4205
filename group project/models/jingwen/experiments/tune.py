"""Step 5: systematic tuning of the multi-view SVM on the 591-row pool (repeated 5-fold CV, test never read).

A fusion type, B view weights, C an extra general encoder view, D instruction wording, E kernel type,
F input length and pooling. Every fitted step (scaling inside the SVM) is fitted inside each fold.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the core modules
from data import EXPERIMENTS_DIR, load_splits
from evaluate import macro_f1
from exp_features import load_matrices


C_GRID = (2, 4, 8, 16, 32)
MAIN = ["e5i_sentiment", "e5i_topic", "e5i_style"]


def rbf(a, b, gamma):
    sq = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
    return np.exp(-gamma * np.maximum(sq, 0))


class Folds:
    def __init__(self, m, y, pool, cv):
        self.m, self.y, self.pool, self.cv = m, y, pool, cv

    def view_kernels(self, x, kind="rbf", gamma_scale=2.0):
        out = []
        for a, b in self.cv:
            fa, fb = normalize(x[self.pool[a]]), normalize(x[self.pool[b]])
            if kind == "rbf":
                g = gamma_scale / (fa.shape[1] * fa.var())
                out.append((rbf(fa, fa, g), rbf(fb, fa, g)))
            elif kind == "linear":
                out.append((fa @ fa.T, fb @ fa.T))
            elif kind == "poly2":
                out.append(((1 + fa @ fa.T) ** 2, (1 + fb @ fa.T) ** 2))
        return out

    def score(self, fold_kernels):
        best = None
        for c in C_GRID:
            f1s = [macro_f1(self.y[self.pool[b]], SVC(C=c, kernel="precomputed", class_weight="balanced")
                            .fit(kf, self.y[self.pool[a]]).predict(ko)) for (a, b), (kf, ko) in zip(self.cv, fold_kernels)]
            if best is None or np.mean(f1s) > best["mean"]:
                best = {"mean": float(np.mean(f1s)), "std": float(np.std(f1s)), "C": c}
        return best


def combine(kernel_sets, weights, mode="sum"):
    weights = np.asarray(weights, float) / np.sum(weights)
    out = []
    for parts in zip(*kernel_sets):
        if mode == "sum":
            out.append((sum(w * p[0] for w, p in zip(weights, parts)), sum(w * p[1] for w, p in zip(weights, parts))))
        else:
            out.append((np.prod([p[0] ** w for w, p in zip(weights, parts)], 0),
                        np.prod([p[1] ** w for w, p in zip(weights, parts)], 0)))
    return out


def main():
    train, validation, test = load_splits()
    m = load_matrices([train, validation, test])
    pool = np.arange(len(train) + len(validation))
    y = np.concatenate([train.Sentiment, validation.Sentiment, np.full(len(test), -1)])
    cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1).split(pool, y[pool]))
    f = Folds(m, y, pool, cv)
    res = {}

    def log(section, name, r):
        res.setdefault(section, {})[name] = r
        print(f"[{section}] {name:34s} {r['mean']:.4f} +- {r['std']:.4f}  C {r['C']}", flush=True)

    k = {v: f.view_kernels(m[v]) for v in MAIN + ["bge_large", "e5_large", "e5i_sentiment_v2", "e5i_topic_v2", "e5i_style_v2"]}
    # part A: fusion type
    log("A_fusion", "sum of view kernels (MKL)", f.score(combine([k[v] for v in MAIN], [1, 1, 1])))
    log("A_fusion", "product of view kernels", f.score(combine([k[v] for v in MAIN], [1, 1, 1], "product")))
    concat = np.hstack([normalize(m[v]) for v in MAIN])
    log("A_fusion", "concatenation, one RBF", f.score(f.view_kernels(concat)))
    # part B: view weights
    for w in itertools.product((1, 2, 3), repeat=3):
        if np.gcd.reduce(w) == 1:
            log("B_weights", f"sent {w[0]} topic {w[1]} style {w[2]}", f.score(combine([k[v] for v in MAIN], w)))
    # part C: extra general view
    for extra in ("bge_large", "e5_large"):
        for w in (0.5, 1):
            log("C_extra", f"+{extra} weight {w}", f.score(combine([k[v] for v in MAIN] + [k[extra]], [1, 1, 1, w])))
    # part D: instruction wording
    log("D_wording", "wording v1", f.score(combine([k[v] for v in MAIN], [1, 1, 1])))
    log("D_wording", "wording v2", f.score(combine([k[v + "_v2"] for v in MAIN], [1, 1, 1])))
    log("D_wording", "v1 and v2 kernels averaged", f.score(combine([k[v] for v in MAIN] + [k[v + "_v2"] for v in MAIN], [1] * 6)))
    # part E: kernel type
    for kind in ("linear", "poly2"):
        log("E_kernel", kind, f.score(combine([f.view_kernels(m[v], kind) for v in MAIN], [1, 1, 1])))
    for g in (1.0, 2.0, 4.0):
        log("E_kernel", f"rbf gamma x{g}", f.score(combine([f.view_kernels(m[v], "rbf", g) for v in MAIN], [1, 1, 1])))
    (EXPERIMENTS_DIR / "tune.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
