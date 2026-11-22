"""Step 7b: does paraphrase augmentation help the six-view SVM? (pooled CV, test never read)

In every fold only the paraphrases of that fold's training rows are added; the held-out rows stay original.
Augmented rows can get a smaller sample weight.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the core modules
from data import EXPERIMENTS_DIR, load_splits, WORK_DIR
from encoders import cached_encode
from evaluate import macro_f1
from exp_features import load_matrices
from views import rbf


SIX = ["e5i_sentiment", "e5i_topic", "e5i_style", "e5i_sentiment_v2", "e5i_topic_v2", "e5i_style_v2"]
C, GAMMA_SCALE = 4, 2.0


PREAMBLE_START = ("okay", "sure", "here")


def clean_paraphrase(text):
    """Drop a leading 'Okay, here is the rewritten tweet:' line that the LLM sometimes adds, keep the rest."""
    lines = [line.strip().strip('"') for line in str(text).splitlines() if line.strip()]
    if len(lines) > 1 and lines[0].lower().startswith(PREAMBLE_START) and lines[0].endswith(":"):
        lines = lines[1:]
    return " ".join(lines)


def main():
    train, validation, test = load_splits()
    pool_frame = pd.concat([train, validation], ignore_index=True)
    m = load_matrices([train, validation, test])
    para = pd.read_csv(WORK_DIR / "aug" / "paraphrases.csv", keep_default_na=False)
    para["text"] = para.text.map(clean_paraphrase)
    para = para[para.text.str.len() > 3].reset_index(drop=True)
    pm = {v: cached_encode(para.text.tolist(), v, WORK_DIR / "cache", tag="paraphrase") for v in SIX}
    position = {r: i for i, r in enumerate(pool_frame.row_id)}
    para_parent = para.row_id.map(position).to_numpy()
    y = pool_frame.Sentiment.to_numpy()
    pool = np.arange(len(pool_frame))
    cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1).split(pool, y))
    results = {}
    for copies in (0, 1, 2):
        for aug_weight in ((1.0,) if copies == 0 else (0.5, 1.0)):
            f1s = []
            for a, b in cv:
                keep = np.isin(para_parent, a) & (para["copy"].to_numpy() < copies)
                kf, ko = 0, 0
                for v in SIX:
                    fit = normalize(np.vstack([m[v][a], pm[v][keep]]))
                    held = normalize(m[v][b])
                    g = GAMMA_SCALE / (fit.shape[1] * normalize(m[v][a]).var())
                    kf, ko = kf + rbf(fit, fit, g) / 6, ko + rbf(held, fit, g) / 6
                labels = np.concatenate([y[a], para.Sentiment.to_numpy()[keep]])
                weights = np.concatenate([np.ones(len(a)), np.full(keep.sum(), aug_weight)])
                model = SVC(C=C, kernel="precomputed", class_weight="balanced").fit(kf, labels, sample_weight=weights)
                f1s.append(macro_f1(y[b], model.predict(ko)))
            key = f"copies={copies} weight={aug_weight}"
            results[key] = {"mean": float(np.mean(f1s)), "std": float(np.std(f1s))}
            print(f"{key:24s} {np.mean(f1s):.4f} +- {np.std(f1s):.4f}", flush=True)
    (EXPERIMENTS_DIR / "aug_cv.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
