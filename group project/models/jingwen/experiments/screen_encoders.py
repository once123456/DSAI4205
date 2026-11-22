"""Step 1: which frozen encoder gives the best validation macro-F1 under the same SVM grid?

Test labels are never read here.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the core modules
from data import EXPERIMENTS_DIR, load_splits, WORK_DIR
from encoders import ENCODERS, cached_encode
from evaluate import macro_f1


def svm_grid(x_train, y_train, x_val, y_val):
    best = None
    for scale in (False, True):
        for c in (0.5, 1, 2, 4, 8, 16, 32):
            steps = [StandardScaler()] if scale else []
            model = make_pipeline(*steps, SVC(C=c, kernel="rbf", gamma="scale", class_weight="balanced"))
            model.fit(x_train, y_train)
            score = macro_f1(y_val, model.predict(x_val))
            if best is None or score > best["val_macro_f1"]:
                best = {"val_macro_f1": score, "C": c, "standardize": scale}
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", nargs="+", default=list(ENCODERS))
    args = parser.parse_args()
    train, validation, test = load_splits()
    tfidf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                          LogisticRegression(C=1.0, max_iter=2000, random_state=7))
    tfidf.fit(train.text_clean, train.Sentiment)
    results = {"tfidf_lr_reference": {"val_macro_f1": macro_f1(validation.Sentiment, tfidf.predict(validation.text_clean))}}
    print("tfidf + lr (reference)", round(results["tfidf_lr_reference"]["val_macro_f1"], 4), flush=True)
    for name in args.encoders:
        vectors = {split: normalize(cached_encode(frame.text.tolist(), name, WORK_DIR / "cache"))
                   for split, frame in [("train", train), ("validation", validation), ("test", test)]}
        best = svm_grid(vectors["train"], train.Sentiment, vectors["validation"], validation.Sentiment)
        results[name] = best | {"dim": int(vectors["train"].shape[1])}
        print(name, json.dumps(results[name]), flush=True)
    out = EXPERIMENTS_DIR / "screen_encoders.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
