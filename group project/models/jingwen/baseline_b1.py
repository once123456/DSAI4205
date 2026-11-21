"""B1: the tuned TF-IDF + logistic regression baseline.

B0 stays unchanged. B1 searches the vectorizer and the classifier of the same model family with the selection protocol
of MV-SVM: stratified five-fold cross-validation repeated three times (seed 1) on the 591 labelled tweets, macro F1,
every vectorizer fitted inside each fold. The test labels are read once, after the choice, for the final comparison.
"""
import itertools
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import FeatureUnion

from data import RESULTS_DIR, load_splits
from evaluate import macro_f1, metrics, paired_bootstrap, per_class_table
from final_eval import paraphrase_flags

OUT = RESULTS_DIR.parent / "b1"
INPUTS = ["text_clean", "text"]                   # the cleaned text of B0, or the lightly repaired original tweet
FEATURES = ["word 1", "word 1-2", "word 1-3", "char 2-5", "word 1-2 + char 2-5"]
MIN_DF = [1, 2]
C_GRID = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
WEIGHTS = [None, "balanced"]
B0 = {"input": "text_clean", "features": "word 1-2", "min_df": 2, "C": 1, "class_weight": "none"}


def vectorizer(features, min_df):
    word = {"word 1": (1, 1), "word 1-2": (1, 2), "word 1-3": (1, 3)}
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=min_df, sublinear_tf=True)
    if features in word:
        return TfidfVectorizer(ngram_range=word[features], min_df=min_df, sublinear_tf=True)
    if features == "char 2-5":
        return char
    return FeatureUnion([("word", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)), ("char", char)])


def classifier(c, weight):
    return LogisticRegression(C=c, class_weight=weight, solver="lbfgs", max_iter=2000, random_state=7)


def score_fold(texts, y, fit, held, features, min_df):
    """Macro F1 of every (C, class weight) on one fold; the vectorizer is fitted on the training part only."""
    vec = vectorizer(features, min_df)
    a, b = vec.fit_transform(texts[fit]), vec.transform(texts[held])
    return {(c, w): macro_f1(y[held], classifier(c, w).fit(a, y[fit]).predict(b)) for c, w in itertools.product(C_GRID, WEIGHTS)}


def fit_predict(config, fit_frame, test):
    vec = vectorizer(config["features"], config["min_df"])
    weight = None if config["class_weight"] == "none" else config["class_weight"]
    model = classifier(config["C"], weight).fit(vec.fit_transform(fit_frame[config["input"]]), fit_frame.Sentiment)
    return model.predict(vec.transform(test[config["input"]]))


def single_split_range(texts, y, seeds=20):
    """B0 macro F1 under single five-fold splits with different seeds, to place the interim single-split figure."""
    scores = []
    for seed in range(seeds):
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(texts, y)
        scores.append(np.mean([score_fold(texts, y, a, b, B0["features"], B0["min_df"])[(B0["C"], None)] for a, b in folds]))
    return {"mean": float(np.mean(scores)), "min": float(np.min(scores)), "max": float(np.max(scores))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, validation, test = load_splits()
    pool = pd.concat([train, validation], ignore_index=True)
    y = pool.Sentiment.to_numpy()
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1).split(pool, y))

    rows = []
    for text_col, features, min_df in itertools.product(INPUTS, FEATURES, MIN_DF):
        texts = pool[text_col].to_numpy()
        per_fold = Parallel(n_jobs=15)(delayed(score_fold)(texts, y, a, b, features, min_df) for a, b in folds)
        for (c, w) in per_fold[0]:
            scores = [fold[(c, w)] for fold in per_fold]
            rows.append({"input": text_col, "features": features, "min_df": min_df, "C": c,
                         "class_weight": w or "none", "cv_macro_f1": float(np.mean(scores)),
                         "cv_std": float(np.std(scores))})
        print(text_col, features, min_df, round(max(r["cv_macro_f1"] for r in rows[-len(per_fold[0]):]), 4), flush=True)
    grid = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)
    grid.to_csv(OUT / "cv_grid.csv", index=False)
    chosen = grid.iloc[0][["input", "features", "min_df", "C", "class_weight"]].to_dict()
    b0_cv = grid[(grid.input == B0["input"]) & (grid.features == B0["features"]) & (grid.min_df == B0["min_df"])
                 & (grid.C == B0["C"]) & (grid.class_weight == B0["class_weight"])].iloc[0]

    # the test set is read only from here on, after the configuration is fixed
    gold = test.Sentiment.to_numpy()
    results, per_class, predictions = [], [], {}
    for name, config in [("B0", B0), ("B1", chosen)]:
        for n_fit, fit_frame in [(591, pool), (473, train)]:
            pred = fit_predict(config, fit_frame, test)
            keep = ~paraphrase_flags(fit_frame, test)
            full, clean = metrics(gold, pred), metrics(gold[keep], pred[keep])
            report = pd.DataFrame(full["per_class"]).T
            results.append({"model": name, "train_n": n_fit, "accuracy": full["accuracy"],
                            "macro_precision": report["precision"].mean(), "macro_recall": report["recall"].mean(),
                            "macro_f1": full["macro_f1"], "macro_f1_without_paraphrases": clean["macro_f1"],
                            "irrelevant_recall": full["per_class"]["Irrelevant"]["recall"]})
            per_class.append(per_class_table(gold, pred, f"{name}, {n_fit} rows"))
            predictions[f"{name}, {n_fit} rows"] = pred
    boot = [{"train_n": n, **paired_bootstrap(gold, predictions[f"B1, {n} rows"], predictions[f"B0, {n} rows"])}
            for n in (591, 473)]
    pd.DataFrame(results).to_csv(OUT / "test_comparison.csv", index=False)
    pd.concat(per_class).to_csv(OUT / "per_class_metrics.csv", index=False)
    pd.DataFrame(boot).to_csv(OUT / "paired_bootstrap.csv", index=False)
    pd.DataFrame({"row_id": test.row_id, "Sentiment": gold} | predictions).to_csv(OUT / "test_predictions.csv", index=False)
    (OUT / "chosen.json").write_text(json.dumps({"B1": chosen, "B1_cv_macro_f1": float(grid.iloc[0].cv_macro_f1),
                                                 "B1_cv_std": float(grid.iloc[0].cv_std),
                                                 "B0_cv_macro_f1": float(b0_cv.cv_macro_f1),
                                                 "B0_cv_std": float(b0_cv.cv_std), "configurations": len(grid),
                                                 "B0_single_split_20_seeds": single_split_range(pool.text_clean.to_numpy(), y)},
                                                default=str, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 4):
        print(grid.head(10))
        print(pd.DataFrame(results))
        print(pd.DataFrame(boot))


if __name__ == "__main__":
    main()
