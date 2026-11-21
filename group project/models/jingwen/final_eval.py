"""Final evaluation (the only script that reads test labels).

Every configuration was fixed beforehand on the 591-row pool; C is chosen here by the same pooled CV, before
any test score is computed. Models are fitted on the 473 training rows (the group's protocol), and the final
model is also refitted on all 591 rows as a second reference. Outputs follow the group's file format.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from data import LABEL_NAMES, PROCESSED_DIR, RESULTS_DIR, load_splits
from evaluate import macro_f1, metrics, paired_bootstrap, per_class_table, prediction_frame
from views import SIX_VIEWS, fused_kernel, load_views

C_GRID = (1, 2, 4, 8, 16, 32)
LADDER = {
    "L1 bge-large + SVM": ["bge_large"],
    "L2 e5-instruct sentiment view + SVM": ["e5i_sentiment"],
    "L3 three task views + multi-kernel SVM": ["e5i_sentiment", "e5i_topic", "e5i_style"],
    "L4 six views (prompt ensemble) + multi-kernel SVM": SIX_VIEWS,
}
FINAL = "L4 six views (prompt ensemble) + multi-kernel SVM"


def choose_c(m, views, y, pool):
    cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1).split(pool, y[pool]))
    kernels = [fused_kernel(m, views, pool[a], pool[b]) for a, b in cv]
    scores = {c: np.mean([macro_f1(y[pool[b]], SVC(C=c, kernel="precomputed", class_weight="balanced")
                                   .fit(kf, y[pool[a]]).predict(ko)) for (a, b), (kf, ko) in zip(cv, kernels)])
              for c in C_GRID}
    best = max(scores, key=scores.get)
    return best, float(scores[best])


def paraphrase_flags(fit_frame, test):
    """Test tweets whose word set has Jaccard >= 0.5 with some tweet the model was fitted on (leakage found at the
    interim): measured against the 473 training rows for those models and the 591 pool rows for the refit."""
    pool_sets = [set(t.split()) for t in fit_frame.text_clean]
    flags = []
    for t in test.text_clean:
        w = set(t.split())
        flags.append(max(len(w & s) / len(w | s) for s in pool_sets) >= 0.5)
    return np.array(flags)


def main():
    out = RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    train, validation, test = load_splits()
    m = load_views(sorted({v for views in LADDER.values() for v in views}), [train, validation, test])
    n_tr, n_va = len(train), len(validation)
    y = np.concatenate([train.Sentiment, validation.Sentiment, np.full(len(test), -1)])
    pool = np.arange(n_tr + n_va)
    tr_idx, va_idx, te_idx = np.arange(n_tr), np.arange(n_tr, n_tr + n_va), np.arange(n_tr + n_va, len(y))
    gold = test.Sentiment.to_numpy()
    leak = {473: paraphrase_flags(train, test), 591: paraphrase_flags(pd.concat([train, validation]), test)}

    tfidf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                          LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=7))
    predictions = {"L0 TF-IDF + LR (shared baseline)": tfidf.fit(train.text_clean, train.Sentiment).predict(test.text_clean)}
    val_scores = {"L0 TF-IDF + LR (shared baseline)": macro_f1(validation.Sentiment, tfidf.predict(validation.text_clean))}
    chosen = {}
    for name, views in LADDER.items():
        c, cv_score = choose_c(m, views, y, pool)
        chosen[name] = {"views": views, "C": c, "pool_cv_macro_f1": cv_score}
        kf, kv = fused_kernel(m, views, tr_idx, va_idx)
        _, kt = fused_kernel(m, views, tr_idx, te_idx)
        model = SVC(C=c, kernel="precomputed", class_weight="balanced").fit(kf, y[tr_idx])
        val_scores[name] = macro_f1(y[va_idx], model.predict(kv))
        predictions[name] = model.predict(kt)
    views, c = LADDER[FINAL], chosen[FINAL]["C"]
    kf, kt = fused_kernel(m, views, pool, te_idx)
    predictions["L4 refit on all 591 rows"] = SVC(C=c, kernel="precomputed", class_weight="balanced").fit(kf, y[pool]).predict(kt)
    base_591 = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                             LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=7))
    predictions["L0 TF-IDF + LR, 591 rows"] = base_591.fit(pd.concat([train, validation]).text_clean,
                                                           y[pool]).predict(test.text_clean)

    rows, per_class, boot = [], [], []
    for name, pred in predictions.items():
        n_fit = len(pool) if "591" in name else n_tr
        full, clean = metrics(gold, pred), metrics(gold[~leak[n_fit]], pred[~leak[n_fit]])
        rows.append({"model": name, "train_n": n_fit,
                     "val_macro_f1": val_scores.get(name), "test_accuracy": full["accuracy"],
                     "test_macro_f1": full["macro_f1"], "test_macro_f1_without_paraphrases": clean["macro_f1"],
                     **{f"{lab}_f1": full["per_class"][lab]["f1-score"] for lab in LABEL_NAMES}})
        per_class.append(per_class_table(gold, pred, name))
        key = name.split()[0] + ("_591" if "591" in name else "")
        (out / f"confusion_{key}.json").write_text(json.dumps(full["confusion_matrix"]))
        control = "L0 TF-IDF + LR, 591 rows" if "591" in name else "L0 TF-IDF + LR (shared baseline)"
        if not name.startswith("L0"):
            boot.append({"treatment": name, "control": control, **paired_bootstrap(gold, pred, predictions[control])})
    boot.append({"treatment": FINAL, "control": "L1 bge-large + SVM",
                 **paired_bootstrap(gold, predictions[FINAL], predictions["L1 bge-large + SVM"])})
    pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)
    pd.concat(per_class).to_csv(out / "per_class_metrics.csv", index=False)
    pd.DataFrame(boot).to_csv(out / "paired_bootstrap.csv", index=False)
    prediction_frame(test, predictions[FINAL]).assign(paraphrase_in_train=leak[n_tr]).to_csv(out / "test_predictions.csv", index=False)
    prediction_frame(test, predictions["L4 refit on all 591 rows"]).assign(paraphrase_in_pool=leak[len(pool)]).to_csv(
        out / "test_predictions_refit591.csv", index=False)
    pd.DataFrame({"row_id": test.row_id, "Sentiment": gold} | {k: v for k, v in predictions.items()}).to_csv(
        out / "all_test_predictions.csv", index=False)
    shutil.copy(PROCESSED_DIR / "split_manifest.csv", out / "split_manifest.csv")
    (out / "chosen_settings.json").write_text(json.dumps(chosen | {"gamma_scale": 2.0,
                                                                   "paraphrased_test_rows_vs_473": int(leak[n_tr].sum()),
                                                                   "paraphrased_test_rows_vs_591": int(leak[len(pool)].sum())},
                                                    indent=2))
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.precision", 4):
        print(pd.DataFrame(rows).drop(columns=["train_n"]))
        print(pd.DataFrame(boot)[["treatment", "control", "delta_macro_f1", "delta_ci_low", "delta_ci_high"]])
    print("paraphrased test rows vs 473 / 591:", int(leak[n_tr].sum()), int(leak[len(pool)].sum()))


if __name__ == "__main__":
    main()
