"""The interim report's data and baseline figures, redrawn in the same house style as figures.py.

Inputs are the group's existing tables and cleaned data (reports/*.csv, data/processed/*.csv) and the saved baseline
predictions; nothing is recomputed differently from the original scripts.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import TfidfVectorizer

from data import PROCESSED_DIR, PROJECT_DIR, RESULTS_DIR
from figures import BLUE, COLUMN, GRID, INK, TEXT, confusion, header, save, style

LABELS = ["Negative", "Positive", "Irrelevant", "Neutral"]
# shared figures use the blue family of the group report; the four classes are four distinct blues and a slate
CLASS_COLOR = {"Negative": BLUE["dark"], "Positive": BLUE["mid"], "Irrelevant": BLUE["light"], "Neutral": "#8C99A6"}
REPORTS = PROJECT_DIR / "reports"


def surface_features(train):
    features = [("exclaim_count", "!"), ("question_count", "?"), ("url_count", "URL"), ("mention_count", "@mention"),
                ("caps_ratio", "Caps"), ("has_elongation", "Elongated")]
    fig, ax = plt.subplots(figsize=(COLUMN, 1.5))
    for f, (col, _) in enumerate(features):
        for c, label in enumerate(LABELS):
            values = train.loc[train.sentiment_label == label, col]
            if col == "caps_ratio":
                share = (values >= 0.3).mean()
            elif col == "has_elongation":
                share = values.mean()
            else:
                share = (values > 0).mean()
            ax.bar(f + (c - 1.5) * 0.2, share, 0.19, color=CLASS_COLOR[label], label=label if f == 0 else None)
    ax.set_xticks(range(len(features)), [name for _, name in features])
    style(ax, "Share of tweets", (0, 0.45))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=8.5, handlelength=1.0)
    save(fig, "e1_surface_features")


def class_profile():
    profile = pd.read_csv(REPORTS / "class_profile.csv", encoding="utf-8-sig").set_index("sentiment").loc[LABELS]
    cols = [("mean_words", "Mean\nwords"), ("mean_exclaim", "Mean\n!"), ("mean_question", "Mean\n?"),
            ("share_url", "URL"), ("share_mention", "@"), ("share_negation", "Nega-\ntion"),
            ("share_game_word", "Game\nword"), ("share_brand_word", "Brand\nword")]
    values = profile[[c for c, _ in cols]].to_numpy(float)
    # shading is relative within each column (the columns have different units): lowest class light, highest dark
    rel = (values - values.min(0)) / (values.max(0) - values.min(0))
    cmap = LinearSegmentedColormap.from_list("b", ["#F1F5FA", BLUE["light"], BLUE["mid"], BLUE["dark"]])
    fig, ax = plt.subplots(figsize=(COLUMN, 1.4))
    ax.imshow(rel, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(rel.shape[0]):
        for j in range(rel.shape[1]):
            v = values[i, j]
            ax.text(j, i, f"{v:.1f}" if j == 0 else f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=8,
                    color="white" if rel[i, j] > 0.7 else INK)
    ax.set_xticks(range(len(cols)), [name for _, name in cols], fontsize=8)
    ax.set_yticks(range(4), LABELS)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    ax.spines[:].set_visible(False)
    save(fig, "e2_class_profile")


def distinctive_words():
    words = pd.read_csv(REPORTS / "distinctive_words.csv", encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(COLUMN, 2.55))
    for ax, label in zip(axes.ravel(), LABELS):
        rows = words[words.sentiment.eq(label)].sort_values("z_score")
        ax.barh(range(len(rows)), rows.z_score, color=CLASS_COLOR[label], height=0.7)
        ax.set_yticks(range(len(rows)), rows.word, fontsize=8)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 5)
        ax.set_xticks([0, 2, 4])
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        header(ax, label, "blue", height=0.14)
    for ax in axes[1]:
        ax.set_xlabel("Log-odds z-score")
    fig.subplots_adjust(hspace=0.6, wspace=0.55)
    save(fig, "e3_distinctive_words")


def tfidf_heatmap(train, test, docs=30, terms=40):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(train.text_clean)
    names = vectorizer.get_feature_names_out()
    # zero cells (most of a sparse TF-IDF matrix) in light grey; non-zero weights from a visible blue upward.
    # each panel keeps its own colour scale up to its largest weight, as the baseline section describes
    cmap = LinearSegmentedColormap.from_list("b", [BLUE["light"], BLUE["mid"], BLUE["dark"]])
    cmap.set_under("#EDEDED")
    fig, axes = plt.subplots(1, 2, figsize=(TEXT, 1.95))
    for ax, frame, title in zip(axes, [train, test], ["Training set", "Test set"]):
        matrix = vectorizer.transform(frame.text_clean[:docs]).toarray()
        # the 40 terms with the largest summed weight in this panel, kept in vocabulary order
        chosen = np.sort(np.argsort(matrix.sum(0))[-terms:])
        shown = matrix[:, chosen]
        image = ax.imshow(shown, cmap=cmap, vmin=1e-6, vmax=shown.max(), aspect="auto")
        ax.set_xticks(range(terms), names[chosen], rotation=90, fontsize=6.5)
        ax.set_yticks([0, 9, 19, docs - 1], ["1", "10", "20", str(docs)], fontsize=8)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
        header(ax, title, "blue", height=0.09)
        bar = fig.colorbar(image, ax=ax, extend="min", extendfrac=0.06, fraction=0.04, pad=0.02)
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Tweet")
    axes[1].figure.axes[-1].set_ylabel("TF-IDF weight")
    fig.subplots_adjust(wspace=0.22)
    save(fig, "e4_tfidf_heatmap")


def baseline_confusion():
    cm = json.loads((RESULTS_DIR / "confusion_L0_591.json").read_text())
    confusion(cm, "TF-IDF + LR, trained on 591 labelled tweets", BLUE, "blue", "e5_baseline_confusion")


def main():
    train = pd.read_csv(PROCESSED_DIR / "train_cleaned.csv", keep_default_na=False)
    test = pd.read_csv(PROCESSED_DIR / "test_cleaned.csv", keep_default_na=False)
    surface_features(train)
    class_profile()
    distinctive_words()
    tfidf_heatmap(train, test)
    baseline_confusion()
    print("done")


if __name__ == "__main__":
    main()
