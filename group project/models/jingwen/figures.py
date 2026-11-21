"""Report figures for Jingwen's part, drawn from the saved results (no model is refitted here).

House style of the group's paper figures: Liberation Serif at the printed size (CVPR column 3.25 in, text 6.875 in),
8 pt text, flat panels, legend in one row above the axes, tinted header bands, direct labels on lines. All figures use
the blue ramp of the group report; the TF-IDF baseline is hatched grey. Each figure is saved as PDF and 300 dpi PNG.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle

from data import EXPERIMENTS_DIR as RUNS, FIGURES_DIR as FIG_DIR, RESULTS_DIR

COLUMN, TEXT = 3.25, 6.875
BLUE = {"light": "#A9C4DE", "mid": "#4A7BA7", "dark": "#1F3A5F"}          # blue ramp of the group report
ORANGE = "#FC8D62"
BASE, INK, MUTED, GRID = "#BDBDBD", "#222222", "#6b6f76", "#e6e6e6"
TINT = {"blue": "#E2EAF3", "grey": "#EBEBEB"}
LABELS = ["Negative", "Positive", "Irrelevant", "Neutral"]
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#333333", "xtick.color": "#333333", "ytick.color": "#333333", "axes.labelcolor": INK,
    "pdf.fonttype": 42, "legend.frameon": False, "legend.handlelength": 1.4, "legend.handletextpad": 0.45,
    "legend.columnspacing": 1.0, "hatch.linewidth": 0.6, "savefig.dpi": 300})


def header(ax, text, tint="blue", x0=0.0, x1=1.0, height=0.12):
    ax.add_patch(Rectangle((x0, 1.03), x1 - x0, height, transform=ax.transAxes, facecolor=TINT.get(tint, tint),
                           edgecolor="none", clip_on=False))
    ax.text((x0 + x1) / 2, 1.03 + height / 2, text, transform=ax.transAxes, ha="center", va="center", fontsize=9,
            fontweight="bold", color=INK)


def group_label(ax, x, text, y=-0.2):
    ax.text(x, y, text, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9, fontweight="bold",
            color=INK)


def bar(ax, x, v, color, width=0.7, hatch=None, label=None, value=True, digits=3):
    ax.bar(x, v, width, color="white" if hatch else color, edgecolor=MUTED if hatch else "none", hatch=hatch,
           linewidth=0.6, label=label)
    if value:
        ax.text(x, v + 0.006, f"{v:.{digits}f}".lstrip("0"), ha="center", va="bottom", fontsize=8, color=INK)


def style(ax, ylabel, ylim):
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def confusion(cm, title, ramp, tint, name):
    """Flat confusion matrix for one column: rows are true classes, cells show counts, shading is the row share."""
    cm = np.asarray(cm)
    share = cm / cm.sum(1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("c", ["#ffffff", ramp["light"], ramp["dark"]])
    fig, ax = plt.subplots(figsize=(COLUMN, 1.45))
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8.5, color="white" if share[i, j] > 0.5 else INK)
    ax.set_xticks(range(4), LABELS, fontsize=8.5)
    ax.set_yticks(range(4), LABELS, fontsize=8.5)
    ax.set_xlabel("Predicted", labelpad=2)
    ax.set_ylabel("True")
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    header(ax, title, tint, height=0.16)
    save(fig, name)


def ladder(comp):
    score = comp.set_index("model").test_macro_f1
    steps = [("TF-IDF\n+ LR", "L0 TF-IDF + LR (shared baseline)", None), ("bge", "L1 bge-large + SVM", BLUE["light"]),
             ("1\nview", "L2 e5-instruct sentiment view + SVM", BLUE["light"]),
             ("3\nviews", "L3 three task views + multi-kernel SVM", BLUE["mid"]),
             ("6\nviews", "L4 six views (prompt ensemble) + multi-kernel SVM", BLUE["dark"])]
    fig, ax = plt.subplots(figsize=(COLUMN, 1.6))
    for i, (_, model, color) in enumerate(steps):
        bar(ax, i, score[model], color, hatch="////" if color is None else None)
    bar(ax, 6, score["L0 TF-IDF + LR, 591 rows"], None, hatch="////")
    bar(ax, 7, score["L4 refit on all 591 rows"], BLUE["dark"])
    ax.set_xticks(list(range(5)) + [6, 7], [s[0] for s in steps] + ["TF-IDF\n+ LR", "6\nviews"], fontsize=8)
    ax.axvline(5, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    group_label(ax, 2, "473 training tweets", y=-0.34)
    group_label(ax, 6.5, "591 labelled tweets", y=-0.34)
    style(ax, "Test macro F1", (0.35, 0.64))
    ax.legend(handles=[Patch(facecolor="white", edgecolor=MUTED, hatch="////", label="TF-IDF + LR"),
                       Patch(color=BLUE["light"], label="single embedding"), Patch(color=BLUE["mid"], label="3 views"),
                       Patch(color=BLUE["dark"], label="MV-SVM (6 views)")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=8, handlelength=1.0, columnspacing=0.7)
    save(fig, "j1_ablation_ladder")


def selection(comp, settings):
    """(a) the same variants ranked by validation, repeated CV and test; (b, c) C and class weights in repeated CV."""
    fig, axes = plt.subplots(1, 3, figsize=(TEXT, 1.55), gridspec_kw={"width_ratios": [2.0, 1, 1]})
    rows = comp[comp.model.str.match(r"L[1-4] ") & ~comp.model.str.contains("591")]
    cv = [settings[m]["pool_cv_macro_f1"] for m in rows.model]
    x = np.arange(len(rows))
    ax = axes[0]
    for vals, color, marker, name, dy in [(rows.val_macro_f1.to_numpy(), "#9a9a9a", "s", "Validation (118)", -0.012),
                                          (np.array(cv), ORANGE, "o", "CV (591)", -0.004),
                                          (rows.test_macro_f1.to_numpy(), BLUE["dark"], "^", "Test (397)", 0.012)]:
        ax.plot(x, vals, marker=marker, color=color, lw=1.2, ms=3.5)
        ax.text(x[-1] + 0.15, vals[-1] + dy, name, color=color, fontsize=8, va="center")
    ax.set_xticks(x, ["bge", "1 view", "3 views", "6 views"], fontsize=8)
    ax.set_xlim(-0.3, 5.3)
    style(ax, "Macro F1", (0.45, 0.61))
    header(ax, "(a) Ranking of variants")
    weights = pd.DataFrame(json.loads((RUNS / "imbalance.json").read_text()))
    for ax, col, title, ylim in [(axes[1], "macro_f1", "(b) Macro F1", (0.43, 0.59)),
                                 (axes[2], "Irrelevant_recall", "(c) Irrelevant recall", (0, 0.45))]:
        for weight, color, name, xy in [("balanced", BLUE["dark"], "balanced", (2.4, 0.415)),
                                        ("none", "#9a9a9a", "no weights", (2.5, 0.15))]:
            part = weights[weights.class_weight.eq(weight)]
            ax.plot(np.log2(part.C.to_numpy()), part[col], marker="o", ms=3, lw=1.2, color=color)
            if col == "Irrelevant_recall":
                ax.text(*xy, name, color=color, fontsize=8, va="center")
        chosen = weights[weights.class_weight.eq("balanced") & weights.C.eq(4)][col].iloc[0]
        ax.plot(2, chosen, marker="o", ms=7, mfc="none", mec=ORANGE, mew=1.1)
        ax.set_xticks(range(6), ["1", "2", "4", "8", "16", "32"], fontsize=8.5)
        ax.set_xlabel("C", labelpad=1)
        style(ax, "", ylim)
        header(ax, title)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "j4_selection")


def main():
    comp = pd.read_csv(RESULTS_DIR / "comparison.csv")
    settings = json.loads((RESULTS_DIR / "chosen_settings.json").read_text())
    ladder(comp)
    confusion(json.loads((RESULTS_DIR / "confusion_L4_591.json").read_text()), "MV-SVM, trained on 591 labelled tweets",
              BLUE, "blue", "j3_confusion")
    selection(comp, settings)
    print("figures in", FIG_DIR)


if __name__ == "__main__":
    main()
