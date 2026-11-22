"""Two report diagrams in the same house style as figures.py: the TF-IDF + LR baseline workflow (shared, blue) and the
MV-SVM architecture (ours), both in the blue ramp of the group report."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from figures import BLUE, COLUMN, INK, MUTED, TEXT, TINT, save


def box(ax, x, y, w, h, title, body, face, edge, title_size=7.5, body_size=6.5):
    """Rounded box; the title sits on the top line and the body lines below it."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", facecolor=face, edgecolor=edge,
                                linewidth=0.8))
    if body:
        ax.text(x + w / 2, y + h - 0.07, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.27, body, ha="center", va="top", fontsize=body_size, color=INK, linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=INK)


def arrow(ax, start, end, color):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7, color=color, lw=0.8,
                                 shrinkA=0, shrinkB=0))


def canvas(width, height):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def baseline_workflow():
    fig, ax = canvas(COLUMN, 1.6)
    face, edge = TINT["blue"], BLUE["dark"]
    w, h, gap = 1.0, 0.72, 0.12
    top, bottom = 0.86, 0.02
    col = [0.0, w + gap, 2 * (w + gap)]
    box(ax, col[0], top, w, h, "Tweets", "591 train,\n397 test", face, edge, body_size=7.5)
    box(ax, col[1], top, w, h, "Cleaning", "ftfy, lowercase,\nlemmatize", face, edge, body_size=7.5)
    box(ax, col[2], top, w, h, "TF-IDF", "1-2 grams,\n1951 features", face, edge, body_size=7.5)
    box(ax, col[2], bottom, w, h, "Classifier", "logistic\nregression, C = 1", face, edge, body_size=7.5)
    box(ax, col[1], bottom, w, h, "Evaluation", "accuracy, P, R,\nF1 on test", face, edge, body_size=7.5)
    for x in col[:2]:
        arrow(ax, (x + w, top + h / 2), (x + w + gap, top + h / 2), edge)
    arrow(ax, (col[2] + w / 2, top), (col[2] + w / 2, bottom + h), edge)
    arrow(ax, (col[2], bottom + h / 2), (col[1] + w, bottom + h / 2), edge)
    save(fig, "e0_baseline_workflow")


def strip(ax, x, y, shades, cell=0.1, height=0.13, edge=BLUE["mid"]):
    """A short embedding vector drawn as a row of shaded cells."""
    for k, shade in enumerate(shades):
        ax.add_patch(Rectangle((x + k * cell, y), cell, height, facecolor=shade, edgecolor="white", linewidth=0.6))
    ax.add_patch(Rectangle((x, y), cell * len(shades), height, facecolor="none", edgecolor=edge, linewidth=0.6))


def matrix(ax, x, y, size, values, cmap):
    """A small kernel matrix icon: values in [0, 1] drawn as an n x n grid."""
    n = len(values)
    cell = size / n
    for r in range(n):
        for c in range(n):
            ax.add_patch(Rectangle((x + c * cell, y + (n - 1 - r) * cell), cell, cell, facecolor=cmap(values[r][c]),
                                   edgecolor="white", linewidth=0.4))
    ax.add_patch(Rectangle((x, y), size, size, facecolor="none", edgecolor=BLUE["mid"], linewidth=0.6))


def mv_svm():
    fig, ax = canvas(TEXT, 1.95)
    edge, face = BLUE["dark"], TINT["blue"]
    cmap = LinearSegmentedColormap.from_list("k", ["#ffffff", BLUE["light"], BLUE["dark"]])
    rng = np.random.default_rng(3)
    rows = [1.47, 1.23, 0.99, 0.75, 0.51, 0.27]
    names = ["sentiment 1", "sentiment 2", "topic 1", "topic 2", "style 1", "style 2"]
    heads = [(0.3, "Tweet"), (1.2, "Instructions"), (2.22, "Encoder"), (3.15, "Views $\\mathbf{e}_v(x)$"),
             (3.88, "Kernels"), (4.6, "Average $K$"), (5.37, "SVM"), (6.4, "Classes")]
    for x, text in heads:
        ax.text(x, 1.82, text, ha="center", va="center", fontsize=8.5, fontweight="bold", color=INK)
    box(ax, 0.0, 0.72, 0.6, 0.42, "Tweet $x$", "", "white", MUTED, title_size=9)
    for y, name in zip(rows, names):
        ax.add_patch(FancyBboxPatch((0.8, y - 0.09), 0.8, 0.18, boxstyle="round,pad=0,rounding_size=0.03",
                                    facecolor=face, edgecolor=edge, linewidth=0.6))
        ax.text(1.2, y, name, ha="center", va="center", fontsize=8, color=INK)
        arrow(ax, (0.6, 0.93), (0.8, y), MUTED)
        arrow(ax, (1.6, y), (1.85, y), edge)
        arrow(ax, (2.6, y), (2.84, y), edge)
        strip(ax, 2.85, y - 0.065, [cmap(v) for v in rng.uniform(0.15, 0.9, 6)])
        arrow(ax, (3.45, y), (3.77, y), edge)
        matrix(ax, 3.78, y - 0.1, 0.2, np.clip(np.eye(4) + rng.uniform(0, 0.6, (4, 4)), 0, 1), cmap)
        arrow(ax, (4.0, y), (4.34, 0.87), MUTED)
    box(ax, 1.85, 0.18, 0.75, 1.38, "e5-instruct", "560M\nparameters,\nfrozen", "#F2F2F2",
        MUTED, title_size=8.5, body_size=8)
    avg = np.clip(np.eye(5) * 0.8 + rng.uniform(0, 0.45, (5, 5)), 0, 1)
    matrix(ax, 4.35, 0.62, 0.5, (avg + avg.T) / 2, cmap)
    arrow(ax, (4.86, 0.87), (5.08, 0.87), edge)
    box(ax, 5.08, 0.45, 0.58, 0.84, "SVM", "$C = 4$,\nclass\nweights", face, edge, title_size=9, body_size=8)
    for k, label in enumerate(["Negative", "Positive", "Irrelevant", "Neutral"]):
        y = 1.4 - k * 0.33
        box(ax, 5.95, y - 0.12, 0.9, 0.24, label, "", "white", edge, title_size=8.5)
        arrow(ax, (5.66, 0.87), (5.95, y), MUTED)
    ax.text(2.22, 0.03, "pretrained, not updated", ha="center", fontsize=8, color=MUTED, style="italic")
    ax.text(5.37, 0.03, "fitted on the 591 labelled tweets", ha="center", fontsize=8, color=edge, style="italic")
    save(fig, "j0_architecture")


if __name__ == "__main__":
    baseline_workflow()
    mv_svm()
    print("done")
