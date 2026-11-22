"""Write notebooks/jingwen/Jingwen_MultiView_SVM.ipynb from the core modules, so the notebook is self-contained and
always matches the code. Run from this folder: python build_notebook.py, then execute the notebook."""
import re
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "notebooks" / "jingwen" / "Jingwen_MultiView_SVM.ipynb"


def module(name):
    """Source of a core module without its docstring and its leading imports (the setup cell imports everything)."""
    text = (HERE / f"{name}.py").read_text()
    text = re.sub(r'^"""[\s\S]*?"""\n', "", text, count=1)
    lines = text.split("\n")
    while lines and (not lines[0].strip() or lines[0].startswith(("import ", "from "))):
        lines.pop(0)
    return "\n".join(lines).strip()


md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
cells = [
    md("""# Multi-View SVM (MV-SVM): instruction embeddings with a multi-kernel SVM (Jingwen Yang)

**Task.** Four-class tweet sentiment (Negative, Positive, Irrelevant, Neutral) on the group's fixed split:
473 training, 118 validation and 397 test tweets.

**Embedding model.** `intfloat/multilingual-e5-large-instruct`, an XLM-RoBERTa-large encoder (24 layers, 560M
parameters) trained for instruction-conditioned embeddings. It is kept frozen. Each tweet is encoded under six
task descriptions (sentiment, topic and style, two wordings each), which gives six 1024-dimensional views.

**Analytics model.** A multi-kernel support vector machine trained from scratch: one RBF kernel per view, the six
kernels averaged, balanced class weights. The encoder never sees a label and the SVM never changes the encoder, so
the two models are independent, as the course requires.

**Protocol.** Every choice is made by 5-fold cross-validation repeated three times on the 591-tweet training pool
(training plus validation). The test labels are read once, in the evaluation section."""),
    md("## 1. Setup\nVectors are cached on disk, so a second run takes seconds. Set `JINGWEN_WORK_DIR` to move the cache."),
    code("""import hashlib
import json
import os
import re
import shutil
import unicodedata
from pathlib import Path

import ftfy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from transformers import AutoModel, AutoTokenizer

pd.set_option("display.precision", 4)
pd.set_option("display.max_colwidth", 120)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU is slower but works)")"""),
    md("## 2. Data\nThe group's cleaned rows and fixed split. The model input is the original tweet with its broken "
       "encoding repaired; case, punctuation, emoji and negations are kept, as in the other neural models of the group."),
    code("# paths are resolved from the notebook folder, notebooks/jingwen/\n"
         + module("data").replace("Path(__file__).resolve().parents[2]", "Path.cwd().resolve().parents[1]")
         + "\n\ntrain, validation, test = load_splits()\nprint(len(train), len(validation), len(test))\n"
           "display(train.Sentiment.map(dict(enumerate(LABEL_NAMES))).value_counts())"),
    md("## 3. Embedding model: frozen encoders and instruction views\n`ENCODERS` maps a view name to the model, the "
       "pooling and the text prefix. The six final views share one encoder and differ only in the task description."),
    code(module("encoders")),
    md("## 4. Analytics model: multi-kernel SVM\nEach view gets an RBF kernel whose width follows sklearn's `scale` "
       "rule on the fitting rows (times 2, chosen by cross-validation); the kernels are averaged and passed to an SVM "
       "with a precomputed kernel."),
    code(module("views")),
    md("## 5. Metrics in the group's format"),
    code(module("evaluate")),
    md("## 6. Model selection and the single test evaluation\nThe ladder adds one idea at a time. For every rung, C is "
       "chosen by pooled cross-validation before any test score is computed. The final model is the six-view model; "
       "it is also refitted on all 591 labelled tweets (training plus validation, never test)."),
    code(module("final_eval").split("def main():")[0].strip()),
    code("\n".join(line[4:] for line in module("final_eval").split("def main():")[1].split("\n\nif __name__")[0].split("\n")[1:])),
    md("## 7. Figures"),
    code("""results = pd.read_csv(RESULTS_DIR / "comparison.csv")
fig, ax = plt.subplots(figsize=(6.5, 3))
order = results.model.tolist()
ax.barh(order[::-1], results.test_macro_f1[::-1], color=["#6DB1A3" if m.startswith("L4") else "#BDBDBD" if m.startswith("L0") else "#A9D5F0" for m in order[::-1]])
for i, v in enumerate(results.test_macro_f1[::-1]):
    ax.text(v + 0.004, i, f"{v:.3f}", va="center")
ax.set_xlim(0.3, 0.66)
ax.set_xlabel("Test macro-F1 (397 tweets)")
plt.show()

per = pd.read_csv(RESULTS_DIR / "per_class_metrics.csv")
display(per.pivot(index="model", columns="label", values="f1")[LABEL_NAMES])"""),
    md("The report figures (paper style, PDF and PNG) are produced by `figures.py` into `reports/jingwen/figures/`."),
    md("## 8. Experiment log\nEvery idea tried during development, with its pooled cross-validation macro-F1 "
       "(591 tweets, 5-fold x 3). The scripts are in `models/jingwen/experiments/`; their logs are read here, not rerun."),
    code("""log = EXPERIMENTS_DIR
pool = {**json.loads((log / "pool_cv.json").read_text()), **json.loads((log / "pool_cv_r2.json").read_text())}
tune = json.loads((log / "tune.json").read_text())
rows = [("bge-large, frozen", pool["bge"]["pool_cv_mean"], "single encoder"),
        ("e5-large, frozen", pool["e5"]["pool_cv_mean"], "single encoder"),
        ("+ entity-masked view", pool["bge+masked"]["pool_cv_mean"], "not adopted"),
        ("one instruction view", pool["e5i_sentiment"]["pool_cv_mean"], "single view"),
        ("sentiment + topic views", pool["e5i_sent+topic"]["pool_cv_mean"], "adopted"),
        ("sentiment + topic + style views", pool["sent+topic+style"]["pool_cv_mean"], "adopted"),
        ("concatenation instead of kernel sum", tune["A_fusion"]["concatenation, one RBF"]["mean"], "not adopted"),
        ("kernel product", tune["A_fusion"]["product of view kernels"]["mean"], "not adopted"),
        ("linear kernel", tune["E_kernel"]["linear"]["mean"], "not adopted"),
        ("+ general encoder view", tune["C_extra"]["+bge_large weight 1"]["mean"], "not adopted"),
        ("+ last-4-layer pooling", json.loads((log / "round1.json").read_text())["six, last-4 layers"]["mean"], "not adopted"),
        ("+ label-noise filter (10%)", json.loads((log / "noise.json").read_text())["0.1"]["mean"], "not adopted"),
        ("+ LLM paraphrases (1 copy)", json.loads((log / "aug_cv.json").read_text())["copies=1 weight=1.0"]["mean"], "not adopted (gain in only half of the folds)"),
        ("six views, prompt ensemble (final)", tune["D_wording"]["v1 and v2 kernels averaged"]["mean"], "final")]
display(pd.DataFrame(rows, columns=["idea", "pooled CV macro-F1", "decision"]))
families = json.loads((log / "families.json").read_text())
display(pd.DataFrame([(k, v["mean"], v["setting"]) for k, v in families.items()],
                     columns=["classifier on the same six views", "pooled CV macro-F1", "best setting"]))
# class weights against no weights over the C grid (six views), with the effect on the Irrelevant class
weights = pd.DataFrame(json.loads((log / "imbalance.json").read_text()))
display(weights[["class_weight", "C", "macro_f1", "Irrelevant_recall", "Irrelevant_f1", "Positive_recall"]])"""),
    md("## 9. Error analysis\nTweets the baseline got wrong and the final model got right, the most common remaining "
       "confusions, and training tweets whose label disagrees most with out-of-fold predictions (likely label noise)."),
    code("""preds = pd.read_csv(RESULTS_DIR / "all_test_predictions.csv").merge(test[["row_id", "text"]], on="row_id")
base, final = "L0 TF-IDF + LR (shared baseline)", "L4 six views (prompt ensemble) + multi-kernel SVM"
name = dict(enumerate(LABEL_NAMES))
fixed = preds[(preds[base] != preds.Sentiment) & (preds[final] == preds.Sentiment)]
print(f"fixed by the final model: {len(fixed)} tweets; broken by it: "
      f"{int(((preds[base] == preds.Sentiment) & (preds[final] != preds.Sentiment)).sum())}")
display(fixed.assign(label=fixed.Sentiment.map(name), baseline=fixed[base].map(name))[["row_id", "label", "baseline"]].head(8))
wrong = preds[preds[final] != preds.Sentiment]
display(pd.crosstab(wrong.Sentiment.map(name), wrong[final].map(name)).rename_axis(index="true", columns="predicted"))
display(pd.read_csv(EXPERIMENTS_DIR / "suspect_labels.csv").assign(label=lambda f: f.Sentiment.map(name))[["row_id", "label", "margin"]].head(8))"""),
    md("## 10. Baseline refinement: B0 to B1\nB1 keeps TF-IDF + logistic regression and tunes the input text, the n-grams, "
       "the minimum document frequency, C and the class weights with the same repeated 5-fold x 3 CV on the 591 labelled "
       "tweets (320 configurations). The search is `models/jingwen/baseline_b1.py`; its outputs are read here, not rerun."),
    code("""b1 = RESULTS_DIR.parent / "b1"
print(json.loads((b1 / "chosen.json").read_text()))
display(pd.read_csv(b1 / "cv_grid.csv").head(10))
display(pd.read_csv(b1 / "test_comparison.csv"))
display(pd.read_csv(b1 / "paired_bootstrap.csv"))"""),
]
nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python",
                                                                    "name": "python3"}})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, OUT)
print("wrote", OUT)
