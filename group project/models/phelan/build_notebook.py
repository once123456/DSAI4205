"""Build Phelan's standalone Colab notebook (frozen BERT embedding + MLP)."""
import json
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def build():
    notebook = runpy.run_path(str(HERE / "notebook.py"))["notebook"]
    out = PROJECT / "notebooks/phelan/BERT_MLP_Sentiment.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    build()
