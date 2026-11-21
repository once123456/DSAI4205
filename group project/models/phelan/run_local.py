"""Run Phelan's frozen-BERT + MLP notebook locally on CUDA.

Faithfully executes the same code cells as `notebooks/phelan/BERT_MLP_Sentiment.ipynb`,
skipping only the `%pip install` cell. Overrides the Colab `/content` output root with a
local `runs/phelan/` directory and points Hugging Face at a mirror (HF is blocked in
mainland China). Use `--smoke` to run only the smallest MLP config as a pipeline check.

Usage (from the `group project` directory):
  python models/phelan/run_local.py            # full grid -> official metrics
  python models/phelan/run_local.py --smoke    # quick pipeline validation
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import runpy
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]  # "group project"


class _FakeIPython:
    def run_line_magic(self, name, arg):
        print(f"[run_local] skip %{name} {arg}", flush=True)


def get_ipython():  # noqa: N802 — notebook install cell expects this name
    return _FakeIPython()


def _cell_source(cell) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run only the smallest MLP config")
    parser.add_argument("--runs-root", default=str(PROJECT / "runs" / "phelan"),
                        help="Where to write per-run output (default: runs/phelan)")
    args = parser.parse_args(argv)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # huggingface.co is unreachable here; route model downloads through the mirror.
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    notebook = runpy.run_path(str(HERE / "notebook.py"))["notebook"]

    ns = {
        "__name__": "__main__",
        "__file__": str(HERE / "notebook.py"),
        "get_ipython": get_ipython,
    }
    try:
        from IPython.display import display as _display
        ns["display"] = _display
    except Exception:
        ns["display"] = print

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        cid = cell.get("id", "")
        source = _cell_source(cell)
        if not source.strip():
            continue
        if cid == "install":
            print("[run_local] skip install", flush=True)
            continue
        if cid == "config":
            source = source.replace(
                'OUTPUT_ROOT = Path("/content/bert_mlp_runs")',
                f'OUTPUT_ROOT = Path({str(Path(args.runs_root).resolve())!r})',
            )
            if args.smoke:
                source = source.replace("SMOKE_TEST = False", "SMOKE_TEST = True")

        print(f"[run_local] RUN {cid}", flush=True)
        try:
            ast.parse(source)
            exec(compile(source, f"<{cid}>", "exec"), ns, ns)
        except Exception:
            traceback.print_exc()
            raise SystemExit(f"failed at cell {cid!r}")

    run_dir = ns.get("RUN_DIR")
    print("[run_local] done", flush=True)
    print("RUN_DIR:", run_dir, flush=True)
    if run_dir is not None and str(run_dir).startswith("/content"):
        raise SystemExit("BUG: output went to /content — local override did not apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
