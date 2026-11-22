"""Shared data for Jingwen's models: the group's cleaned rows, fixed split and light text normalization."""
import hashlib
import os
import re
import unicodedata
from pathlib import Path

import ftfy
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "runs" / "jingwen" / "results"            # final evaluation (committed)
EXPERIMENTS_DIR = PROJECT_DIR / "runs" / "jingwen" / "experiments"    # cross-validation logs (committed)
FIGURES_DIR = PROJECT_DIR / "reports" / "jingwen" / "figures"
# vector caches, LLM paraphrases and contrastive checkpoints (git-ignored); can be moved with JINGWEN_WORK_DIR
WORK_DIR = Path(os.environ.get("JINGWEN_WORK_DIR", PROJECT_DIR / "runs" / "jingwen" / "work"))
LABEL_NAMES = ["Negative", "Positive", "Irrelevant", "Neutral"]
LABEL_IDS = list(range(len(LABEL_NAMES)))


def normalize_text(text):
    # same light repair as the group's neural models: fix broken encoding, keep case, punctuation and emoji
    text = str(text).replace("\u00e2\u0080?", "'")
    text = ftfy.fix_text(text)
    text = re.sub(r"([A-Za-z])\s*\ufffd\s*(ve|re|ll|t|s|m|d)\b", r"\1'\2", text)
    text = unicodedata.normalize("NFC", text).replace("\ufffd", " ")
    return re.sub(r"\s+", " ", text).strip()


def load_splits():
    """Return train (473), validation (118) and test (397) frames with a `text` column."""
    pool = pd.read_csv(PROCESSED_DIR / "train_cleaned.csv", keep_default_na=False)
    test = pd.read_csv(PROCESSED_DIR / "test_cleaned.csv", keep_default_na=False)
    split = pd.read_csv(PROCESSED_DIR / "split_manifest.csv").set_index("row_id")
    pool["partition"] = pool.row_id.map(split.partition)
    if pool.partition.isna().any() or not pool.Sentiment.eq(pool.row_id.map(split.Sentiment)).all():
        raise ValueError("train_cleaned.csv and split_manifest.csv do not match")
    for frame in (pool, test):
        frame["text"] = frame.Phrase.map(normalize_text)
    train = pool[pool.partition.eq("train")].reset_index(drop=True)
    validation = pool[pool.partition.eq("validation")].reset_index(drop=True)
    return train, validation, test.reset_index(drop=True)


def text_fingerprint(texts):
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]
