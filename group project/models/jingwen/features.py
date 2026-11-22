"""Feature engineering beyond plain concatenation: masked views, whitening, prototype and neighbour
features, and kernels that can be fused with learned weights."""
import re

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize

MENTION_RE = re.compile(r"@\s*\w+")
URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[\w.-]+\.(?:com|org|net|ly|gg|tv)(?:/\S*)?", re.IGNORECASE)
NEGATIONS = {"no", "not", "never", "nothing", "nobody", "none", "nor", "cannot", "can't", "don't", "doesn't",
             "didn't", "isn't", "wasn't", "won't", "aren't", "haven't", "hasn't", "shouldn't", "wouldn't"}
SURFACE_COLUMNS = ["exclaim_count", "question_count", "caps_ratio", "url_count", "mention_count", "n_words",
                   "has_elongation"]


def mask_entities(text):
    """Replace accounts and links by placeholders, so the encoder sees the opinion, not the topic handle."""
    return re.sub(r"\s+", " ", URL_RE.sub("http", MENTION_RE.sub("@user", text))).strip()


def surface_features(frame):
    """Writing-style signals from the EDA (raw counts from the cleaning step plus a negation flag)."""
    values = frame[SURFACE_COLUMNS].astype(float).to_numpy()
    negation = frame.text.str.lower().str.split().map(lambda words: float(bool(NEGATIONS & set(words))))
    return np.column_stack([np.log1p(values), negation.to_numpy()])


class Whitener:
    """BERT-whitening: center, decorrelate and keep the top `dim` directions, fitted on training rows only."""

    def __init__(self, dim=128):
        self.dim = dim

    def fit(self, x):
        self.mean = x.mean(0, keepdims=True)
        u, s, _ = np.linalg.svd(np.cov((x - self.mean).T))
        w = u / np.sqrt(s + 1e-6)
        self.w = w[:, :self.dim] if self.dim else w
        return self

    def transform(self, x):
        return normalize((x - self.mean) @ self.w)


def prototype_knn(x_ref, y_ref, x, k=10):
    """Cosine similarity to each class mean, and the similarity-weighted label share of the k nearest rows."""
    x_ref, x = normalize(x_ref), normalize(x)
    centroids = normalize(np.stack([x_ref[y_ref == c].mean(0) for c in range(4)]))
    sims = x @ x_ref.T
    top = np.argsort(-sims, axis=1)[:, :k]
    weights = np.take_along_axis(sims, top, axis=1).clip(min=0)
    votes = np.zeros((len(x), 4))
    for c in range(4):
        votes[:, c] = (weights * (y_ref[top] == c)).sum(1)
    votes /= votes.sum(1, keepdims=True) + 1e-9
    return np.column_stack([x @ centroids.T, votes])


def prototype_knn_oof(x_train, y_train, others, k=10, folds=5, seed=0):
    """Out-of-fold version for the training rows, full-train version for the other splits."""
    y_train = np.asarray(y_train)
    train_feats = np.zeros((len(x_train), 8))
    for fit_idx, held_idx in StratifiedKFold(folds, shuffle=True, random_state=seed).split(x_train, y_train):
        train_feats[held_idx] = prototype_knn(x_train[fit_idx], y_train[fit_idx], x_train[held_idx], k)
    return train_feats, [prototype_knn(x_train, y_train, x, k) for x in others]


def scaled(train, others):
    scaler = StandardScaler().fit(train)
    return scaler.transform(train), [scaler.transform(x) for x in others]
