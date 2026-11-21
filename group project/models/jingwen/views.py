"""The final model's representation: instruction views of one frozen encoder, fused as an average of RBF kernels."""
import numpy as np
from sklearn.preprocessing import normalize

from data import WORK_DIR
from encoders import cached_encode

SIX_VIEWS = ["e5i_sentiment", "e5i_topic", "e5i_style", "e5i_sentiment_v2", "e5i_topic_v2", "e5i_style_v2"]
GAMMA_SCALE = 2.0


def load_views(names, frames):
    """Encode the texts of `frames` (in order) under every view in `names`; returns {view: (rows, dim)}."""
    texts = [t for frame in frames for t in frame.text]
    return {name: cached_encode(texts, name, WORK_DIR / "cache") for name in names}


def rbf(a, b, gamma):
    sq = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
    return np.exp(-gamma * np.maximum(sq, 0))


def fused_kernel(m, views, fit_idx, other_idx, gamma_scale=GAMMA_SCALE):
    """Average of one RBF kernel per view: (fit x fit, other x fit). Each width uses sklearn's 'scale' rule on the
    fit rows, times gamma_scale."""
    k_fit, k_other = 0, 0
    for view in views:
        a, b = normalize(m[view][fit_idx]), normalize(m[view][other_idx])
        gamma = gamma_scale / (a.shape[1] * a.var())
        k_fit = k_fit + rbf(a, a, gamma) / len(views)
        k_other = k_other + rbf(b, a, gamma) / len(views)
    return k_fit, k_other
