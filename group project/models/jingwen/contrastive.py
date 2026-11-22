"""Step 2: adapt an encoder with supervised contrastive learning (SupCon), then freeze it.

Only the training rows are used for the loss. The epoch is chosen by the validation macro-F1 of an SVM on
the frozen vectors, so the encoder and the classifier stay two separate models.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

from encoders import ENCODERS, encode, load_encoder
from evaluate import macro_f1


def supcon_loss(z, y, temperature=0.07):
    z = F.normalize(z, dim=1)
    sim = z @ z.T / temperature
    eye = torch.eye(len(y), device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)
    positive = (y[:, None] == y[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    has_pos = positive.any(1)
    return -(log_prob * positive).sum(1)[has_pos].div(positive.sum(1)[has_pos]).mean()


def pooled(model, inputs, pooling):
    hidden = model(**inputs).last_hidden_state
    if pooling == "cls":
        return hidden[:, 0]
    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1)


def balanced_batches(labels, batch_size, rng):
    """Batches with roughly equal rows per class, so every anchor has positives and the rare class is seen."""
    by_class = [rng.permutation(np.flatnonzero(labels == c)) for c in range(4)]
    per_class = batch_size // 4
    steps = int(np.ceil(len(labels) / batch_size))
    for step in range(steps):
        idx = [np.take(group, range(step * per_class, (step + 1) * per_class), mode="wrap") for group in by_class]
        yield rng.permutation(np.concatenate(idx))


def svm_val_score(x_train, y_train, x_val, y_val):
    best = 0.0
    for c in (1, 4, 16):
        model = SVC(C=c, kernel="rbf", gamma="scale", class_weight="balanced").fit(x_train, y_train)
        best = max(best, macro_f1(y_val, model.predict(x_val)))
    return best


def train_contrastive(name, train_texts, train_labels, val_texts, val_labels, epochs=8, lr=2e-5,
                      batch_size=32, temperature=0.07, seed=42, max_length=192, device="cuda", log=print):
    """Return the adapted encoder (best epoch by validation), its tokenizer and the training log."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    _, pooling, prefix = ENCODERS[name]
    tokenizer, model = load_encoder(name, device)
    labels = np.asarray(train_labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history, best_state, best_score = [], None, -1.0

    def score_now():
        model.eval()
        xt = normalize(encode(train_texts, name, tokenizer, model, max_length=max_length))
        xv = normalize(encode(val_texts, name, tokenizer, model, max_length=max_length))
        return svm_val_score(xt, labels, xv, np.asarray(val_labels))

    start_score = score_now()
    history.append({"epoch": 0, "loss": None, "val_macro_f1": start_score})
    log(f"{name} seed {seed} epoch 0 val {start_score:.4f}")
    best_score, best_state = start_score, copy.deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for idx in balanced_batches(labels, batch_size, rng):
            batch = [prefix + train_texts[i] for i in idx]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = pooled(model, inputs, pooling)
            loss = supcon_loss(z.float(), torch.as_tensor(labels[idx], device=device), temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        score = score_now()
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_macro_f1": score})
        log(f"{name} seed {seed} epoch {epoch} loss {np.mean(losses):.4f} val {score:.4f}")
        if score > best_score:
            best_score, best_state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return tokenizer, model, history
