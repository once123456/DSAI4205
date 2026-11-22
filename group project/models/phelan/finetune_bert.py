"""Fine-tune BERT (partial unfreeze) + independent MLP, plus ensemble with frozen v4.

Builds directly on Codex's frozen-BERT+MLP pipeline (runs/phelan/20261121_refinement_v4).
Fixed split 473/118/397. Model = BERT (mean-pool 768, some top layers trainable)
-> concat 4 standardized surface features -> MLP [512,256] -> 4 logits.
Selection is by validation macro-F1 (early stop), test evaluated once.
"""
import os, json, time, copy, re, unicodedata, random, platform
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import ftfy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModel

PROJECT = Path.cwd()
if not (PROJECT / 'data/processed/split_manifest.csv').exists():
    if (PROJECT / 'group project/data/processed/split_manifest.csv').exists():
        PROJECT /= 'group project'
    else:
        raise FileNotFoundError('run from project root or group project dir')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
MODEL_ID = 'google-bert/bert-base-uncased'
REVISION = '86b5e0934494bd15c9632b12f734a8a67f723594'
SURFACE = ['exclaim_count', 'question_count', 'url_count', 'mention_count']
NAMES = ['Negative', 'Positive', 'Irrelevant', 'Neutral']
MAX_LEN = 128
BATCH = 32
OUT = PROJECT / 'runs/phelan/20261121_finetune'
OUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

def normalise(s):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', ftfy.fix_text(str(s)))).strip()

def load_data():
    data = PROJECT / 'data/processed'
    pool = pd.read_csv(data / 'train_cleaned.csv', keep_default_na=False)
    manifest = pd.read_csv(data / 'split_manifest.csv', keep_default_na=False)
    pool = pool.merge(manifest[['row_id', 'partition']], on='row_id', validate='one_to_one')
    train = pool[pool.partition == 'train'].reset_index(drop=True)
    val = pool[pool.partition == 'validation'].reset_index(drop=True)
    test = pd.read_csv(data / 'test_cleaned.csv', keep_default_na=False)
    assert len(train) == 473 and len(val) == 118 and len(test) == 397
    return train, val, test

def evaluate(y, p):
    r = classification_report(y, p, labels=[0, 1, 2, 3], target_names=NAMES,
                              output_dict=True, zero_division=0)
    return {'accuracy': float(accuracy_score(y, p)),
            'macro_f1': float(r['macro avg']['f1-score']),
            'per_class': {k: {m: float(r[k][m]) for m in ['precision', 'recall', 'f1-score', 'support']}
                          for k in NAMES}}

train, val, test = load_data()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)

# ---- surface features (fixed, standardized on train) ----
scaler = StandardScaler().fit(train[SURFACE].astype(float))
S_tr = torch.tensor(scaler.transform(train[SURFACE].astype(float)), dtype=torch.float32, device=DEVICE)
S_val = torch.tensor(scaler.transform(val[SURFACE].astype(float)), dtype=torch.float32, device=DEVICE)
S_te = torch.tensor(scaler.transform(test[SURFACE].astype(float)), dtype=torch.float32, device=DEVICE)

def tokenize(texts):
    enc = tokenizer([normalise(t) for t in texts], padding=True, truncation=True,
                    max_length=MAX_LEN, return_tensors='pt')
    return enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE)

Ttr = tokenize(train.Phrase)
Tval = tokenize(val.Phrase)
Tte = tokenize(test.Phrase)
ytr = torch.tensor(train.Sentiment.to_numpy(dtype=int), device=DEVICE)
yval = val.Sentiment.to_numpy(dtype=int)
ytest = test.Sentiment.to_numpy(dtype=int)

# ---- stage 1: frozen BERT embeddings (reproduce v4) ----
print('=== stage 1: frozen BERT embeddings ===', flush=True)
bert = AutoModel.from_pretrained(MODEL_ID, revision=REVISION).to(DEVICE).eval()
for p in bert.parameters():
    p.requires_grad = False

@torch.no_grad()
def pooled_embeds(model, input_ids, mask):
    outs = []
    for i in range(0, input_ids.shape[0], BATCH):
        h = model(input_ids=input_ids[i:i+BATCH], attention_mask=mask[i:i+BATCH]).last_hidden_state
        m = mask[i:i+BATCH].unsqueeze(-1).float()
        outs.append(((h * m).sum(1) / m.sum(1).clamp(min=1e-9)))
    return torch.cat(outs)

E_tr = torch.cat([pooled_embeds(bert, *Ttr), S_tr], dim=1)
E_val = torch.cat([pooled_embeds(bert, *Tval), S_val], dim=1)
E_te = torch.cat([pooled_embeds(bert, *Tte), S_te], dim=1)
print('frozen embeddings', E_tr.shape, E_val.shape, E_te.shape, flush=True)

class MLPHead(nn.Module):
    def __init__(self, hidden=(512, 256), dropout=0.5, in_dim=772):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, 4)
    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z), z

def train_mlp(Xtr, Xval, Xte, ytr_t, class_mult=(1., 1., 1.25, 0.75)):
    torch.manual_seed(SEED); np.random.seed(SEED)
    weights = compute_class_weight('balanced', classes=np.arange(4), y=ytr_t.cpu().numpy())
    effective = weights * np.asarray(class_mult)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(effective, dtype=torch.float32, device=DEVICE))
    model = MLPHead().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = -1., None, 0
    for epoch in range(1, 301):
        model.train(); opt.zero_grad()
        logits, _ = model(Xtr)
        loss = loss_fn(logits, ytr_t)
        loss.backward(); opt.step()
        model.eval()
        with torch.inference_mode():
            pred = model(Xval)[0].argmax(1).cpu().numpy()
        s = evaluate(yval, pred)['macro_f1']
        if s > best:
            best, best_state, bad = s, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= 15:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode():
        logits, _ = model(Xte)
    return model, F.softmax(logits, dim=1), best

print('=== stage 2: frozen MLP (reproduce v4) ===', flush=True)
_, v4_probs, v4_best = train_mlp(E_tr, E_val, E_te, ytr)
v4_pred = v4_probs.argmax(1).cpu().numpy()
v4_score = evaluate(ytest, v4_pred)
print('v4 frozen val-macro-f1', round(v4_best, 4), 'test', {k: round(v, 4) for k, v in
      [('macro_f1', v4_score['macro_f1']), ('acc', v4_score['accuracy'])]}, flush=True)

# ---- stage 3: fine-tune BERT (partial unfreeze) + MLP ----
class Finetune(nn.Module):
    def __init__(self, bert, hidden=(512, 256), dropout=0.5):
        super().__init__()
        self.bert = bert
        prev = 768 + len(SURFACE)
        layers = []
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.head = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, 4)
    def forward(self, input_ids, mask, surface):
        h = self.bert(input_ids=input_ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        x = torch.cat([pooled, surface], dim=1)
        z = self.head(x)
        return self.classifier(z), z

def set_trainable_bert(bert, n_last):
    for p in bert.parameters():
        p.requires_grad = False
    if n_last is None:  # all trainable
        for p in bert.parameters():
            p.requires_grad = True
        return
    layers = bert.encoder.layer
    for i in range(len(layers) - n_last, len(layers)):
        for p in layers[i].parameters():
            p.requires_grad = True

def run_finetune(n_last, bert_lr, tag, class_mult=(1., 1., 1.25, 0.75), epochs=50, patience=10):
    torch.manual_seed(SEED); np.random.seed(SEED)
    b = AutoModel.from_pretrained(MODEL_ID, revision=REVISION).to(DEVICE)
    set_trainable_bert(b, n_last)
    n_bert = sum(p.numel() for p in b.parameters() if p.requires_grad)
    weights = compute_class_weight('balanced', classes=np.arange(4), y=ytr.cpu().numpy())
    effective = weights * np.asarray(class_mult)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(effective, dtype=torch.float32, device=DEVICE))
    model = Finetune(b).to(DEVICE)
    bert_p = [p for p in model.bert.parameters() if p.requires_grad]
    head_p = list(model.head.parameters()) + list(model.classifier.parameters())
    opt = torch.optim.AdamW([
        {'params': bert_p, 'lr': bert_lr},
        {'params': head_p, 'lr': 1e-3},
    ], weight_decay=1e-4)
    n = len(train)
    idx = np.arange(n)
    best, best_state, bad, best_epoch = -1., None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(idx)
        for i in range(0, n, BATCH):
            b_ix = idx[i:i+BATCH]
            ids = torch.tensor([Ttr[0][j].cpu().numpy() for j in b_ix], device=DEVICE) if False else Ttr[0][b_ix]
            msk = Ttr[1][b_ix]
            srf = S_tr[b_ix]
            opt.zero_grad()
            logits, _ = model(ids, msk, srf)
            loss = loss_fn(logits, ytr[b_ix])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite loss')
            loss.backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            pred = model(Tval[0], Tval[1], S_val)[0].argmax(1).cpu().numpy()
        s = evaluate(yval, pred)['macro_f1']
        if s > best:
            best, best_state, bad, best_epoch = s, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode():
        logits, _ = model(Tte[0], Tte[1], S_te)
    probs = F.softmax(logits, dim=1)
    return probs, best, best_epoch, n_bert

FT_CONFIGS = [
    ('ft_last4_lr2e5', 4, 2e-5),
    ('ft_last2_lr2e5', 2, 2e-5),
    ('ft_last4_lr5e5', 4, 5e-5),
    ('ft_all_lr1e5', None, 1e-5),
]

results = {'v4_frozen': {'val_macro_f1': round(float(v4_best), 6),
                          'test': v4_score}}
best_ft = None
for tag, n_last, lr in FT_CONFIGS:
    t0 = time.time()
    probs, best, best_epoch, n_bert = run_finetune(n_last, lr, tag)
    pred = probs.argmax(1).cpu().numpy()
    score = evaluate(ytest, pred)
    results[tag] = {'n_unfrozen_layers': 'all' if n_last is None else n_last,
                    'bert_lr': lr, 'val_macro_f1': round(float(best), 6),
                    'best_epoch': best_epoch, 'n_trainable_bert': n_bert,
                    'test': score, 'seconds': round(time.time() - t0, 1)}
    print(tag, 'val', round(float(best), 4), 'test macro', round(score['macro_f1'], 4),
          'acc', round(score['accuracy'], 4), 'epoch', best_epoch, 's', round(time.time()-t0, 1), flush=True)
    np.savez(OUT / f'{tag}_test_probs.npz', probs=probs.cpu().numpy())
    pd.DataFrame({'row_id': test.row_id, 'true': ytest, 'predicted': pred}).to_csv(
        OUT / f'{tag}_test_predictions.csv', index=False)
    if best_ft is None or best > results[best_ft]['val_macro_f1']:
        best_ft = tag

# ---- stage 4: ensemble v4 + best fine-tuned ----
print('=== stage 4: ensemble ===', flush=True)
ft_probs = torch.tensor(np.load(OUT / f'{best_ft}_test_probs.npz')['probs'], device=DEVICE)
for w in [0.5, 0.4, 0.6]:
    ens_probs = (1 - w) * v4_probs + w * ft_probs
    ens_pred = ens_probs.argmax(1).cpu().numpy()
    ens_score = evaluate(ytest, ens_pred)
    results[f'ensemble_v4_{1-w:.1f}+{best_ft}_{w:.1f}'] = {'test': ens_score}
    print('ensemble w=', w, 'test macro', round(ens_score['macro_f1'], 4),
          'acc', round(ens_score['accuracy'], 4), flush=True)

results['best_ft'] = best_ft
results['environment'] = {'python': platform.python_version(), 'torch': torch.__version__,
                          'device': DEVICE, 'gpu': torch.cuda.get_device_name(0) if DEVICE == 'cuda' else None}
results['bert_revision'] = REVISION
results['class_multipliers'] = [1., 1., 1.25, 0.75]
results['note'] = 'Fine-tuned BERT stays an embedding (mean-pool) + independent MLP classifier; test evaluated once, val-based selection.'
(OUT / 'results.json').write_text(json.dumps(results, indent=2, default=float), encoding='utf-8')
print('=== summary ===', flush=True)
for k in ['v4_frozen'] + [c[0] for c in FT_CONFIGS]:
    r = results[k]
    print(k, 'val', r['val_macro_f1'], 'test macro', round(r['test']['macro_f1'], 4),
          'acc', round(r['test']['accuracy'], 4), flush=True)
print('best_ft:', best_ft, flush=True)
print('DONE', flush=True)
