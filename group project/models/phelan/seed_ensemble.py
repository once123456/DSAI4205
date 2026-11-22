"""Multi-seed ensemble of the frozen BERT + MLP (v4 config) to reduce variance.

Caches frozen embeddings; trains the v4 MLP (class weight [1,1,1.25,0.75],
hidden [512,256], dropout 0.5) over seeds 42..46 and averages softmax.
"""
import os, json, re, unicodedata, random, copy, platform
from pathlib import Path
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
    PROJECT /= 'group project'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_ID = 'google-bert/bert-base-uncased'
REVISION = '86b5e0934494bd15c9632b12f734a8a67f723594'
SURFACE = ['exclaim_count', 'question_count', 'url_count', 'mention_count']
NAMES = ['Negative', 'Positive', 'Irrelevant', 'Neutral']
MAX_LEN = 128
BATCH = 32
SEEDS = [42, 43, 44, 45, 46]
CLASS_MULT = (1., 1., 1.25, 0.75)
OUT = PROJECT / 'runs/phelan/20261121_seed_ensemble'
OUT.mkdir(parents=True, exist_ok=True)
EMBED_CACHE = OUT / 'frozen_embeds.npz'

def normalise(s):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', ftfy.fix_text(str(s)))).strip()

data = PROJECT / 'data/processed'
pool = pd.read_csv(data / 'train_cleaned.csv', keep_default_na=False)
manifest = pd.read_csv(data / 'split_manifest.csv', keep_default_na=False)
pool = pool.merge(manifest[['row_id', 'partition']], on='row_id', validate='one_to_one')
train = pool[pool.partition == 'train'].reset_index(drop=True)
val = pool[pool.partition == 'validation'].reset_index(drop=True)
test = pd.read_csv(data / 'test_cleaned.csv', keep_default_na=False)
assert len(train) == 473 and len(val) == 118 and len(test) == 397

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
scaler = StandardScaler().fit(train[SURFACE].astype(float))
S_tr = torch.tensor(scaler.transform(train[SURFACE].astype(float)), dtype=torch.float32, device=DEVICE)
S_val = torch.tensor(scaler.transform(val[SURFACE].astype(float)), dtype=torch.float32, device=DEVICE)
S_te = torch.tensor(scaler.transform(test[SURFACE].astype(float)), dtype=torch.float32, device=DEVICE)

def tokenize(texts):
    enc = tokenizer([normalise(t) for t in texts], padding=True, truncation=True,
                    max_length=MAX_LEN, return_tensors='pt')
    return enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE)

Ttr = tokenize(train.Phrase); Tval = tokenize(val.Phrase); Tte = tokenize(test.Phrase)
ytr = torch.tensor(train.Sentiment.to_numpy(dtype=int), device=DEVICE)
yval = val.Sentiment.to_numpy(dtype=int)
ytest = test.Sentiment.to_numpy(dtype=int)

if EMBED_CACHE.exists():
    z = np.load(EMBED_CACHE)
    E_tr = torch.tensor(z['E_tr'], device=DEVICE); E_val = torch.tensor(z['E_val'], device=DEVICE)
    E_te = torch.tensor(z['E_te'], device=DEVICE)
    print('loaded cached embeddings', flush=True)
else:
    bert = AutoModel.from_pretrained(MODEL_ID, revision=REVISION).to(DEVICE).eval()
    for p in bert.parameters():
        p.requires_grad = False
    @torch.no_grad()
    def emb(input_ids, mask):
        outs = []
        for i in range(0, input_ids.shape[0], BATCH):
            h = bert(input_ids=input_ids[i:i+BATCH], attention_mask=mask[i:i+BATCH]).last_hidden_state
            m = mask[i:i+BATCH].unsqueeze(-1).float()
            outs.append((h * m).sum(1) / m.sum(1).clamp(min=1e-9))
        return torch.cat(outs)
    E_tr = torch.cat([emb(*Ttr), S_tr], dim=1)
    E_val = torch.cat([emb(*Tval), S_val], dim=1)
    E_te = torch.cat([emb(*Tte), S_te], dim=1)
    np.savez(EMBED_CACHE, E_tr=E_tr.cpu().numpy(), E_val=E_val.cpu().numpy(), E_te=E_te.cpu().numpy())
    print('computed embeddings', E_tr.shape, flush=True)
    del bert

class MLP(nn.Module):
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

def evaluate(y, p):
    r = classification_report(y, p, labels=[0, 1, 2, 3], target_names=NAMES, output_dict=True, zero_division=0)
    return {'accuracy': float(accuracy_score(y, p)), 'macro_f1': float(r['macro avg']['f1-score']),
            'per_class': {k: float(r[k]['f1-score']) for k in NAMES}}

weights = compute_class_weight('balanced', classes=np.arange(4), y=ytr.cpu().numpy()) * np.asarray(CLASS_MULT)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))

probs_all = []
per_seed = {}
for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = MLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = -1., None, 0
    for epoch in range(1, 301):
        model.train(); opt.zero_grad()
        logits, _ = model(E_tr)
        loss = loss_fn(logits, ytr)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(E_val)[0].argmax(1).cpu().numpy()
        s = evaluate(yval, pred)['macro_f1']
        if s > best:
            best, best_state, bad = s, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= 15:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        logits, _ = model(E_te)
    probs = F.softmax(logits, dim=1)
    pred = probs.argmax(1).cpu().numpy()
    sc = evaluate(ytest, pred)
    probs_all.append(probs)
    per_seed[seed] = {'val_macro_f1': round(float(best), 6), 'test_macro_f1': round(sc['macro_f1'], 6),
                      'test_acc': round(sc['accuracy'], 6), 'per_class': {k: round(v, 6) for k, v in sc['per_class'].items()}}
    print(f'seed {seed}: val {best:.4f} test macro {sc["macro_f1"]:.4f} acc {sc["accuracy"]:.4f}', flush=True)
    np.savez(OUT / f'seed{seed}_probs.npz', probs=probs.cpu().numpy())

ens = torch.stack(probs_all).mean(0)
ens_pred = ens.argmax(1).cpu().numpy()
ens_sc = evaluate(ytest, ens_pred)
print(f'ENSEMBLE ({len(SEEDS)} seeds): test macro {ens_sc["macro_f1"]:.4f} acc {ens_sc["accuracy"]:.4f}', flush=True)
print('per class:', {k: round(v, 4) for k, v in ens_sc['per_class'].items()}, flush=True)

result = {'seeds': SEEDS, 'class_multipliers': list(CLASS_MULT), 'per_seed': per_seed,
          'ensemble_test': ens_sc, 'environment': {'torch': torch.__version__, 'device': DEVICE}}
(OUT / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print('DONE', flush=True)
