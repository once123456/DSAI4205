from __future__ import annotations
"""Frozen official ELMo Small + trainable layer mix/projection/TextCNN, two CSVs only."""
import argparse
import hashlib
import itertools
import json
import re
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from ftfy import fix_encoding
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from elmo_vendor.runtime import _ElmoBiLm,batch_to_ids,remove_sentence_boundaries

PROJECT=Path(__file__).resolve().parents[2]
OUT=PROJECT/"runs/kun/elmo_textcnn"
RESOURCES=OUT/"resources"
TRAIN=PROJECT/"data/processed/train_cleaned.csv"
TEST=PROJECT/"data/processed/test_cleaned.csv"
LABELS=["Negative","Positive","Irrelevant","Neutral"]
MAX_LEN=64
SEEDS=[42,43,44]
FINAL_SEEDS=[101,102,103,104,105]
EPOCHS=[10,20]
CONFIGS=[dict(name="dropout05",projection=64,channels=32,kernels=[2,3,4],dropout=.5),
         dict(name="dropout06",projection=64,channels=32,kernels=[2,3,4],dropout=.6)]
TOKENIZER=TweetTokenizer(preserve_case=True,reduce_len=True)


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def metrics(y,p):
    return dict(accuracy=float(accuracy_score(y,p)),macro_f1=float(f1_score(y,p,labels=range(4),average="macro",zero_division=0)))
def tokenize(text):
    text=fix_encoding(str(text))
    text=re.sub(r"https?://\S+|www\.\S+"," URL ",text,flags=re.I)
    text=re.sub(r"(?<!\w)@\s*\w+"," USER ",text)
    return (TOKENIZER.tokenize(text) or ["EMPTY"])[:MAX_LEN]
def groups(frame): return frame.text_clean.str.lower().str.replace(r"\s+"," ",regex=True).str.strip()
def read_data(path):
    path=Path(path).resolve()
    if path not in {TRAIN.resolve(),TEST.resolve()}: raise ValueError("Only the two authorized CSV files are allowed")
    frame=pd.read_csv(path).fillna("")
    assert frame.row_id.is_unique and set(frame.Sentiment)==set(range(4))
    assert len(frame)==(591 if path==TRAIN.resolve() else 397)
    return frame


class FrozenELMo:
    def __init__(self):
        provenance=load(RESOURCES/"provenance.json")
        for name in ["options.json","weights.hdf5"]: assert sha(RESOURCES/name)==provenance[name]["sha256"]
        assert sha(Path(__file__).parent/"elmo_vendor/runtime.py")==provenance["runtime"]["sha256"]
        self.model=_ElmoBiLm(str(RESOURCES/"options.json"),str(RESOURCES/"weights.hdf5"),requires_grad=False).eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        assert self.model.get_output_dim()==256

    def encode(self,tokens,batch_size=8):
        result=np.zeros((len(tokens),3,MAX_LEN,256),dtype=np.float32)
        lengths=np.asarray([len(t) for t in tokens],dtype=np.int64)
        assert np.all((lengths>=1)&(lengths<=MAX_LEN))
        with torch.inference_mode():
            for start in range(0,len(tokens),batch_size):
                # No state may cross rows, batches, CV folds or train/test boundaries.
                self.model._elmo_lstm.reset_states()
                out=self.model(batch_to_ids(tokens[start:start+batch_size]))
                for layer,activation in enumerate(out["activations"]):
                    values,mask=remove_sentence_boundaries(activation,out["mask"])
                    values=values*mask.unsqueeze(-1)
                    result[start:start+len(values),layer,:values.shape[1]]=values.numpy()
        assert np.isfinite(result).all()
        return result,lengths


def feature_cache(frame,name):
    key=dict(source_sha256=sha(TRAIN if name=="train" else TEST),code_sha256=sha(__file__),
             encoder_sha256=sha(RESOURCES/"weights.hdf5"),runtime_sha256=sha(Path(__file__).parent/"elmo_vendor/runtime.py"))
    folder=OUT/"features";folder.mkdir(exist_ok=True)
    path=folder/f"{name}.npy";meta=folder/f"{name}.json";lp=folder/f"{name}_lengths.npy"
    if path.exists() and meta.exists():
        assert load(meta)==key
    else:
        encoder=FrozenELMo();tokens=[tokenize(t) for t in frame.Phrase]
        output=np.lib.format.open_memmap(path.with_suffix(".partial.npy"),mode="w+",dtype="float32",shape=(len(frame),3,MAX_LEN,256))
        lengths=np.zeros(len(frame),dtype=np.int64)
        for start in range(0,len(frame),32):
            values,lens=encoder.encode(tokens[start:start+32])
            output[start:start+len(values)]=values;lengths[start:start+len(values)]=lens
            print(f"ELMo {name}: {min(start+32,len(frame))}/{len(frame)}",flush=True)
        output.flush();del output,encoder
        path.with_suffix(".partial.npy").replace(path)
        np.save(lp,lengths);save(meta,key)
    return np.load(path,mmap_mode="r"),np.load(lp)


class TextCNN(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
        self.layer_logits=nn.Parameter(torch.zeros(3))
        self.gamma=nn.Parameter(torch.ones(()))
        self.normalization=nn.LayerNorm(256,elementwise_affine=False)
        self.projection=nn.Linear(256,config["projection"])
        self.convolutions=nn.ModuleList([nn.Conv1d(config["projection"],config["channels"],k) for k in config["kernels"]])
        self.dropout=nn.Dropout(config["dropout"])
        self.classifier=nn.Linear(2*config["channels"]*len(config["kernels"]),4)
    def forward(self,features,lengths):
        weights=self.layer_logits.softmax(0)
        x=(features*weights[None,:,None,None]).sum(1)
        x=torch.relu(self.projection(self.gamma*self.normalization(x)))
        mask=torch.arange(x.shape[1],device=x.device)[None,:]<lengths[:,None]
        x=(x*mask.unsqueeze(-1)).transpose(1,2)
        pooled=[]
        for kernel,conv in zip(self.config["kernels"],self.convolutions):
            a=conv(x).relu();count=(lengths-kernel+1).clamp(min=1)
            valid=torch.arange(a.shape[-1],device=x.device)[None,:]<count[:,None]
            pooled.extend([a.masked_fill(~valid[:,None,:],float("-inf")).max(-1).values,
                           (a*valid[:,None,:]).sum(-1)/count[:,None]])
        return self.classifier(self.dropout(torch.cat(pooled,dim=1)))


def probabilities(model,x,lengths):
    model.eval();result=[]
    with torch.inference_mode():
        for start in range(0,len(x),64):
            features=torch.from_numpy(np.array(x[start:start+64],copy=True))
            result.append(model(features,torch.from_numpy(lengths[start:start+64])).softmax(-1).numpy())
    return np.concatenate(result)


def fit(config,x,lengths,y,seed,epochs,validation=None):
    torch.manual_seed(seed);model=TextCNN(config)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    weights=torch.tensor(len(y)/(4*np.bincount(y,minlength=4)),dtype=torch.float32)
    loss_fn=nn.CrossEntropyLoss(weight=weights)
    generator=torch.Generator().manual_seed(seed);history=[];snapshots={}
    for epoch in range(1,epochs+1):
        model.train();total=0.
        for indices in torch.randperm(len(y),generator=generator).split(64):
            ix=indices.numpy();tx=torch.from_numpy(np.array(x[ix],copy=True));tl=torch.from_numpy(lengths[ix]);ty=torch.from_numpy(y[ix]).long()
            optimizer.zero_grad(set_to_none=True);loss=loss_fn(model(tx,tl),ty);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=float(loss.detach())*len(ix)
        history.append(dict(epoch=epoch,loss=total/len(y)))
        if validation is not None and epoch in EPOCHS:
            snapshots[epoch]=probabilities(model,validation[0],validation[1])
    return model.eval(),snapshots,history


def disagreement(p):
    pred=p.argmax(-1)
    return float(np.mean([np.mean(pred[i]!=pred[j]) for i,j in itertools.combinations(range(len(pred)),2)]))


def run_cv(frame,x,lengths,configs,split,epochs,stage):
    y=frame.Sentiment.to_numpy(dtype=np.int64);g=groups(frame);rows=[]
    for cfg in configs:
        oof={ep:np.zeros((len(SEEDS),len(y),4),np.float32) for ep in epochs};seen=np.zeros(len(y),int)
        for fold,(it,iv) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=split).split(frame,y,g)):
            assert not set(g.iloc[it])&set(g.iloc[iv]);seen[iv]+=1
            for si,seed in enumerate(SEEDS):
                path=OUT/stage/cfg["name"]/f"fold{fold}_seed{seed}.npz"
                if path.exists():
                    with np.load(path) as z:
                        np.testing.assert_array_equal(z["validation_indices"],iv)
                        snapshots={ep:z[f"epoch{ep}"] for ep in epochs}
                else:
                    start=time.perf_counter()
                    _,snapshots,history=fit(cfg,x[it],lengths[it],y[it],seed,max(epochs),(x[iv],lengths[iv]))
                    path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path,validation_indices=iv,**{f"epoch{ep}":snapshots[ep] for ep in epochs})
                    pd.DataFrame(history).to_csv(path.with_suffix(".csv"),index=False)
                    print(f"{stage} {cfg['name']} fold={fold+1}/5 seed={seed} {time.perf_counter()-start:.1f}s",flush=True)
                for ep in epochs: oof[ep][si,iv]=snapshots[ep]
            save(OUT/"status.json",dict(stage=stage,model=cfg["name"],completed_fold=fold+1,updated_utc=now()))
        assert np.all(seen==1)
        for ep,p in oof.items():
            seed_metrics=[metrics(y,v.argmax(1))["macro_f1"] for v in p]
            row=dict(model=cfg["name"],epochs=ep,split=split,**metrics(y,p.mean(0).argmax(1)),
                     seed_f1_sd=float(np.std(seed_metrics,ddof=1)),prediction_disagreement=disagreement(p))
            row["selection_score"]=row["macro_f1"]-.5*row["seed_f1_sd"]
            rows.append(row)
            np.savez_compressed(OUT/stage/cfg["name"]/f"oof_epoch{ep}.npz",probabilities=p,seeds=SEEDS)
        pd.DataFrame(rows).to_csv(OUT/stage/"metrics.csv",index=False)
    return pd.DataFrame(rows)


def register():
    OUT.mkdir(parents=True,exist_ok=True)
    record=dict(registered_utc=now(),source_hashes={"train":sha(TRAIN),"test":sha(TEST)},code_sha256=sha(__file__),
                runtime_sha256=sha(Path(__file__).parent/"elmo_vendor/runtime.py"),encoder_sha256=sha(RESOURCES/"weights.hdf5"),
                data_scope="Only train_cleaned.csv (591) and test_cleaned.csv (397). No other task dataset.",
                architecture="Official ELMo Small frozen (13,593,312 parameters), 3 layers x 256d; trainable scalar mix, token LayerNorm, 64d projection, kernels 2/3/4 x 32, max+mean pool, four classes.",
                configs=CONFIGS,epochs=EPOCHS,folds=5,development_split=42,check_split=2026,seeds=SEEDS,final_seeds=FINAL_SEEDS,
                selection="Maximize complete OOF ensemble Macro-F1 - 0.5 * seed F1 sample SD; ties prefer fewer epochs then config name. Training data only.",
                check="Frozen winner on another five-fold partition; no reselection.",
                final="Train five seeds on all 591 rows before parsing test CSV; average probabilities on all 397 test rows once.",
                encoder_isolation="Frozen encoder eval mode; recurrent states reset every batch. No fitting of features on validation/test text.",
                limits="Small repeated training splits are not independent datasets. Test file has historical evaluation exposure.")
    path=OUT/"protocol.json"
    if path.exists():
        old=load(path)
        for k in ["source_hashes","code_sha256","runtime_sha256","encoder_sha256"]: assert old[k]==record[k],k
    else: save(path,record)
    return record


def train_only():
    register()
    if (OUT/"train_completed.json").exists(): print(json.dumps(load(OUT/"selection.json"),indent=2));return
    data=read_data(TRAIN);x,lengths=feature_cache(data,"train")
    ranking=run_cv(data,x,lengths,CONFIGS,42,EPOCHS,"development").sort_values(["selection_score","epochs","model"],ascending=[False,True,True])
    ranking.to_csv(OUT/"selection_ranking.csv",index=False)
    best=ranking.iloc[0];cfg=next(c for c in CONFIGS if c["name"]==best.model)
    selection=dict(config=cfg,epochs=int(best.epochs),frozen_utc=now(),test_scored=False,statistics=best.to_dict())
    save(OUT/"selection.json",selection)
    run_cv(data,x,lengths,[cfg],2026,[int(best.epochs)],"check")
    save(OUT/"train_completed.json",dict(completed_utc=now(),test_scored=False))
    print(json.dumps(selection,indent=2),flush=True)


def finalize():
    protocol=register();assert (OUT/"train_completed.json").exists()
    if (OUT/"final/completed.json").exists(): print(json.dumps(load(OUT/"final/results.json"),indent=2));return
    selection=load(OUT/"selection.json");cfg=selection["config"];epochs=selection["epochs"]
    data=read_data(TRAIN);x,lengths=feature_cache(data,"train");y=data.Sentiment.to_numpy(dtype=np.int64)
    final=OUT/"final";final.mkdir(exist_ok=True);models=[]
    for seed in FINAL_SEEDS:
        path=final/f"seed{seed}.pt"
        if path.exists():
            record=torch.load(path,map_location="cpu",weights_only=True)
            model=TextCNN(cfg);model.load_state_dict(record["state_dict"])
        else:
            model,_,history=fit(cfg,x,lengths,y,seed,epochs)
            torch.save(dict(config=cfg,epochs=epochs,seed=seed,state_dict=model.state_dict(),encoder_sha256=protocol["encoder_sha256"]),path)
            pd.DataFrame(history).to_csv(path.with_suffix(".csv"),index=False)
        models.append(model);print(f"Full-train ELMo + TextCNN seed={seed} complete",flush=True)
    # First test CSV parsing in this experiment occurs after all choices and full fits.
    test=read_data(TEST);xt,lt=feature_cache(test,"test");yt=test.Sentiment.to_numpy(dtype=np.int64)
    all_p=[];rows=[];predictions=test[["row_id","Phrase","Sentiment","sentiment_label"]].copy()
    for seed,model in zip(FINAL_SEEDS,models):
        p=probabilities(model,xt,lt);all_p.append(p)
        record=torch.load(final/f"seed{seed}.pt",map_location="cpu",weights_only=True)
        check=TextCNN(record["config"]);check.load_state_dict(record["state_dict"])
        np.testing.assert_array_equal(p,probabilities(check,xt,lt))
        predictions[f"seed{seed}"]=p.argmax(1);rows.append(dict(seed=seed,**metrics(yt,p.argmax(1))))
    all_p=np.asarray(all_p);p=all_p.mean(0);predictions["prediction"]=p.argmax(1)
    for i,label in enumerate(LABELS): predictions[f"probability_{label}"]=p[:,i]
    np.savez_compressed(final/"probabilities.npz",probabilities=all_p,seeds=FINAL_SEEDS)
    predictions.to_csv(final/"test_predictions.csv",index=False);pd.DataFrame(rows).to_csv(final/"seed_metrics.csv",index=False)
    result=dict(model="ELMo Small + TextCNN",train_n=len(data),test_n=len(test),seeds=FINAL_SEEDS,epochs=epochs,
                **metrics(yt,p.argmax(1)),seed_f1_sd=float(np.std([r["macro_f1"] for r in rows],ddof=1)),prediction_disagreement=disagreement(all_p),
                trainable_parameters=sum(v.numel() for v in models[0].parameters()),frozen_encoder_parameters=13593312,
                layer_weights=[m.layer_logits.softmax(0).detach().tolist() for m in models])
    save(final/"results.json",result)
    save(final/"classification_report.json",classification_report(yt,p.argmax(1),labels=range(4),target_names=LABELS,output_dict=True,zero_division=0))
    pd.DataFrame(confusion_matrix(yt,p.argmax(1),labels=range(4)),index=LABELS,columns=LABELS).to_csv(final/"confusion_matrix.csv")
    assert protocol["source_hashes"]=={"train":sha(TRAIN),"test":sha(TEST)}
    save(final/"completed.json",dict(completed_utc=now(),test_scored=True,reloaded_probabilities_exact=True,source_hashes_unchanged=True))
    save(OUT/"status.json",dict(stage="trained_and_tested",updated_utc=now()))
    print(json.dumps(result,indent=2),flush=True)


def predict(texts):
    if not texts: return np.empty((0,4),np.float32)
    protocol=load(OUT/"protocol.json");assert protocol["code_sha256"]==sha(__file__)
    selection=load(OUT/"selection.json")
    encoder=FrozenELMo();x,lengths=encoder.encode([tokenize(t) for t in texts]);del encoder
    ps=[]
    for seed in FINAL_SEEDS:
        record=torch.load(OUT/"final"/f"seed{seed}.pt",map_location="cpu",weights_only=True)
        model=TextCNN(selection["config"]);model.load_state_dict(record["state_dict"])
        ps.append(probabilities(model,x,lengths))
    return np.mean(ps,axis=0)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--finalize",action="store_true");parser.add_argument("--predict",nargs="+");args=parser.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        if args.predict:
            p=predict(args.predict)
            print(json.dumps([dict(text=t,label=LABELS[int(v.argmax())],probabilities=dict(zip(LABELS,v.tolist()))) for t,v in zip(args.predict,p)],ensure_ascii=False,indent=2))
        elif args.finalize: finalize()
        else: train_only()
