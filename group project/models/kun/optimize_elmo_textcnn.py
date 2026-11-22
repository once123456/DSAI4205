from __future__ import annotations
"""Train-only representation/head search within frozen ELMo Small + TextCNN."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report,confusion_matrix
from threadpoolctl import threadpool_limits
import elmo_textcnn as base

OUT=base.OUT/"optimization_v1"
DEFAULT=dict(projection=128,channels=32,kernels=[2,3,4],dropout=.5,normalize=False,projection_relu=False,context_only=False,mean_skip=False,direct=False)
CONFIGS=[dict(name="control",**{**DEFAULT,"projection":64,"dropout":.6,"normalize":True,"projection_relu":True}),
         dict(name="linear128",**{**DEFAULT,"normalize":True,"dropout":.6}),
         dict(name="raw128",**DEFAULT),
         dict(name="direct256",**{**DEFAULT,"projection":256,"channels":16,"direct":True}),
         dict(name="contextual128",**{**DEFAULT,"context_only":True}),
         dict(name="mean_skip128",**{**DEFAULT,"mean_skip":True})]
EPOCHS=[10,20]
SEEDS=[42,43,44]
FINAL_SEEDS=[101,102,103,104,105]


class Variant(base.TextCNN):
    def __init__(self,config):
        super().__init__(config)
        if not config["normalize"]: self.normalization=nn.Identity()
        if config["direct"]: self.projection=nn.Identity()
        if config["mean_skip"]: self.classifier=nn.Linear(2*config["channels"]*len(config["kernels"])+config["projection"],4)
    def forward(self,features,lengths):
        logits=self.layer_logits
        if self.config["context_only"]: logits=logits.masked_fill(torch.arange(3,device=logits.device)==0,float("-inf"))
        weights=logits.softmax(0)
        x=(features*weights[None,:,None,None]).sum(1)
        x=self.projection(self.gamma*self.normalization(x))
        if self.config["projection_relu"]: x=x.relu()
        mask=torch.arange(x.shape[1],device=x.device)[None,:]<lengths[:,None]
        x=x*mask.unsqueeze(-1);pooled=[]
        if self.config["mean_skip"]: pooled.append(x.sum(1)/lengths[:,None])
        x=x.transpose(1,2)
        for kernel,conv in zip(self.config["kernels"],self.convolutions):
            a=conv(x).relu();count=(lengths-kernel+1).clamp(min=1)
            valid=torch.arange(a.shape[-1],device=x.device)[None,:]<count[:,None]
            pooled.extend([a.masked_fill(~valid[:,None,:],float("-inf")).max(-1).values,
                           (a*valid[:,None,:]).sum(-1)/count[:,None]])
        return self.classifier(self.dropout(torch.cat(pooled,1)))


def make_model(config):
    return base.TextCNN(config) if config["name"]=="control" else Variant(config)


def fit(config,x,lengths,y,seed,epochs,validation=None):
    torch.manual_seed(seed);model=make_model(config)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(len(y)/(4*np.bincount(y,minlength=4)),dtype=torch.float32))
    generator=torch.Generator().manual_seed(seed);history=[];snapshots={}
    for epoch in range(1,epochs+1):
        model.train();total=0.
        for indices in torch.randperm(len(y),generator=generator).split(64):
            ix=indices.numpy()
            # Advanced indexing already produces an independent writable array.
            tx=torch.from_numpy(x[ix]);tl=torch.from_numpy(lengths[ix]);ty=torch.from_numpy(y[ix]).long()
            optimizer.zero_grad(set_to_none=True);loss=loss_fn(model(tx,tl),ty)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=float(loss.detach())*len(ix)
        history.append(dict(epoch=epoch,loss=total/len(y)))
        if validation is not None and epoch in EPOCHS:
            snapshots[epoch]=base.probabilities(model,validation[0],validation[1])
    return model.eval(),snapshots,history


def register():
    parent=base.load(base.OUT/"protocol.json")
    assert parent["code_sha256"]==base.sha(base.__file__)
    record=dict(registered_utc=base.now(),code_sha256=base.sha(__file__),base_sha256=base.sha(base.__file__),
                source_hashes={"train":base.sha(base.TRAIN),"test":base.sha(base.TEST)},
                encoder_sha256=base.sha(base.RESOURCES/"weights.hdf5"),configs=CONFIGS,epochs=EPOCHS,seeds=SEEDS,final_seeds=FINAL_SEEDS,
                scope="Only the two authorized CSV files; same frozen ELMo Small features, no archived schemes or additional task data.",
                screening="Five-fold split 42, initialization seed 42, epochs 10 and 20. Promote two distinct noncontrol configurations, each at its best screened epoch, by complete OOF Macro-F1; ties prefer fewer epochs then name.",
                confirmation="Promoted pairs and fixed control (20 epochs), five-fold split 42 with seeds 42/43/44. Then same candidates on five-fold split 2026 with identical seeds.",
                selection="Use both training partitions: mean ensemble F1 - .5 * mean seed F1 SD. Candidate eligible only if mean F1 >= control + .003 and neither partition F1 is below control by more than .005. Highest eligible score must exceed control; otherwise retain control.",
                final="Freeze before test. Use five fixed seeds on all 591 rows. Evaluate all 397 test rows once and compare with the saved equal-budget ELMo control, never select from test.",
                cache="Reuse original control predictions only after matching source/code hashes and fold indices. Reuse screening trajectories for confirmation when epoch and seed match.",
                limits="Both partitions are development data; this is exploratory selection, not independent confirmation. The test file has historical exposure.")
    assert parent["source_hashes"]==record["source_hashes"]
    path=OUT/"protocol.json"
    if path.exists():
        old=base.load(path)
        for key in ["code_sha256","base_sha256","source_hashes","encoder_sha256"]: assert old[key]==record[key],key
    else: base.save(path,record)
    return record


def run_cv(data,x,lengths,pairs,split,seeds,stage):
    y=data.Sentiment.to_numpy(dtype=np.int64);groups=base.groups(data);rows=[]
    for cfg,epochs in pairs:
        name=cfg["name"];oof={ep:np.zeros((len(seeds),len(y),4),np.float32) for ep in epochs};seen=np.zeros(len(y),int)
        for fold,(it,iv) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=split).split(data,y,groups)):
            assert not set(groups.iloc[it])&set(groups.iloc[iv]);seen[iv]+=1
            for si,seed in enumerate(seeds):
                path=OUT/"cache"/name/f"split{split}"/f"fold{fold}_seed{seed}.npz"
                source=None
                if name=="control":
                    parent_stage="development" if split==42 else "check"
                    source=base.OUT/parent_stage/"dropout06"/f"fold{fold}_seed{seed}.npz"
                available={}
                if path.exists():
                    with np.load(path) as z:
                        np.testing.assert_array_equal(z["validation_indices"],iv)
                        available={ep:z[f"epoch{ep}"] for ep in epochs if f"epoch{ep}" in z.files}
                elif source is not None:
                    with np.load(source) as z:
                        np.testing.assert_array_equal(z["validation_indices"],iv)
                        available={ep:z[f"epoch{ep}"] for ep in epochs}
                    path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path,validation_indices=iv,**{f"epoch{k}":v for k,v in available.items()})
                    base.save(path.with_suffix(".json"),dict(reused_from=str(source),source_sha256=base.sha(source)))
                if len(available)!=len(epochs):
                    start=time.perf_counter()
                    _,snapshots,history=fit(cfg,x[it],lengths[it],y[it],seed,max(epochs),(x[iv],lengths[iv]))
                    for ep in set(available)&set(snapshots): np.testing.assert_array_equal(available[ep],snapshots[ep])
                    path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path,validation_indices=iv,**{f"epoch{k}":v for k,v in snapshots.items()})
                    pd.DataFrame(history).to_csv(path.with_suffix(".csv"),index=False)
                    available=snapshots
                    print(f"{stage} {name} split={split} fold={fold+1}/5 seed={seed}: {time.perf_counter()-start:.1f}s",flush=True)
                for ep in epochs: oof[ep][si,iv]=available[ep]
            base.save(OUT/"status.json",dict(stage=stage,model=name,split=split,completed_fold=fold+1,updated_utc=base.now()))
        assert np.all(seen==1)
        for ep,p in oof.items():
            scores=[base.metrics(y,v.argmax(1))["macro_f1"] for v in p]
            row=dict(model=name,epochs=ep,split=split,seeds=len(seeds),**base.metrics(y,p.mean(0).argmax(1)),
                     seed_f1_sd=float(np.std(scores,ddof=1)) if len(scores)>1 else 0.,
                     prediction_disagreement=base.disagreement(p) if len(scores)>1 else 0.)
            rows.append(row);target=OUT/stage/name/f"oof_epoch{ep}.npz";target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target,probabilities=p,seeds=seeds)
            print(f"{stage} {name} epoch={ep}: OOF F1={row['macro_f1']:.4f}, seed SD={row['seed_f1_sd']:.4f}",flush=True)
        pd.DataFrame(rows).to_csv(OUT/stage/"metrics.csv",index=False)
    return pd.DataFrame(rows)


def train_only():
    register()
    if (OUT/"train_completed.json").exists(): print(json.dumps(base.load(OUT/"selection.json"),indent=2));return
    data=base.read_data(base.TRAIN);x,lengths=base.feature_cache(data,"train")
    screen=run_cv(data,x,lengths,[(c,EPOCHS) for c in CONFIGS],42,[42],"screen")
    ranked=screen[screen.model.ne("control")].sort_values(["macro_f1","epochs","model"],ascending=[False,True,True])
    # Promote distinct model configurations, each at its best screened epoch.
    nominated=ranked.drop_duplicates("model").head(2)
    nominees=[(next(c for c in CONFIGS if c["name"]==row.model),[int(row.epochs)]) for row in nominated.itertuples()]
    base.save(OUT/"nomination.json",dict(frozen_utc=base.now(),candidates=[dict(config=c,epochs=e[0]) for c,e in nominees],test_scored=False))
    pairs=[(CONFIGS[0],[20])]+nominees
    first=run_cv(data,x,lengths,pairs,42,SEEDS,"confirmation")
    second=run_cv(data,x,lengths,pairs,2026,SEEDS,"transfer")
    combined=pd.concat([first,second],ignore_index=True);combined.to_csv(OUT/"development_metrics.csv",index=False)
    summary=combined.groupby(["model","epochs"]).agg(mean_f1=("macro_f1","mean"),split_sd=("macro_f1","std"),mean_seed_sd=("seed_f1_sd","mean"),disagreement=("prediction_disagreement","mean")).reset_index()
    control=summary[summary.model.eq("control")].iloc[0];control_splits=combined[combined.model.eq("control")].set_index("split").macro_f1
    summary["eligible"]=False
    for i,row in summary.iterrows():
        by_split=combined[(combined.model==row.model)&(combined.epochs==row.epochs)].set_index("split").macro_f1
        summary.loc[i,"eligible"]=row.model=="control" or (row.mean_f1>=control.mean_f1+.003 and bool((by_split>=control_splits-.005).all()))
    summary["score"]=summary.mean_f1-.5*summary.mean_seed_sd
    ranking=summary.sort_values(["score","epochs","model"],ascending=[False,True,True]);ranking.to_csv(OUT/"selection_ranking.csv",index=False)
    best=ranking[ranking.eligible].iloc[0];ctrl=ranking[ranking.model.eq("control")].iloc[0]
    if best.score<=ctrl.score: best=ctrl
    cfg=next(c for c in CONFIGS if c["name"]==best.model)
    selection=dict(config=cfg,epochs=int(best.epochs),statistics=best.to_dict(),frozen_utc=base.now(),test_scored=False)
    base.save(OUT/"selection.json",selection);base.save(OUT/"train_completed.json",dict(completed_utc=base.now(),test_scored=False))
    print(json.dumps(selection,indent=2),flush=True)


def finalize():
    protocol=register();assert (OUT/"train_completed.json").exists()
    if (OUT/"final/completed.json").exists(): print(json.dumps(base.load(OUT/"final/results.json"),indent=2));return
    selection=base.load(OUT/"selection.json");cfg=selection["config"];epochs=selection["epochs"]
    data=base.read_data(base.TRAIN);x,lengths=base.feature_cache(data,"train");y=data.Sentiment.to_numpy(dtype=np.int64)
    folder=OUT/"final";folder.mkdir(exist_ok=True);models=[]
    for seed in FINAL_SEEDS:
        path=folder/f"seed{seed}.pt"
        if path.exists():
            record=torch.load(path,weights_only=True,map_location="cpu");model=make_model(cfg);model.load_state_dict(record["state_dict"])
        elif cfg["name"]=="control":
            record=torch.load(base.OUT/"final"/f"seed{seed}.pt",weights_only=True,map_location="cpu")
            model=make_model(cfg);model.load_state_dict(record["state_dict"])
            torch.save(dict(config=cfg,epochs=epochs,seed=seed,state_dict=model.state_dict()),path)
        else:
            model,_,history=fit(cfg,x,lengths,y,seed,epochs)
            torch.save(dict(config=cfg,epochs=epochs,seed=seed,state_dict=model.state_dict()),path)
            pd.DataFrame(history).to_csv(path.with_suffix(".csv"),index=False)
        models.append(model);print(f"Full-train {cfg['name']} seed={seed} ready",flush=True)
    test=base.read_data(base.TEST);xt,lt=base.feature_cache(test,"test");yt=test.Sentiment.to_numpy(dtype=np.int64)
    all_p=[];rows=[];preds=test[["row_id","Phrase","Sentiment","sentiment_label"]].copy()
    for seed,model in zip(FINAL_SEEDS,models):
        p=base.probabilities(model,xt,lt);all_p.append(p)
        other=make_model(cfg);other.load_state_dict(torch.load(folder/f"seed{seed}.pt",weights_only=True)["state_dict"])
        np.testing.assert_array_equal(p,base.probabilities(other,xt,lt))
        rows.append(dict(seed=seed,**base.metrics(yt,p.argmax(1))));preds[f"seed{seed}"]=p.argmax(1)
    all_p=np.asarray(all_p);p=all_p.mean(0);preds["prediction"]=p.argmax(1)
    for i,label in enumerate(base.LABELS): preds[f"probability_{label}"]=p[:,i]
    with np.load(base.OUT/"final/probabilities.npz") as z:
        np.testing.assert_array_equal(z["seeds"],FINAL_SEEDS);control_p=z["probabilities"]
    comparisons=[]
    for name,probs in [("ELMo control",control_p),(cfg["name"],all_p)]:
        scores=[base.metrics(yt,v.argmax(1))["macro_f1"] for v in probs]
        comparisons.append(dict(model=name,**base.metrics(yt,probs.mean(0).argmax(1)),seed_f1_sd=float(np.std(scores,ddof=1)),disagreement=base.disagreement(probs)))
    np.savez_compressed(folder/"probabilities.npz",probabilities=all_p,seeds=FINAL_SEEDS)
    preds.to_csv(folder/"test_predictions.csv",index=False);pd.DataFrame(rows).to_csv(folder/"seed_metrics.csv",index=False)
    pd.DataFrame(comparisons).to_csv(folder/"comparison.csv",index=False)
    base.save(folder/"classification_report.json",classification_report(yt,p.argmax(1),labels=range(4),target_names=base.LABELS,output_dict=True,zero_division=0))
    pd.DataFrame(confusion_matrix(yt,p.argmax(1),labels=range(4)),index=base.LABELS,columns=base.LABELS).to_csv(folder/"confusion_matrix.csv")
    result=dict(model=cfg["name"],config=cfg,epochs=epochs,train_n=591,test_n=397,seeds=FINAL_SEEDS,
                **base.metrics(yt,p.argmax(1)),trainable_parameters=sum(v.numel() for v in models[0].parameters()),comparison=comparisons)
    base.save(folder/"results.json",result)
    assert protocol["source_hashes"]=={"train":base.sha(base.TRAIN),"test":base.sha(base.TEST)}
    base.save(folder/"completed.json",dict(completed_utc=base.now(),test_scored=True,source_hashes_unchanged=True,reloaded_probabilities_exact=True))
    print(json.dumps(result,indent=2),flush=True)


def predict(texts):
    if not texts: return np.empty((0,4),np.float32)
    protocol=base.load(OUT/"protocol.json");assert protocol["code_sha256"]==base.sha(__file__)
    assert protocol["base_sha256"]==base.sha(base.__file__)
    cfg=base.load(OUT/"selection.json")["config"]
    encoder=base.FrozenELMo();x,lengths=encoder.encode([base.tokenize(t) for t in texts]);del encoder
    ps=[]
    for seed in FINAL_SEEDS:
        model=make_model(cfg);model.load_state_dict(torch.load(OUT/"final"/f"seed{seed}.pt",map_location="cpu",weights_only=True)["state_dict"])
        ps.append(base.probabilities(model,x,lengths))
    return np.mean(ps,axis=0)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--finalize",action="store_true");parser.add_argument("--predict",nargs="+");args=parser.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        if args.predict:
            p=predict(args.predict)
            print(json.dumps([dict(text=t,label=base.LABELS[int(v.argmax())],probabilities=dict(zip(base.LABELS,v.tolist()))) for t,v in zip(args.predict,p)],ensure_ascii=False,indent=2))
        elif args.finalize: finalize()
        else: train_only()
