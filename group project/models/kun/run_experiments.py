"""A/B/C/D training extracted from the contributed report. See README for inputs."""
def main():
    from pathlib import Path
    import sys, json, hashlib, time, itertools, datetime
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score

    P=Path(__file__).resolve().parents[2]
    O=P/'runs/kun/abcd_reproduction'
    O.mkdir(parents=True,exist_ok=True)
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            name=str(args[0]).replace('\\','/').lower()
            if '/group project/' in name and ('test_cleaned' in name or '/features/test' in name or '/elmo_development/' in name):
                raise RuntimeError('Forbidden evaluation artifact access: '+name)
    sys.addaudithook(guard)
    sys.path.insert(0,str(P/'models/kun'))
    import optimize_elmo_textcnn as old
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    LABELS=['Negative','Positive','Irrelevant','Neutral'];SEEDS=[101,102,103,104,105]
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def save(p,d):
        p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2),encoding='utf8')
    data=pd.read_csv(P/'data/processed/train_cleaned.csv');assert len(data)==591
    cache=P/'runs/kun/elmo_textcnn/features'
    meta=json.loads((cache/'train.json').read_text())
    assert sha(P/'data/processed/train_cleaned.csv')==meta['source_sha256']
    assert sha(P/'models/kun/elmo_textcnn.py')==meta['code_sha256']
    assert sha(P/'models/kun/elmo_vendor/runtime.py')==meta['runtime_sha256']
    assert sha(P/'runs/kun/elmo_textcnn/resources/weights.hdf5')==meta['encoder_sha256']
    x=np.load(cache/'train.npy');lengths=np.load(cache/'train_lengths.npy');y=data.Sentiment.to_numpy(dtype=np.int64)
    assert x.shape==(591,3,64,256) and np.isfinite(x).all()
    # Connected components of training-only word-set Jaccard links.
    parent=list(range(len(data)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    sets=[set(str(t).split()) for t in data.text_clean];edges=[]
    for i in range(len(data)):
        for j in range(i):
            a,b=sets[i],sets[j];sim=len(a&b)/len(a|b) if a|b else 0
            if sim>=.5:parent[root(i)]=root(j);edges.append((i,j,sim))
    groups=np.array([root(i) for i in range(len(data))])
    outer=list(StratifiedGroupKFold(5,shuffle=True,random_state=20261121).split(x,y,groups))
    splits=[];folds=np.full(len(y),-1)
    for f,(it,iv) in enumerate(outer):
        assert not set(groups[it])&set(groups[iv]);assert set(y[it])==set(range(4)) and set(y[iv])==set(range(4));folds[iv]=f
        inner=[]
        for j,(a,b) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=7200+f).split(x[it],y[it],groups[it])):
            a,b=it[a],it[b];assert not set(groups[a])&set(groups[b]);assert set(y[a])==set(range(4)) and set(y[b])==set(range(4))
            inner.append({'train':a.tolist(),'validation':b.tolist()})
        splits.append({'train':it.tolist(),'evaluation':iv.tolist(),'inner':inner})
    cfg=dict(name='mean_skip128',projection=128,channels=32,kernels=[2,3,4],dropout=.5,normalize=False,projection_relu=False,context_only=False,mean_skip=True,direct=False)
    protocol=dict(training_sha256=meta['source_sha256'],script_sha256=sha(__file__),model_source_sha256=sha(P/'models/kun/optimize_elmo_textcnn.py'),
        source_cache_sha256=sha(cache/'train.npy'),configuration=cfg,seeds=SEEDS,epochs=10,optimizer='AdamW lr=.001 weight_decay=.01',batch_size=64,gradient_clip=1,
        outer_folds=5,inner_folds=3,group_rule='Training-only whitespace word-set Jaccard >= .5, connected components',groups=int(len(set(groups))),near_duplicate_edges=len(edges),
        variants={'A':'full global mean head, weighted CE, no offsets','B':'A without global mean branch only','C':'A with unweighted CE only','D':'same outer A probabilities, offsets selected using inner OOF only'},
        selection='Architecture and 10 epochs fixed before current outer evaluation; inner folds select only D offsets. No test/evaluation file read. Historical architecture choices had prior development exposure; retrospective CV, not pristine new test.',
        offset_grid=[-.6,-.4,-.2,0,.2,.4,.6],fine_grid=[-.1,-.05,0,.05,.1],offset_tie_order='macro F1 descending, accuracy descending, squared norm ascending, offsets lexicographic',
        total_head_fits=150,splits=splits)
    if (O/'protocol.json').exists():assert json.loads((O/'protocol.json').read_text())==protocol,'Protocol changed; refuse mixed resume'
    else:save(O/'protocol.json',protocol)
    pd.DataFrame({'row_id':data.row_id,'label':y,'near_duplicate_group':groups,'outer_fold':folds}).to_csv(O/'split_manifest.csv',index=False)
    pd.DataFrame(edges,columns=['row_i','row_j','jaccard']).to_csv(O/'near_duplicate_edges.csv',index=False)

    def fit_predict(tag,variant,it,iv,seed):
        dest=O/'fits'/tag;dest.mkdir(parents=True,exist_ok=True)
        path=dest/f'seed{seed}.npz';ck=dest/f'seed{seed}.pt'
        if path.exists():
            z=np.load(path);np.testing.assert_array_equal(z['evaluation_indices'],iv);np.testing.assert_array_equal(z['training_indices'],it);return z['probabilities']
        start=time.perf_counter();torch.manual_seed(seed)
        config={**cfg,'mean_skip':variant!='B'};model=old.make_model(config)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        weight=None if variant=='C' else torch.tensor(len(it)/(4*np.bincount(y[it],minlength=4)),dtype=torch.float32)
        loss_fn=torch.nn.CrossEntropyLoss(weight=weight);gen=torch.Generator().manual_seed(seed);history=[]
        for ep in range(10):
            model.train();total=0
            for local in torch.randperm(len(it),generator=gen).split(64):
                ix=it[local.numpy()];opt.zero_grad(set_to_none=True)
                loss=loss_fn(model(torch.from_numpy(x[ix]),torch.from_numpy(lengths[ix])),torch.from_numpy(y[ix]))
                assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=loss.item()*len(ix)
            history.append({'epoch':ep+1,'loss':total/len(it)})
        model.eval();prob=old.base.probabilities(model,x[iv],lengths[iv])
        assert np.isfinite(prob).all();np.testing.assert_allclose(prob.sum(1),1,atol=1e-6)
        torch.save({'state_dict':model.state_dict(),'config':config,'seed':seed,'variant':variant,'epochs':10,'training_ids':data.iloc[it].row_id.tolist()},ck)
        # Verify saved state reload on a real batch before checkpoint completion.
        other=old.make_model(config);other.load_state_dict(torch.load(ck,weights_only=True)['state_dict'])
        np.testing.assert_array_equal(prob[:min(64,len(iv))],old.base.probabilities(other,x[iv[:64]],lengths[iv[:64]]))
        pd.DataFrame(history).to_csv(dest/f'seed{seed}_learning.csv',index=False)
        np.savez_compressed(path,probabilities=prob,training_indices=it,evaluation_indices=iv)
        print(f'{tag} seed={seed} saved ({time.perf_counter()-start:.1f}s)',flush=True)
        return prob

    def score(yt,p):return {'accuracy':float(accuracy_score(yt,p)), 'macro_f1':float(f1_score(yt,p,average='macro',labels=range(4),zero_division=0))}
    def bias_search(prob,yt):
        log=np.log(prob.astype(float).clip(1e-9));rows=[]
        def evaluate(grid,stage):
            for b in grid:
                s=score(yt,(log+b).argmax(1));rows.append({'stage':stage,'b0':b[0],'b1':b[1],'b2':b[2],'b3':b[3],'norm':float(np.dot(b,b)),**s})
        def rank():return pd.DataFrame(rows).sort_values(['macro_f1','accuracy','norm','b1','b2','b3'],ascending=[False,False,True,True,True,True])
        evaluate([np.array([0,*b]) for b in itertools.product(protocol['offset_grid'],repeat=3)],'coarse')
        center=rank().iloc[0][['b0','b1','b2','b3']].to_numpy(float)
        evaluate([center+np.array([0,*b]) for b in itertools.product(protocol['fine_grid'],repeat=3)],'fine')
        return rank().iloc[0][['b0','b1','b2','b3']].to_numpy(float),pd.DataFrame(rows)

    oof={k:np.zeros((len(y),4)) for k in 'ABCD'};seed_oof={k:np.zeros((5,len(y),4)) for k in 'ABC'};stats=[]
    for f,split in enumerate(splits):
        it=np.array(split['train']);iv=np.array(split['evaluation'])
        # Select offsets before fitting/evaluating outer heads; no outer labels here.
        ip=np.zeros((5,len(y),4));seen=np.zeros(len(y),int)
        for j,inside in enumerate(split['inner']):
            a=np.array(inside['train']);b=np.array(inside['validation']);seen[b]+=1
            for si,seed in enumerate(SEEDS):ip[si,b]=fit_predict(f'outer{f+1}/inner{j+1}/A','A',a,b,seed)
        assert (seen[it]==1).all() and (seen[iv]==0).all()
        offsets,trace=bias_search(ip.mean(0)[it],y[it]);trace.to_csv(O/f'outer{f+1}_bias_search.csv',index=False)
        save(O/f'outer{f+1}_selection.json',{'bias':offsets.tolist(),'selected_from':'inner OOF only','selection_row_ids':data.iloc[it].row_id.tolist()})
        for variant in 'ABC':
            ps=np.array([fit_predict(f'outer{f+1}/final/{variant}',variant,it,iv,seed) for seed in SEEDS])
            seed_oof[variant][:,iv]=ps;oof[variant][iv]=ps.mean(0)
        logits=np.log(oof['A'][iv].clip(1e-9))+offsets;q=np.exp(logits-logits.max(1,keepdims=True));q/=q.sum(1,keepdims=True);oof['D'][iv]=q
        for k in 'ABCD':stats.append({'outer_fold':f+1,'variant':k,'n':len(iv),**score(y[iv],oof[k][iv].argmax(1))})
        pd.DataFrame(stats).to_csv(O/'fold_metrics.csv',index=False)
        save(O/'status.json',{'completed_outer_folds':f+1,'total':5,'last_results':stats[-4:]})
        print('OUTER COMPLETED',f+1,stats[-4:],flush=True)

    results={};predframe=pd.DataFrame({'row_id':data.row_id,'truth':y,'outer_fold':folds})
    for k in 'ABCD':
        pred=oof[k].argmax(1);predframe[k]=pred
        rep=classification_report(y,pred,labels=range(4),target_names=LABELS,output_dict=True,zero_division=0)
        results[k]={'n':591,**score(y,pred),'macro_precision':rep['macro avg']['precision'],'macro_recall':rep['macro avg']['recall'],'per_class':rep,'confusion_matrix':confusion_matrix(y,pred,labels=range(4)).tolist()}
        for c in range(4):predframe[f'{k}_score_{c}']=oof[k][:,c]
    predframe.to_csv(O/'oof_predictions.csv',index=False);save(O/'results.json',results)
    np.savez_compressed(O/'oof_probabilities.npz',**oof,**{k+'_seeds':v for k,v in seed_oof.items()})
    save(O/'completed.json',{'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fits':150,'evaluation_rows':591,'test_file_access':False,'protocol_sha256':sha(O/'protocol.json')})
    print('ALL COMPLETED',json.dumps({k:{s:v[s] for s in ['accuracy','macro_f1']} for k,v in results.items()}),flush=True)

if __name__ == "__main__":
    main()
