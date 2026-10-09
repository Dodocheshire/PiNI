"""Paired held-out uncertainty, including all concepts rather than selected wins.

Bootstrap multiplicities preserve paired images. AP uses the fixed score order
and exact tie groups, so resampling does not repeatedly sort the same scores.
Simultaneous concept intervals use the max absolute centered bootstrap error;
they do not turn post-hoc inspection into a preregistered hypothesis.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from .data import ATTRS
from .train_spatial import load_cache,head_predict


PAIRS=[('noise_independent','mean_independent'),('noise_sequential','mean_sequential'),
       ('local_noise','local_mean'),('full_joint_noise','full_joint_mean'),('region_noise','region_mean'),
       ('noise_separate_heads','mean_separate_heads'),('fixed_noise','fixed_mean'),('region_seed43_noise','region_seed43_mean')]


def ci_correct(a,b,seed=1149,repetitions=2000):
    diff=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    rng=np.random.default_rng(seed)
    samples=np.array([diff[rng.integers(0,len(diff),len(diff))].mean() for _ in range(repetitions)])
    return dict(difference_pp=float(diff.mean()*100),CI95_pp=(np.quantile(samples,[.025,.975])*100).tolist())


def weighted_ap(y,score,counts):
    order=np.argsort(-score,kind='stable');ys=y[order];ss=score[order]
    ends=np.r_[np.flatnonzero(ss[1:]!=ss[:-1]),len(y)-1]
    weights=counts[:,order]
    true=(weights*ys).cumsum(1)[:,ends]
    total=weights.cumsum(1)[:,ends]
    increments=np.diff(true,axis=1,prepend=0)
    return ((true/np.maximum(total,1))*increments).sum(1)/np.maximum(true[:,-1],1)


def bootstrap_concepts(y,a,b,repetitions,seed):
    observed=average_precision_score(y,a,average=None)-average_precision_score(y,b,average=None)
    # Exact consistency check of weighted/tied AP against the trusted metric.
    unit=np.ones((1,len(y)),dtype=np.uint16)
    for k in range(y.shape[1]):
        for scores in [a,b]:
            assert np.isclose(weighted_ap(y[:,k],scores[:,k],unit)[0],average_precision_score(y[:,k],scores[:,k]),atol=1e-10)
    rng=np.random.default_rng(seed);samples=[]
    for start in range(0,repetitions,32):
        n=min(32,repetitions-start)
        indices=rng.integers(0,len(y),size=(n,len(y)))
        counts=np.stack([np.bincount(row,minlength=len(y)) for row in indices]).astype(np.uint16)
        delta=np.stack([weighted_ap(y[:,k],a[:,k],counts)-weighted_ap(y[:,k],b[:,k],counts) for k in range(y.shape[1])],1)
        samples.append(delta)
    samples=np.concatenate(samples)
    bands=np.quantile(samples,[.025,.975],axis=0)
    radius=np.quantile(np.max(np.abs(samples-observed),axis=1),.95)
    per=[dict(name=attr,difference_AP_pp=float(observed[k]*100),
              CI95_pp=(bands[:,k]*100).tolist(),
              simultaneous_CI95_pp=((observed[k]+np.array([-radius,radius]))*100).tolist(),
              test_positives=int(y[:,k].sum())) for k,attr in enumerate(ATTRS)]
    return dict(mAP_difference_pp=float(observed.mean()*100),
                mAP_CI95_pp=(np.quantile(samples.mean(1),[.025,.975])*100).tolist(),per_concept=per,
                simultaneous_method='95% quantile of maximum absolute centered paired bootstrap error across all 18 concepts')


def main():
    p=argparse.ArgumentParser();p.add_argument('--repetitions',type=int,default=1000);args=p.parse_args()
    arrays,splits,labels,_=load_cache('artifacts/rival_vitl14');y=np.array(arrays['attrs'][splits['test']])
    target=labels[splits['test']];report=Path('reports/spatial_noise_v1');report.mkdir(parents=True,exist_ok=True)
    dst=report/'paired_uncertainty.json';prior=json.loads(dst.read_text()) if dst.exists() else {}
    result=dict(repetitions=args.repetitions,n_test=len(y),pairs={},interpretation='Single seed and split; intervals quantify paired test-sample uncertainty, not training-seed variability. All 18 concepts are reported. Test-inspected concept gains remain exploratory.')
    for a,b in PAIRS:
        ra=Path('artifacts')/a;rb=Path('artifacts')/b
        if not all((r/'results.json').exists() for r in [ra,rb]):continue
        key=a+'_minus_'+b
        files=[r/f for r in [ra,rb] for f in ['test_probs.npy','test_class_logits.npy','val_probs.npy']]
        fingerprint=hashlib.sha256(b''.join(f.read_bytes() for f in files)+str(args.repetitions).encode()+Path(__file__).read_bytes()).hexdigest()
        old=prior.get('pairs',{}).get(key,{})
        if old.get('input_sha256')==fingerprint:
            row=old
        else:
            ap=np.load(ra/'test_probs.npy');bp=np.load(rb/'test_probs.npy')
            row=bootstrap_concepts(y,ap,bp,args.repetitions,1617)
            row['classification']=ci_correct(np.load(ra/'test_class_logits.npy').argmax(1)==target,np.load(rb/'test_class_logits.npy').argmax(1)==target)
            yv=np.array(arrays['attrs'][splits['val']])
            vd=average_precision_score(yv,np.load(ra/'val_probs.npy'),average=None)-average_precision_score(yv,np.load(rb/'val_probs.npy'),average=None)
            row['validation_AP_differences_pp']=(vd*100).tolist()
            row['validation_largest_gain_concept']=ATTRS[int(np.argmax(vd))]
            row['input_sha256']=fingerprint
        semantic=Path('artifacts/native_semantic_head')
        for suffix,name in [('correct','shared_GT_semantic_head'),('binary_correct','shared_GT_semantic_head_binary')]:
            if all((semantic/(n+'_'+suffix+'.npy')).exists() for n in [a,b]):
                row[name]=ci_correct(np.load(semantic/(a+'_'+suffix+'.npy')),np.load(semantic/(b+'_'+suffix+'.npy')))
        if (ra/'test_mean_class_logits.npy').exists():
            noisy=np.load(ra/'test_class_logits.npy').argmax(1)==target
            mean=np.load(ra/'test_mean_class_logits.npy').argmax(1)==target
            deterministic=np.load(rb/'test_class_logits.npy').argmax(1)==target
            row['same_checkpoint_sampling']=ci_correct(noisy,mean)
            row['trained_noise_mean_inference_minus_trained_mean']=ci_correct(mean,deterministic)
            if (semantic/'head.pt').exists():
                import torch
                head=torch.load(semantic/'head.pt',map_location='cpu',weights_only=False)
                correct=head_predict(head,np.load(ra/'test_mean_probs.npy')).argmax(1)==target
                comparator=head_predict(head,np.load(rb/'test_probs.npy')).argmax(1)==target
                row['shared_GT_head_trained_noise_mean_minus_trained_mean']=ci_correct(correct,comparator)
        result['pairs'][key]=row
        print(key,{k:v for k,v in row.items() if k not in ['per_concept','validation_AP_differences_pp','input_sha256']},flush=True)
    dst.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
