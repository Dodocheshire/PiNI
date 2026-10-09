"""Native background precision/recall: operating points and ranking diagnostic.

Precision envelopes on test PR curves are evaluation metrics, not thresholds
selected for inference. Deployed thresholds remain 0.5 or validation-derived.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from .train_spatial import load_cache


def envelope(y,scores,counts,recall=.9):
    values=[]
    for k in range(y.shape[1]):
        order=np.argsort(-scores[:,k],kind='stable');ss=scores[order,k]
        ends=np.r_[np.flatnonzero(ss[1:]!=ss[:-1]),len(y)-1]
        weights=counts[:,order]
        true=(weights*y[order,k]).cumsum(1)[:,ends];total=weights.cumsum(1)[:,ends]
        precision=true/np.maximum(total,1)
        eligible=true>=recall*true[:,-1,None]
        values.append(np.where(eligible,precision,0).max(1))
    return np.stack(values,1).mean(1)


def point(y,pred,counts):
    true=counts@(pred*y);predicted=counts@pred.astype(float);positive=counts@y
    return (true/np.maximum(predicted,1)).mean(1),(true/np.maximum(positive,1)).mean(1)


def bootstrap(y,a,b,ta,tb,repetitions=1000):
    unit=np.ones((1,len(y)))
    for scores in [a,b]:
        reference=[]
        for k in range(18):
            p,r,_=precision_recall_curve(y[:,k],scores[:,k]);reference.append(p[r>=.9].max())
        assert np.isclose(envelope(y,scores,unit)[0],np.mean(reference),atol=1e-12)
    observed=dict(precision_envelope_at_recall90=float((envelope(y,a,unit)-envelope(y,b,unit))[0]*100))
    for label,pa,pb in [('threshold05',a>=.5,b>=.5),('validation_thresholds',a>=ta,b>=tb)]:
        ap,ar=point(y,pa,unit);bp,br=point(y,pb,unit)
        observed[label+'_precision']=float((ap-bp)[0]*100);observed[label+'_recall']=float((ar-br)[0]*100)
    samples={key:[] for key in observed};rng=np.random.default_rng(4851)
    for start in range(0,repetitions,32):
        indices=rng.integers(0,len(y),size=(min(32,repetitions-start),len(y)))
        counts=np.stack([np.bincount(row,minlength=len(y)) for row in indices]).astype(float)
        samples['precision_envelope_at_recall90'].append((envelope(y,a,counts)-envelope(y,b,counts))*100)
        for label,pa,pb in [('threshold05',a>=.5,b>=.5),('validation_thresholds',a>=ta,b>=tb)]:
            ap,ar=point(y,pa,counts);bp,br=point(y,pb,counts)
            samples[label+'_precision'].append((ap-bp)*100);samples[label+'_recall'].append((ar-br)*100)
    return {key:dict(difference_pp=value,CI95_pp=np.quantile(np.concatenate(samples[key]),[.025,.975]).tolist()) for key,value in observed.items()}


def main():
    arrays,splits,_,_=load_cache('artifacts/rival_vitl14');yv=np.array(arrays['attrs'][splits['val']],dtype=float)
    out=Path('reports/spatial_noise_v1');dst=out/'background_precision.json'
    prior=json.loads(dst.read_text()) if dst.exists() else {'pairs':{}}
    result=dict(target_recall=.9,repetitions=1000,models={},pairs={},
        interpretation='All 18 native concepts on the same 1000 background-swapped test images. PR-envelope precision at recall >=90% is a ranking diagnostic computed from test curves; no test threshold is used for model inference/selection. Operating points separately use 0.5 and original validation positive-score 10th percentile. Single-split adaptation seeds share clean pretraining; intervals measure paired image uncertainty only.')
    grid=np.linspace(.5,1,51);fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');plotted=False
    for panel,(folder,a,b,label) in enumerate([
        ('background_intervention_region','region_noise','region_mean','Adaptation seed 42'),
        ('background_intervention_region_seed43','region_seed43_noise','region_seed43_mean','Adaptation seed 43')]):
        d=Path('artifacts')/folder
        if not (d/'results.json').exists():continue
        protocol=json.loads((d/'protocol.json').read_text());y=np.array(arrays['attrs'][protocol['ids']],dtype=float)
        probabilities={};thresholds={};names=[b,a,a+'_same_checkpoint_mean']
        for name,color,legend in zip(names,['#009E73','#E69F00','#D55E00'],['Mean-trained','Noise-trained MC1','Noise-trained mean inference']):
            p=np.load(d/(name+'_background_swap_probs.npy'));probabilities[name]=p
            source=name.removesuffix('_same_checkpoint_mean');val=np.load(Path('artifacts')/source/'val_probs.npy')
            t=np.array([np.quantile(val[yv[:,k]==1,k],.1) for k in range(18)]);thresholds[name]=t
            curves=[]
            for k in range(18):
                precision,recall,_=precision_recall_curve(y[:,k],p[:,k]);curves.append([float(precision[recall>=r].max()) for r in grid])
            curve=np.array(curves).mean(0);unit=np.ones((1,len(y)))
            p05,r05=point(y,p>=.5,unit);pv,rv=point(y,p>=t,unit)
            result['models'][name]=dict(n=len(y),precision_envelope_at_recall90=float(envelope(y,p,unit)[0]),
                threshold05=dict(macro_precision=float(p05[0]),macro_recall=float(r05[0])),
                validation_thresholds=dict(macro_precision=float(pv[0]),macro_recall=float(rv[0]),thresholds=t.tolist(),
                    calibration='Original-image MC1 validation scores of this trained model; reused for its mean-inference diagnostic'),
                recall_grid=grid.tolist(),macro_precision_envelope=curve.tolist())
            axes[panel].plot(grid*100,curve*100,color=color,label=legend,lw=2)
        for source,comparator in [(a,b),(a,a+'_same_checkpoint_mean'),(a+'_same_checkpoint_mean',b)]:
            x=probabilities[source];z=probabilities[comparator];tx=thresholds[source];tz=thresholds[comparator];key=source+'_minus_'+comparator
            fingerprint=hashlib.sha256(y.tobytes()+x.tobytes()+z.tobytes()+tx.tobytes()+tz.tobytes()+Path(__file__).read_bytes()).hexdigest();old=prior['pairs'].get(key,{})
            if old.get('input_sha256')==fingerprint:row=old
            else:row=dict(metrics=bootstrap(y,x,z,tx,tz),input_sha256=fingerprint,n=len(y),folder=str(d))
            result['pairs'][key]=row
            print(key,row['metrics']['precision_envelope_at_recall90'],flush=True)
        axes[panel].set_title(label);axes[panel].set_xlabel('Native concept recall (%)');axes[panel].set_ylabel('Macro interpolated precision (%)');axes[panel].set_ylim(40,100);axes[panel].grid(alpha=.2);axes[panel].legend(fontsize=8);axes[panel].axvline(90,color='gray',linestyle='--',lw=1);plotted=True
    if plotted:
        if len(result['models'])<6:axes[1].axis('off');axes[1].text(.5,.5,'Second-seed background audit pending',ha='center',va='center',transform=axes[1].transAxes)
        fig.suptitle('Background-swap native PR envelopes — evaluation curves, not deployed thresholds',fontsize=10)
        for ext in ['png','svg','pdf']:fig.savefig(out/('background_precision.'+ext),dpi=180)
    plt.close(fig);dst.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
