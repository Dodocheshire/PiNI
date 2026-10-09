"""Average two fixed adaptation runs, resampling the same paired images.

This is not an ensemble, a 2000-image test, or a seed-population interval.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from .train_spatial import load_cache
from .uncertainty import ci_correct,weighted_ap


def localization_summary():
    rows={}
    for seed,name in [(42,'region_noise'),(43,'region_seed43_noise')]:
        path=Path('artifacts')/name/'localization_audit.json'
        if not path.exists():return
        value=json.loads(path.read_text())
        rows[str(seed)]=dict(run=name,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            checkpoint_sha256=hashlib.sha256((path.parent/'best.pt').read_bytes()).hexdigest(),
            input_sha256=value['input_sha256'],summary=value['summary'],per_concept=value['per_concept'])
    Path('reports/spatial_noise_region/two_seed_localization.json').write_text(json.dumps(
        dict(adaptation_seeds=rows,protocol=value['protocol'],inference_gt_mask_inputs=False,
            note='Summaries of complete validation audits; all per-pair rows remain in local ignored artifacts. Shared clean initialization and split.'),indent=2))


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Noise training, mean inference\nminus mean training',
            'Noise training, MC1 inference\nminus mean training',
            'MC1 minus mean inference\n(same noise checkpoints)']
    keys=['noise_trained_mean_inference_minus_mean_trained',
          'noise_trained_MC1_minus_mean_trained',
          'noise_trained_MC1_minus_noise_trained_mean_inference']
    colors=['#009E73','#E69F00','#CC79A7']
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),layout='constrained')
    for ax,metric,title in zip(axes,['classification','mAP'],['Background-swap classification','Background-swap native concept mAP']):
        for i,(key,color) in enumerate(zip(keys,colors)):
            p=result['pairs'][key]
            value,ci=(p['classification']['difference_pp'],p['classification']['CI95_pp']) if metric=='classification' else (p['mAP_difference_pp'],p['mAP_CI95_pp'])
            ax.errorbar(value,i,xerr=[[value-ci[0]],[ci[1]-value]],fmt='o',color=color,capsize=5,markersize=8)
            ax.annotate(f'{value:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]',(0.98,i),xycoords=('axes fraction','data'),ha='right',xytext=(0,15),textcoords='offset points',fontsize=9)
        ax.set_yticks(range(3),labels if ax is axes[0] else ['']*3);ax.set_ylim(2.6,-.6)
        ax.axvline(0,color='#555555',ls='--',lw=1);ax.grid(axis='x',alpha=.2)
        ax.set_xlim(-1.4,2.0);ax.set_xlabel('Paired difference (percentage points)');ax.set_title(title,fontsize=10)
    fig.suptitle('Two fixed adaptation seeds; same 1,000 paired images; no prediction ensemble\n95% image-bootstrap intervals condition on these models; training-seed uncertainty excluded',fontsize=10)
    for ext in ['png','svg','pdf']:fig.savefig(Path('reports/spatial_noise_v1')/('two_seed_background.'+ext),dpi=180)
    plt.close(fig)


def main():
    localization_summary()
    folders=[Path('artifacts/background_intervention_region'),Path('artifacts/background_intervention_region_seed43')]
    dst=Path('reports/spatial_noise_v1/two_seed_summary.json')
    if not all((d/'results.json').exists() for d in folders):return
    protocols=[json.loads((d/'protocol.json').read_text()) for d in folders]
    assert protocols[0]['ids']==protocols[1]['ids'] and protocols[0]['donors']==protocols[1]['donors']
    arrays,_,_,_=load_cache('artifacts/rival_vitl14');y=np.array(arrays['attrs'][protocols[0]['ids']])
    names=[('region_mean','region_noise'),('region_seed43_mean','region_seed43_noise')]
    probabilities={key:[] for key in ['mean_trained','noise_trained_MC1','noise_trained_mean_inference']}
    correct={key:[] for key in probabilities};models=[]
    for d,(mean,noise) in zip(folders,names):
        data=json.loads((d/'results.json').read_text())
        models.append({name:data['models'][name]['background_swap'] for name in [mean,noise,noise+'_same_checkpoint_mean']})
        for key,name in [('mean_trained',mean),('noise_trained_MC1',noise),('noise_trained_mean_inference',noise+'_same_checkpoint_mean')]:
            probabilities[key].append(np.load(d/(name+'_background_swap_probs.npy')))
            correct[key].append(np.load(d/(name+'_background_swap_correct.npy')))
    fingerprint=hashlib.sha256(y.tobytes()+b''.join(a.tobytes() for values in probabilities.values() for a in values)+b''.join(a.tobytes() for values in correct.values() for a in values)+Path(__file__).read_bytes()).hexdigest()
    if dst.exists() and json.loads(dst.read_text()).get('input_sha256')==fingerprint:
        plot(json.loads(dst.read_text()))
        print('Validated two-fixed-seed image audit reused',flush=True);return
    result=dict(adaptation_seeds=[42,43],n_unique_test_images=len(y),n_fixed_models_per_mode=2,models=models,
        input_sha256=fingerprint,ensemble=False,shared_clean_initialization=True,pairs={},
        interpretation='Arithmetic average of metrics across two fixed adaptation seeds. Same paired image bootstrap for both runs preserves covariance; interval conditions on these two trained models and excludes training-seed population uncertainty. Same 1000 images, not 2000 independent cases; probabilities are not ensembled.')
    for key in probabilities:
        result[key]=dict(background_accuracy=float(np.mean(correct[key])),
            background_mAP=float(np.mean([average_precision_score(y,p,average='macro') for p in probabilities[key]])))
    rng=np.random.default_rng(4855);samples={key:[] for key in probabilities};unit=np.ones((1,len(y)),dtype=np.uint16)
    for key in probabilities:
        for p in probabilities[key]:
            assert np.isclose(np.mean([weighted_ap(y[:,k],p[:,k],unit)[0] for k in range(18)]),average_precision_score(y,p,average='macro'),atol=1e-10)
    for start in range(0,1000,32):
        indices=rng.integers(0,len(y),size=(min(32,1000-start),len(y)))
        counts=np.stack([np.bincount(row,minlength=len(y)) for row in indices]).astype(np.uint16)
        for key,ps in probabilities.items():
            ap=np.stack([weighted_ap(y[:,k],p[:,k],counts) for p in ps for k in range(18)],1).mean(1)
            samples[key].append(ap)
    for a,b in [('noise_trained_MC1','mean_trained'),('noise_trained_mean_inference','mean_trained'),('noise_trained_MC1','noise_trained_mean_inference')]:
        difference=(np.concatenate(samples[a])-np.concatenate(samples[b]))*100
        result['pairs'][a+'_minus_'+b]=dict(
            classification=ci_correct(np.mean(correct[a],axis=0),np.mean(correct[b],axis=0)),
            mAP_difference_pp=(result[a]['background_mAP']-result[b]['background_mAP'])*100,
            mAP_CI95_pp=np.quantile(difference,[.025,.975]).tolist())
    dst.write_text(json.dumps(result,indent=2));plot(result);print(result['pairs'],flush=True)


if __name__=='__main__':main()
