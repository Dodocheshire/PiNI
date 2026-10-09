"""Fixed validation gallery of unamplified, actually sampled learned RGB noise."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from .data import ATTRS,CLASSES
from .models import ConditionalRGBNoise,SpatialConcepts
from .train_spatial import load_cache


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='artifacts/noise_independent')
    p.add_argument('--cache',default='artifacts/rival_vitl14');p.add_argument('--out',default='reports/spatial_noise_v1')
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(1207)
    cache=Path(a.cache);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    arrays,splits,labels,records=load_cache(cache)
    rgb=np.load(cache/'rgb.npy',mmap_mode='r');texts=torch.tensor(np.load(cache/'text.npy'))
    checkpoint=torch.load(Path(a.run)/'best.pt',map_location='cpu',weights_only=False)
    cfg=checkpoint['args'];noise=ConditionalRGBNoise(SpatialConcepts(texts),cfg['independent_heads'],cfg.get('concept_local',False))
    noise.load_state_dict(checkpoint['model']);noise.eval()
    # Fixed semantic/class rule only, never predicted performance or noise gain.
    choices=[(0,'wheels'),(2,'wings'),(4,'ears'),(5,'long-snout'),(8,'wet'),(9,'beak')]
    ids=[];ks=[]
    for label,attr in choices:
        k=ATTRS.index(attr)
        eligible=[int(i) for i in splits['val'] if labels[i]==label and arrays['attrs'][i,k]==1]
        ids.append(eligible[0]);ks.append(k)
    tokens=torch.tensor(np.array(arrays['tokens'][ids]),dtype=torch.float32)
    images=torch.tensor(np.array(rgb[ids]),dtype=torch.float32)/255
    k=torch.tensor(ks)
    with torch.no_grad():
        noisy,mu,sigma,gate=noise.perturb(images,tokens,k,mode=cfg['mode'],suppress=cfg['suppress'])
        mean,_,_,_=noise.perturb(images,tokens,k,mode='mean',suppress=cfg['suppress'])
        _,_,_,mask,fg=noise.concepts.spatial(tokens)
    variance=sigma.square().mean(1).numpy();mean_rms=mu.square().mean(1).sqrt().numpy()
    panel=plt.subplots(len(ids),6,figsize=(15,2.5*len(ids)),layout='constrained')
    fig,axes=panel
    headers=['Original','Predicted object gate','Mean |μ| RMS (RGB)','Variance σ² (RGB²)','Deterministic view','Actual Gaussian sample']
    mu_max=max(float(mean_rms.max()),.001);var_max=max(float(variance.max()),.001)
    stats=[]
    for j,i in enumerate(ids):
        axes[j,0].imshow(images[j].permute(1,2,0));axes[j,0].set_ylabel(CLASSES[labels[i]]+'\n'+ATTRS[ks[j]],fontsize=11)
        axes[j,1].imshow(torch.sigmoid(fg[j]).reshape(16,16),cmap='cividis',vmin=0,vmax=1)
        hm=axes[j,2].imshow(mean_rms[j],cmap='magma',vmin=0,vmax=mu_max)
        hv=axes[j,3].imshow(variance[j],cmap='viridis',vmin=0,vmax=var_max)
        axes[j,4].imshow(mean[j].permute(1,2,0).clamp(0,1))
        axes[j,5].imshow(noisy[j].permute(1,2,0).clamp(0,1))
        for col in range(6):
            axes[j,col].set_xticks([]);axes[j,col].set_yticks([])
            if j==0:axes[j,col].set_title(headers[col],fontsize=11)
        gt=torch.tensor(np.array(arrays['masks'][i,18]))
        gt=torch.nn.functional.interpolate(gt[None,None],(224,224),mode='nearest')[0,0]>.5
        s2=sigma[j].square().mean(0);mr=mu[j].square().mean(0).sqrt()
        stats.append(dict(id=int(i),path=records[i]['path'],concept=ATTRS[ks[j]],
                          clipping_fraction=float(((noisy[j]<0)|(noisy[j]>1)).float().mean()),
                          foreground_variance=float(s2[gt].mean()),background_variance=float(s2[~gt].mean()),
                          foreground_mean_rms=float(mr[gt].mean()),background_mean_rms=float(mr[~gt].mean())))
    fig.colorbar(hm,ax=axes[:,2],shrink=.4,label='RGB mean RMS',pad=.01)
    fig.colorbar(hv,ax=axes[:,3],shrink=.4,label='RGB variance',pad=.01)
    fig.suptitle('Learned spatial noise CBM — fixed validation examples; no noise amplification\nForeground supervision during training; predicted masks at inference; explicit background variance budget',fontsize=12)
    for ext in ['png','svg','pdf']:fig.savefig(out/('figure4_style.'+ext),dpi=180)
    plt.close(fig)
    metadata=dict(checkpoint=str(Path(a.run)/'best.pt'),epoch=checkpoint['epoch'],sampling_seed=1207,
                  selection='first sorted validation sample per fixed class/concept with positive native label',
                  mu_scale=[0,mu_max],variance_scale=[0,var_max],examples=stats,
                  inference_gt_mask_inputs=False,mean_unit='RGB',variance_unit='RGB squared',noise_amplification=1,
                  claim='Visualization validates actual tensors and imposed/localized energy, not classification benefit by itself.')
    (out/'figure4_style.json').write_text(json.dumps(metadata,indent=2))
    # A machine-readable artifact retains every tensor used to draw the panel.
    np.savez_compressed(Path(a.run)/'figure4_tensors.npz',original=images.numpy(),mu=mu.numpy(),sigma=sigma.numpy(),gate=gate.numpy(),sample=noisy.numpy(),mean=mean.numpy(),ids=ids,concept_ids=ks)
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':main()
