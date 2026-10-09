"""Compact plot of the first three predefined gallery examples, same tensors."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .data import ATTRS,CLASSES
from .train_spatial import load_cache


def main():
    run=Path('artifacts/region_noise');out=Path('reports/spatial_noise_region')
    gallery=json.loads((out/'figure4_style.json').read_text())
    data=np.load(run/'figure4_tensors.npz');chosen=np.arange(3)
    _,_,labels,_=load_cache('artifacts/rival_vitl14')
    assert data['ids'].tolist()==[r['id'] for r in gallery['examples']]
    assert gallery['noise_amplification']==1 and not gallery['inference_gt_mask_inputs']
    mu=np.sqrt(np.mean(data['mu']**2,axis=1));variance=np.mean(data['sigma']**2,axis=1)
    fig,axes=plt.subplots(3,4,figsize=(10.8,7.8),layout='constrained')
    headers=['Original','Concept mean offset RMS(μ)','Variance σ²','Actual Gaussian sample']
    for row,j in enumerate(chosen):
        axes[row,0].imshow(data['original'][j].transpose(1,2,0))
        m=axes[row,1].imshow(mu[j],cmap='magma',vmin=0,vmax=gallery['mu_scale'][1])
        v=axes[row,2].imshow(variance[j],cmap='viridis',vmin=0,vmax=gallery['variance_scale'][1])
        axes[row,3].imshow(np.clip(data['sample'][j].transpose(1,2,0),0,1))
        axes[row,0].set_ylabel(CLASSES[labels[data['ids'][j]]]+' / '+ATTRS[data['concept_ids'][j]],fontsize=10)
        for col in range(4):
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(headers[col],fontsize=11)
    fig.colorbar(m,ax=axes[:,1],shrink=.55,label='RGB RMS',pad=.015)
    fig.colorbar(v,ax=axes[:,2],shrink=.55,label='RGB²',pad=.015)
    fig.suptitle('Spatial Gaussian CBM — same real tensors as the full fixed gallery\nForeground/concept predictions only; unamplified sampling; physical color scales',fontsize=11)
    for ext in ['png','svg','pdf']:fig.savefig(out/('figure4_main.'+ext),dpi=180)
    plt.close(fig)
    metadata=dict(source_tensor_file=str(run/'figure4_tensors.npz'),
        source_tensor_sha256=hashlib.sha256((run/'figure4_tensors.npz').read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(),
        selected_rows=chosen.tolist(),ids=data['ids'][chosen].tolist(),concept_ids=data['concept_ids'][chosen].tolist(),
        selection='First three of the original six predefined class/concept examples, for a compact entry figure. No choice by prediction improvement. Full six examples remain linked.',
        sampling_seed=gallery['sampling_seed'],noise_amplification=1,inference_gt_mask_inputs=False,
        mean_unit='RGB RMS of additive Gaussian offset after foreground gating',variance_unit='RGB squared',
        mu_scale=gallery['mu_scale'],variance_scale=gallery['variance_scale'],
        interpretation='Actual perturbation distribution; usefulness must be read from matched quantitative controls, not this image alone.')
    (out/'figure4_main.json').write_text(json.dumps(metadata,indent=2));print(metadata,flush=True)


if __name__=='__main__':main()
