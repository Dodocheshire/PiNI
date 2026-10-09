"""Same fixed image, different concept queries; actual learned RGB fields."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from .data import ATTRS,mask_from_uri
from .models import ConditionalRGBNoise,SpatialConcepts
from .train_spatial import load_cache


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(1208);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cache=Path('artifacts/rival_vitl14');arrays,splits,labels,records=load_cache(cache)
    # The rule is fixed before the concept-local models finish training.
    concepts=['long-snout','ears','tail'];ks=[ATTRS.index(k) for k in concepts]
    ids=[int(i) for i in splits['val'] if labels[i]==5 and np.all(arrays['attrs'][i,ks]==1) and np.all(arrays['mask_valid'][i,ks]) and arrays['mask_valid'][i,18]]
    if not ids:raise RuntimeError('No dog validation image satisfies the pre-fixed three-concept rule')
    i=ids[0];ck=torch.load(Path(a.run)/'best.pt',map_location='cpu',weights_only=False);cfg=ck['args']
    noise=ConditionalRGBNoise(SpatialConcepts(torch.tensor(np.load(cache/'text.npy'))),cfg['independent_heads'],cfg.get('concept_local',False))
    noise.load_state_dict(ck['model']);noise.eval()
    tokens=torch.tensor(np.array(arrays['tokens'][[i]*3]),dtype=torch.float32)
    rgb=np.load(cache/'rgb.npy',mmap_mode='r');images=torch.tensor(np.array(rgb[[i]*3]),dtype=torch.float32)/255
    with torch.no_grad():
        sample,mu,sigma,gate=noise.perturb(images,tokens,torch.tensor(ks),mode='noise',suppress=cfg['suppress'])
        _,_,_,mask,_=noise.concepts.spatial(tokens)
        relevance=F.interpolate(mask[torch.arange(3),torch.tensor(ks)].sigmoid().reshape(3,1,16,16),(224,224),mode='bilinear',align_corners=False)[:,0]
    path=Path(records[i]['path'])
    with path.with_name(path.stem+'_attr_dict.pkl').open('rb') as f:gt=pickle.load(f)
    decode=lambda name: np.asarray(mask_from_uri(gt[name]).resize((224,224),Image.Resampling.NEAREST))>0
    object_mask=decode('entire-object');parts=[decode(k) for k in concepts]
    mr=mu.square().mean(1).sqrt().numpy();s2=sigma.square().mean(1).numpy()
    fig,axes=plt.subplots(3,6,figsize=(15,8),layout='constrained');metadata=[]
    headers=['Original','Native part reference','Predicted concept relevance','Concept |μ| RMS','Variance σ² (RGB²)','Actual Gaussian sample']
    for j,k in enumerate(concepts):
        axes[j,0].imshow(images[j].permute(1,2,0));axes[j,0].set_ylabel(k,fontsize=12)
        axes[j,1].imshow(images[j].permute(1,2,0));axes[j,1].contour(parts[j],levels=[.5],colors=['#00e0a0'],linewidths=1)
        axes[j,2].imshow(relevance[j],cmap='cividis',vmin=0,vmax=1)
        hm=axes[j,3].imshow(mr[j],cmap='magma',vmin=0,vmax=max(float(mr.max()),.001))
        hv=axes[j,4].imshow(s2[j],cmap='viridis',vmin=0,vmax=max(float(s2.max()),.001))
        axes[j,5].imshow(sample[j].permute(1,2,0).clamp(0,1))
        for col in range(6):
            axes[j,col].set_xticks([]);axes[j,col].set_yticks([])
            if j==0:axes[j,col].set_title(headers[col],fontsize=10)
        other=object_mask&~parts[j]
        metadata.append(dict(concept=k,part_mu_rms=float(mr[j][parts[j]].mean()),
            other_object_mu_rms=float(mr[j][other].mean()) if other.any() else None,
            part_variance=float(s2[j][parts[j]].mean()),other_object_variance=float(s2[j][other].mean()) if other.any() else None,
            clipping_fraction=float(((sample[j]<0)|(sample[j]>1)).float().mean())))
    fig.colorbar(hm,ax=axes[:,3],shrink=.6,label='RGB mean RMS',pad=.01)
    fig.colorbar(hv,ax=axes[:,4],shrink=.6,label='RGB variance',pad=.01)
    fig.suptitle('Same image, three concept-conditioned noise distributions\nFixed validation selection; native mask is a reference only; actual samples without amplification',fontsize=12)
    for ext in ['png','svg','pdf']:fig.savefig(out/('concept_conditioning.'+ext),dpi=180)
    plt.close(fig)
    meta=dict(path=str(path),id=i,checkpoint=str(Path(a.run)/'best.pt'),checkpoint_sha256=hashlib.sha256((Path(a.run)/'best.pt').read_bytes()).hexdigest(),
        epoch=ck['epoch'],selection='First sorted dog validation image positive for long-snout/ears/tail, with all masks valid; rule fixed before local models completed',
        sampling_seed=1208,noise_amplification=1,inference_gt_mask_inputs=False,examples=metadata)
    (out/'concept_conditioning.json').write_text(json.dumps(meta,indent=2))
    np.savez_compressed(Path(a.run)/'concept_conditioning_tensors.npz',original=images.numpy(),mu=mu.numpy(),sigma=sigma.numpy(),gate=gate.numpy(),sample=sample.numpy(),relevance=relevance.numpy(),id=i,concept_ids=ks)
    print(meta,flush=True)


if __name__=='__main__':main()
