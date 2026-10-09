"""Correct physical-RGB Figure 4 panel for a trained official PiNI CNN."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from .data import MEAN,STD
from .official import load_official
from .reproduce_pets import PetsImages


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='artifacts/pets_image_seed1')
    p.add_argument('--out',default='reports/pini_official');p.add_argument('--device',default='cuda');a=p.parse_args()
    torch.set_num_threads(6);torch.manual_seed(4017)
    run=Path(a.run);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ck=torch.load(run/'last.pt',map_location=a.device,weights_only=False);splits=json.loads((run/'split.json').read_text())
    classes,provenance=load_official('image');gen=classes.VPNGenerator(512,3,224,16,768).to(a.device).eval()
    gen.load_state_dict({k.removeprefix('vpnGenerator.'):v for k,v in ck['state'].items() if k.startswith('vpnGenerator.')})
    training_precision=ck['args'].get('precision','amp')
    literal_fp16=a.device=='cuda' and training_precision=='fp16'
    if literal_fp16:gen.half()
    fixed=[]
    for k in [0,7,14,21,28,35]:fixed.append(sorted([r for r in splits['val'] if r['label']==k],key=lambda r:r['path'])[0])
    data=PetsImages(fixed)
    x=torch.stack([data[i][0] for i in range(len(data))]).to(a.device)
    if literal_fp16:x=x.half()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=a.device=='cuda' and not literal_fp16):
        mu,sigma=gen(x);sample=gen.sample(mu,sigma).squeeze(1)
    std=torch.tensor(STD,device=a.device)[None,:,None,None];mean=torch.tensor(MEAN,device=a.device)[None,:,None,None]
    rgb=x.float()*std+mean;mrgb=mu.float()*std;srgb=sigma.float()*std
    y=rgb+sample.float()*std;det=rgb+mrgb
    rms=mrgb.square().mean(1).sqrt().cpu().numpy();var=srgb.square().mean(1).cpu().numpy()
    fig,axes=plt.subplots(len(fixed),5,figsize=(12,2.5*len(fixed)),layout='constrained')
    headers=['Original RGB','Mean RMS |μ| (RGB)','Variance σ² (RGB²)','Mean-only image','Actual PiNI sample']
    rows=[]
    geom=T.Compose([T.Resize(224,interpolation=T.InterpolationMode.NEAREST),T.CenterCrop(224)])
    for j,row in enumerate(fixed):
        axes[j,0].imshow(rgb[j].cpu().permute(1,2,0).clamp(0,1));axes[j,0].set_ylabel(row['classname'],fontsize=10)
        hm=axes[j,1].imshow(rms[j],cmap='magma',vmin=0,vmax=float(rms.max()))
        hv=axes[j,2].imshow(var[j],cmap='viridis',vmin=0,vmax=float(var.max()))
        axes[j,3].imshow(det[j].cpu().permute(1,2,0).clamp(0,1));axes[j,4].imshow(y[j].cpu().permute(1,2,0).clamp(0,1))
        for c in range(5):
            axes[j,c].set_xticks([]);axes[j,c].set_yticks([])
            if j==0:axes[j,c].set_title(headers[c],fontsize=10)
        path=Path(row['path']);seg=np.asarray(geom(Image.open(path.parents[1]/'annotations/trimaps'/(path.stem+'.png'))))
        fg=seg==1;bg=seg==2
        rows.append(dict(path=row['path'],fg_variance=float(var[j][fg].mean()),bg_variance=float(var[j][bg].mean()),
                         fg_mu_rms=float(rms[j][fg].mean()),bg_mu_rms=float(rms[j][bg].mean()),
                         clipping_fraction=float(((y[j]<0)|(y[j]>1)).float().mean())))
    fig.colorbar(hm,ax=axes[:,1],shrink=.4,label='RGB mean RMS',pad=.01)
    fig.colorbar(hv,ax=axes[:,2],shrink=.4,label='RGB variance',pad=.01)
    fig.suptitle(f'Official PiNI image CNN — OxfordPets 16-shot, {ck["epoch"]} epochs\nActual CLIP inverse normalization; common physical scales; no amplification; fixed validation examples',fontsize=11)
    for ext in ['png','svg','pdf']:fig.savefig(out/('figure4_official.'+ext),dpi=180)
    plt.close(fig)
    metadata=dict(checkpoint=str(run/'last.pt'),epoch=ck['epoch'],sha256=hashlib.sha256((run/'last.pt').read_bytes()).hexdigest(),
                  provenance=provenance,sampling_seed=4017,examples=rows,mu_channel_RMS=mu.float().square().mean((-1,-2)).sqrt().cpu().tolist(),
                  sigma_channel_RMS=sigma.float().square().mean((-1,-2)).sqrt().cpu().tolist(),
                  no_amplification=True,mask_supervision=False,inference_gt_mask_inputs=False,
                  training_precision=training_precision,inference_precision='literal CUDA fp16' if literal_fp16 else 'CUDA AMP' if a.device=='cuda' else 'CPU float32',
                  selection='first sorted validation file in six preselected classes')
    (out/'figure4_official.json').write_text(json.dumps(metadata,indent=2));print(metadata,flush=True)


if __name__=='__main__':main()
