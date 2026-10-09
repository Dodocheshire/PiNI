"""Whole OxfordPets validation segmentation audit of trained official CNN."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from .data import STD
from .official import load_official
from .reproduce_pets import PetsImages


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='artifacts/pets_image_seed1');p.add_argument('--batch',type=int,default=4)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--out',default='reports/pini_official');a=p.parse_args()
    torch.set_num_threads(6);run=Path(a.run);ck=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
    classes,provenance=load_official('image');gen=classes.VPNGenerator(512,3,224,16,768).eval()
    gen.load_state_dict({k.removeprefix('vpnGenerator.'):v.float() for k,v in ck['state'].items() if k.startswith('vpnGenerator.')})
    gen=gen.to(a.device);training_precision=ck['args'].get('precision','amp')
    literal_fp16=a.device=='cuda' and training_precision=='fp16'
    if literal_fp16:gen.half()
    records=json.loads((run/'split.json').read_text())['val'];dataset=PetsImages(records)
    geom=T.Compose([T.Resize(224,interpolation=T.InterpolationMode.NEAREST),T.CenterCrop(224)])
    rows=[];std=torch.tensor(STD,device=a.device)[None,:,None,None]
    with torch.no_grad():
        for start in range(0,len(records),a.batch):
            ii=range(start,min(start+a.batch,len(records)));x=torch.stack([dataset[i][0] for i in ii]).to(a.device)
            if literal_fp16:x=x.half()
            with torch.autocast('cuda',dtype=torch.float16,enabled=a.device=='cuda' and not literal_fp16):mu,sigma=gen(x)
            rms=(mu.float()*std).square().mean(1).sqrt().cpu().numpy();var=(sigma.float()*std).square().mean(1).cpu().numpy()
            for j,i in enumerate(ii):
                pth=Path(records[i]['path']);seg=np.asarray(geom(Image.open(pth.parents[1]/'annotations/trimaps'/(pth.stem+'.png'))))
                fg=seg==1;bg=seg==2
                if not fg.any() or not bg.any():continue
                rows.append(dict(path=records[i]['path'],fg_variance=float(var[j][fg].mean()),bg_variance=float(var[j][bg].mean()),
                                 fg_mu_rms=float(rms[j][fg].mean()),bg_mu_rms=float(rms[j][bg].mean())))
            if start%40==0:print('official native energy audit',start,len(records),flush=True)
    fields=['fg_variance','bg_variance','fg_mu_rms','bg_mu_rms'];summary={k:float(np.mean([r[k] for r in rows])) for k in fields}
    summary.update(n=len(rows),sigma2_bg_over_fg=summary['bg_variance']/summary['fg_variance'],
                   mu_fg_over_bg=summary['fg_mu_rms']/summary['bg_mu_rms'],
                   fraction_images_bg_variance_greater=float(np.mean([r['bg_variance']>r['fg_variance'] for r in rows])),
                   fraction_images_fg_mu_greater=float(np.mean([r['fg_mu_rms']>r['bg_mu_rms'] for r in rows])),
                   checkpoint_epoch=ck['epoch'],checkpoint_sha256=hashlib.sha256((run/'last.pt').read_bytes()).hexdigest(),
                   training_precision=training_precision,
                   inference='original CNN, '+('literal CUDA fp16' if literal_fp16 else 'CUDA AMP' if a.device=='cuda' else 'CPU float32 from stored weights'),mask_supervision=False,
                   gt_mask_usage='OxfordPets native trimaps only for independent statistics, exclude boundary label3')
    result=dict(summary=summary,per_image=rows,provenance=provenance)
    (run/'energy_audit.json').write_text(json.dumps(result,indent=2));dest=Path(a.out);dest.mkdir(parents=True,exist_ok=True)
    (dest/'energy_audit.json').write_text(json.dumps(result,indent=2));print(summary,flush=True)


if __name__=='__main__':main()
