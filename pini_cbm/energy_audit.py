"""Whole-validation RGB noise energy, not only attractive gallery examples."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from .data import ATTRS,mask_from_uri
from .models import ConditionalRGBNoise,SpatialConcepts
from .train_spatial import load_cache


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='artifacts/noise_independent');a=p.parse_args()
    torch.set_num_threads(2);cache=Path('artifacts/rival_vitl14');run=Path(a.run)
    arrays,splits,labels,records=load_cache(cache)
    checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=False);cfg=checkpoint['args']
    noise=ConditionalRGBNoise(SpatialConcepts(torch.tensor(np.load(cache/'text.npy'))),cfg['independent_heads'],cfg.get('concept_local',False))
    noise.load_state_dict(checkpoint['model']);noise.eval();rows=[]
    ids=splits['val'][arrays['mask_valid'][splits['val'],18]]
    with torch.no_grad():
        for start in range(0,len(ids),64):
            ii=ids[start:start+64];attr=np.array(arrays['attrs'][ii]);ks=attr.argmax(-1)
            tokens=torch.tensor(np.array(arrays['tokens'][ii]),dtype=torch.float32)
            mu,sigma,gate=noise.distribution(tokens,torch.tensor(ks))
            objects=[];parts=[]
            for i,k in zip(ii,ks):
                p=Path(records[i]['path'])
                with p.with_name(p.stem+'_attr_dict.pkl').open('rb') as f:d=pickle.load(f)
                objects.append(np.asarray(mask_from_uri(d['entire-object']).resize((224,224),Image.Resampling.NEAREST))>0)
                parts.append(np.asarray(mask_from_uri(d[ATTRS[k]]).resize((224,224),Image.Resampling.NEAREST))>0 if ATTRS[k] in d else np.zeros((224,224),dtype=bool))
            obj=torch.tensor(np.stack(objects));part=torch.tensor(np.stack(parts))
            s2=sigma.square().mean(1);mr=mu.square().mean(1).sqrt()
            for j,i in enumerate(ii):
                fg=obj[j];bg=~fg
                if not fg.any() or not bg.any():continue
                row=dict(id=int(i),concept=int(ks[j]),fg_variance=float(s2[j][fg].mean()),bg_variance=float(s2[j][bg].mean()),
                         fg_mu_rms=float(mr[j][fg].mean()),bg_mu_rms=float(mr[j][bg].mean()))
                if part[j].any() and (fg&~part[j]).any():
                    row['concept_mu_rms']=float(mr[j][part[j]].mean());row['other_object_mu_rms']=float(mr[j][fg&~part[j]].mean())
                rows.append(row)
    summary={k:float(np.mean([r[k] for r in rows])) for k in ['fg_variance','bg_variance','fg_mu_rms','bg_mu_rms']}
    summary.update(n=len(rows),sigma2_bg_over_fg=summary['bg_variance']/summary['fg_variance'],
                   mu_fg_over_bg=summary['fg_mu_rms']/summary['bg_mu_rms'],checkpoint_epoch=checkpoint['epoch'],
                   checkpoint_sha256=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(),
                   evaluation='all validation with valid nonempty foreground/background; first present native concept; GT only for statistics')
    partrows=[r for r in rows if 'concept_mu_rms' in r]
    summary['part_localization']=dict(n=len(partrows),concept_mu_rms=float(np.mean([r['concept_mu_rms'] for r in partrows])),other_object_mu_rms=float(np.mean([r['other_object_mu_rms'] for r in partrows])))
    (run/'energy_audit.json').write_text(json.dumps(dict(summary=summary,per_image=rows),indent=2));print(summary,flush=True)


if __name__=='__main__':main()
