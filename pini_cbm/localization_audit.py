"""All positive validation concept regions, including fine-grained locality.

Native PNGs measure exact 224-pixel area. GT is exclusively an audit reference;
the generator receives image tokens and the enumerated concept ID. Attributes
covering >=95% of the object are excluded only from the explicitly local metric.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from .data import ATTRS,mask_from_uri
from .models import ConditionalRGBNoise,SpatialConcepts
from .train_spatial import load_cache


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
    torch.set_num_threads(2);run=Path(a.run);cache=Path('artifacts/rival_vitl14')
    fingerprint=hashlib.sha256((run/'best.pt').read_bytes()+Path(__file__).read_bytes()).hexdigest()
    dst=run/'localization_audit.json'
    if dst.exists() and json.loads(dst.read_text()).get('input_sha256')==fingerprint:
        print('Validated full concept localization audit reused',flush=True);return
    arrays,splits,_,records=load_cache(cache);ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False);cfg=ck['args']
    model=ConditionalRGBNoise(SpatialConcepts(torch.tensor(np.load(cache/'text.npy'))),cfg['independent_heads'],cfg.get('concept_local',False))
    model.load_state_dict(ck['model']);model.eval();rows=[]
    with torch.no_grad():
        for n,i in enumerate(splits['val']):
            ks=np.flatnonzero((arrays['attrs'][i]>0)&arrays['mask_valid'][i,:18])
            if not arrays['mask_valid'][i,18] or not len(ks):continue
            path=Path(records[i]['path'])
            with path.with_name(path.stem+'_attr_dict.pkl').open('rb') as f:gt=pickle.load(f)
            decode=lambda name:np.asarray(mask_from_uri(gt[name]).resize((224,224),Image.Resampling.NEAREST))>0
            obj=decode('entire-object')
            if not obj.any() or obj.all():continue
            tokens=torch.tensor(np.array(arrays['tokens'][[i]*len(ks)]),dtype=torch.float32)
            mu,sigma,_=model.distribution(tokens,torch.tensor(ks));mr=mu.square().mean(1).sqrt().numpy();s2=sigma.square().mean(1).numpy()
            _,_,_,masks,_=model.concepts.spatial(tokens)
            pred=F.interpolate((masks[torch.arange(len(ks)),torch.tensor(ks)].sigmoid()>=.5).float().reshape(-1,1,16,16),(224,224),mode='nearest')[:,0].numpy()>0
            for j,k in enumerate(ks):
                part=decode(ATTRS[k]);intersection=part&obj;other=obj&~part
                if not part.any():continue
                row=dict(id=int(i),concept=ATTRS[k],positive_IoU=float((pred[j]&part).sum()/max(1,(pred[j]|part).sum())),
                    fg_mu_rms=float(mr[j][obj].mean()),bg_mu_rms=float(mr[j][~obj].mean()),
                    fg_variance=float(s2[j][obj].mean()),bg_variance=float(s2[j][~obj].mean()),
                    fraction_object_covered=float(intersection.sum()/obj.sum()))
                if intersection.any() and other.any() and row['fraction_object_covered']<.95:
                    row.update(part_mu_rms=float(mr[j][intersection].mean()),other_object_mu_rms=float(mr[j][other].mean()),
                               part_variance=float(s2[j][intersection].mean()),other_object_variance=float(s2[j][other].mean()))
                rows.append(row)
            if n%250==0:print('validation images',n,'positive concept pairs',len(rows),flush=True)
    def aggregate(items):
        local=[r for r in items if 'part_mu_rms' in r]
        result=dict(n_positive_pairs=len(items),n_local_pairs=len(local),positive_concept_IoU=float(np.mean([r['positive_IoU'] for r in items])))
        for keys,name in [(('bg_variance','fg_variance'),'sigma2_bg_over_fg'),(('fg_mu_rms','bg_mu_rms'),'mu_fg_over_bg')]:
            result[name]=float(np.mean([r[keys[0]] for r in items])/max(1e-12,np.mean([r[keys[1]] for r in items])))
        if local:
            result['local_part_mu_over_other_object']=float(np.mean([r['part_mu_rms'] for r in local])/max(1e-12,np.mean([r['other_object_mu_rms'] for r in local])))
            result['local_other_object_sigma2_over_part']=float(np.mean([r['other_object_variance'] for r in local])/max(1e-12,np.mean([r['part_variance'] for r in local])))
        return result
    result=dict(input_sha256=fingerprint,checkpoint_epoch=ck['epoch'],summary=aggregate(rows),
                per_concept={name:aggregate([r for r in rows if r['concept']==name]) for name in ATTRS if any(r['concept']==name for r in rows)},
                per_pair=rows,inference_gt_mask_inputs=False,
                protocol='All native positive/valid concepts on validation images with nonempty foreground/background. IoU uses native 224 masks, nearest 16-patch predictions. Local parts use object intersection, and cover less than 95% of the object; means compare physical RGB fields without rescaling.')
    dst.write_text(json.dumps(result,indent=2));print(result['summary'],flush=True)


if __name__=='__main__':main()
