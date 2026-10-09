"""Audit whether better concept masks are actually used for concept recognition.

The existing concept head adds a global CLS and a spatial-pool contribution.
AP is invariant to per-concept constant shifts, so the isolated contributions
can be compared without inventing a new threshold or refitting on the test set.
They are diagnostics of the existing joint head, not standalone trained models.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from .data import ATTRS
from .models import SpatialConcepts
from .train_spatial import load_cache


def main():
    torch.set_num_threads(6);cache=Path('artifacts/rival_vitl14')
    arrays,splits,_,_=load_cache(cache);output={}
    for name in ['spatial_clean','spatial_local']:
        run=Path('artifacts')/name
        if not (run/'results.json').exists():continue
        ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        model=SpatialConcepts(torch.tensor(np.load(cache/'text.npy')))
        model.load_state_dict(ck['model']);model.eval();components={'CLS':[],'region':[],'combined':[]}
        with torch.no_grad():
            for start in range(0,len(splits['test']),64):
                ids=splits['test'][start:start+64]
                tokens=torch.tensor(np.array(arrays['tokens'][ids]),dtype=torch.float32)
                z,_,scores,_,_=model.spatial(tokens);d=z.shape[-1]
                pooled=torch.einsum('bkp,bpd->bkd',(scores*2).softmax(-1),z[:,1:])
                global_logits=z[:,0]@model.logit_weights[:,:d].T
                region_logits=(pooled*model.logit_weights[None,:,d:]).sum(-1)
                full=global_logits+region_logits+model.logit_bias
                actual,_,_=model(tokens)
                assert torch.allclose(full,actual,atol=1e-5,rtol=1e-5)
                for key,value in [('CLS',global_logits),('region',region_logits),('combined',full)]:components[key].append(value.numpy())
        y=np.array(arrays['attrs'][splits['test']]);components={k:np.concatenate(v) for k,v in components.items()}
        ap={k:average_precision_score(y,v,average=None) for k,v in components.items()}
        w=model.logit_weights.detach().numpy();d=w.shape[1]//2
        cls_norm=np.linalg.norm(w[:,:d],axis=1);region_norm=np.linalg.norm(w[:,d:],axis=1)
        output[name]=dict(checkpoint_sha256=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(),epoch=ck['epoch'],n_test=len(y),
            mAP={k:float(v.mean()) for k,v in ap.items()},
            per_concept=[dict(name=attr,AP={key:float(value[k]) for key,value in ap.items()},
                              CLS_weight_norm=float(cls_norm[k]),region_weight_norm=float(region_norm[k]),
                              CLS_logit_std=float(components['CLS'][:,k].std()),region_logit_std=float(components['region'][:,k].std())) for k,attr in enumerate(ATTRS)])
        print(name,output[name]['mAP'],flush=True)
    dest=Path('reports/spatial_noise_v1/readout_path_audit.json')
    dest.write_text(json.dumps(dict(models=output,interpretation='Ablations of additive contributions in the existing fitted head, not retrained standalone models. Better masks alone do not establish that the recognition/classification path relies on the indicated region. Spatial CLIP tokens are contextualized and can also contain global information.'),indent=2))


if __name__=='__main__':main()
