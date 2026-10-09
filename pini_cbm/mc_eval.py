"""MC8 on a validation-selected stochastic CBM; original head unchanged."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from clip import clip
from .models import ConditionalRGBNoise,SpatialConcepts
from .train_noise import predict_views,native_metrics
from .train_spatial import load_cache,head_predict


def main():
    p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--draws',type=int,default=8)
    p.add_argument('--selection-description');a=p.parse_args()
    if a.run:run=Path(a.run)
    else:
        names=['noise_independent','noise_sequential','noise_joint','noise_separate_heads']
        candidates=[]
        for name in names:
            path=Path('artifacts')/name
            if (path/'results.json').exists():
                result=json.loads((path/'results.json').read_text());candidates.append((result['concepts']['val']['mAP'],path))
        assert len(candidates)==len(names),'Complete all V1 orders before validation selection'
        run=max(candidates,key=lambda x:x[0])[1]
    torch.set_num_threads(6);cache=Path('artifacts/rival_vitl14');arrays,splits,labels,records=load_cache(cache)
    rgb=np.load(cache/'rgb.npy',mmap_mode='r');ck=torch.load(run/'best.pt',map_location='cuda',weights_only=False);cfg=ck['args']
    noise=ConditionalRGBNoise(SpatialConcepts(torch.tensor(np.load(cache/'text.npy'),device='cuda')),cfg['independent_heads'],cfg.get('concept_local',False)).cuda()
    noise.load_state_dict(ck['model']);backbone,_=clip.load('ViT-L/14',device='cuda',jit=False);backbone.float().eval().requires_grad_(False)
    probs=predict_views(noise,backbone,arrays,rgb,splits['test'],'noise',cfg['suppress'],draws=a.draws)
    head=torch.load(run/'head.pt',map_location='cpu',weights_only=False);logits=head_predict(head,probs)
    np.save(run/('test_MC'+str(a.draws)+'_probs.npy'),probs);np.save(run/('test_MC'+str(a.draws)+'_class_logits.npy'),logits)
    key='MC'+str(a.draws);result=json.loads((run/'results.json').read_text())
    description=a.selection_description or ('Explicit run supplied' if a.run else 'highest complete V1 validation native mAP')
    result[key]=dict(accuracy=float((logits.argmax(1)==labels[splits['test']]).mean()),**native_metrics(probs,np.array(arrays['attrs'][splits['test']])),
                    selection=description+'; original fitted head, no MC test tuning')
    (run/'results.json').write_text(json.dumps(result,indent=2));print(run,result[key],flush=True)


if __name__=='__main__':main()
