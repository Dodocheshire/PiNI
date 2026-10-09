"""Evaluate the stronger clean baselines on the identical saved intervention.

This does not retrain a model, reconstruct donors, or repeat noisy RGB views.
It closes the missing locality/region clean comparisons with matched 6003 heads.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from clip import clip
from .data import normalize
from .models import SpatialConcepts
from .train_noise import native_metrics
from .train_spatial import load_cache,head_predict


def main():
    torch.set_num_threads(4);cache=Path('artifacts/rival_vitl14')
    reference=Path('artifacts/background_intervention');out=Path('artifacts/background_clean_models');out.mkdir(parents=True,exist_ok=True)
    arrays,splits,labels,_=load_cache(cache);saved=np.load(reference/'intervention.npz')
    ids=saved['ids'];donors=saved['donors'];shifted=saved['shifted']
    rgb=np.load(cache/'rgb.npy',mmap_mode='r');original=np.array(rgb[ids],dtype='float32')/255
    backbone,_=clip.load('ViT-L/14',device='cuda',jit=False);backbone.float().eval().requires_grad_(False)
    tokens={}
    with torch.no_grad():
        for condition,images in [('original',original),('background_swap',shifted)]:
            rows=[]
            for start in range(0,len(ids),128):
                x=torch.tensor(images[start:start+128],device='cuda')
                with torch.autocast('cuda',dtype=torch.float16):z=backbone.encode_image(normalize(x),spatial=True)
                rows.append(z.cpu())
            tokens[condition]=torch.cat(rows)
    result=dict(n=len(ids),reference_ids=ids.tolist(),reference_donors=donors.tolist(),models={},
                annotation_role='GT masks only constructed the saved external intervention; learned clean forward receives image tokens only.',
                protocol='Same saved target/donor pixels and batch128 frozen ViT-L/14 encoder as existing noisy intervention; matched6003 sparse heads.')
    for name,folder in [('clean','spatial_clean'),('local_clean','spatial_local'),('region_clean','spatial_region')]:
        run=Path('artifacts')/folder;ck=torch.load(run/'best.pt',map_location='cuda',weights_only=False)
        model=SpatialConcepts(torch.tensor(np.load(cache/'text.npy'),device='cuda')).cuda();model.load_state_dict(ck['model']);model.eval()
        head=torch.load(run/'matched_6000_continuous_head.pt',map_location='cpu',weights_only=False)
        result['models'][name]={}
        with torch.no_grad():
            for condition,features in tokens.items():
                rows=[]
                for start in range(0,len(ids),128):
                    logits,_,_=model(features[start:start+128].cuda().float());rows.append(logits.sigmoid().cpu().numpy())
                probs=np.concatenate(rows);correct=head_predict(head,probs).argmax(1)==labels[ids]
                np.save(out/(name+'_'+condition+'_probs.npy'),probs);np.save(out/(name+'_'+condition+'_correct.npy'),correct)
                result['models'][name][condition]=dict(accuracy=float(correct.mean()),**native_metrics(probs,np.array(arrays['attrs'][ids])))
                if name=='clean':
                    delta=float(np.abs(probs-np.load(reference/('clean_'+condition+'_probs.npy'))).max())
                    assert delta<1e-4,delta
                    assert np.array_equal(correct,np.load(reference/('clean_'+condition+'_correct.npy')))
                    result['models'][name][condition]['reference_probability_max_difference']=delta
        result['models'][name]['checkpoint_sha256']=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()
        print(name,result['models'][name],flush=True)
    temp=out/'results.tmp';temp.write_text(json.dumps(result,indent=2));temp.replace(out/'results.json')
    report=Path('reports/spatial_noise_v1');(report/'background_clean_models.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
