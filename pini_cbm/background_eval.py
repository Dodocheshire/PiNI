"""Fixed test background intervention; no GT masks enter learned inference.

GT masks construct the external intervention only. Donor foreground is removed
before transplantation, preventing a second class object from being introduced.
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from clip import clip
from .data import mask_from_uri,normalize
from .models import ConditionalRGBNoise,SpatialConcepts
from .train_noise import fixed_subset,native_metrics
from .train_spatial import load_cache,head_predict,batch_tensor


def object_mask(record):
    p=Path(record['path'])
    with p.with_name(p.stem+'_attr_dict.pkl').open('rb') as f:d=pickle.load(f)
    im=mask_from_uri(d['entire-object']).resize((224,224),Image.Resampling.NEAREST)
    return (np.asarray(im)>0)[None]


def pair_ci(a,b,seed=1149):
    diff=a.astype(float)-b.astype(float);rng=np.random.default_rng(seed)
    vals=[diff[rng.integers(0,len(diff),size=len(diff))].mean() for _ in range(2000)]
    return dict(difference_pp=float(diff.mean()*100),CI95_pp=(np.quantile(vals,[.025,.975])*100).tolist())


@torch.no_grad()
def real_view_predictions(noise,backbone,images,tokens,mode,suppress,batch,seed=6101):
    torch.manual_seed(seed);noise.eval();probs=np.empty((len(images),18),dtype='float32')
    for start in range(0,len(images)*18,batch):
        flat=np.arange(start,min(start+batch,len(images)*18));pos=flat//18;ks=flat%18
        rgb=torch.tensor(images[pos],dtype=torch.float32,device='cuda');tok=torch.tensor(tokens[pos],device='cuda').float()
        k=torch.tensor(ks,device='cuda');x,_,_,_=noise.perturb(rgb,tok,k,mode=mode,suppress=suppress)
        with torch.autocast('cuda',dtype=torch.float16):
            z=backbone.encode_image(normalize(x),spatial=True);l,_,_=noise.concepts(z)
        probs[pos,ks]=l[torch.arange(len(k),device='cuda'),k].float().sigmoid().cpu().numpy()
    return probs


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',default=['artifacts/mean_sequential','artifacts/noise_sequential'])
    p.add_argument('--size',type=int,default=1000);p.add_argument('--batch',type=int,default=128)
    p.add_argument('--out',default='artifacts/background_intervention');a=p.parse_args()
    torch.set_num_threads(6);torch.manual_seed(42);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cache=Path('artifacts/rival_vitl14');arrays,splits,labels,records=load_cache(cache)
    rgb=np.load(cache/'rgb.npy',mmap_mode='r')
    eligible=splits['test'][arrays['mask_valid'][splits['test'],18]]
    ids=fixed_subset(eligible,labels,a.size,seed=46)
    donor_pool=splits['val'][arrays['mask_valid'][splits['val'],18]];rng=np.random.default_rng(47)
    donors=np.array([rng.choice(donor_pool[labels[donor_pool]!=labels[i]]) for i in ids])
    originals=np.array(rgb[ids],dtype='float32')/255;shifted=[]
    for j,(i,d) in enumerate(zip(ids,donors)):
        fg=object_mask(records[i]);dfg=object_mask(records[d]);background=np.array(rgb[d],dtype='float32')/255
        background=background*(1-dfg)+dfg*.5
        shifted.append(originals[j]*fg+background*(1-fg))
    shifted=np.stack(shifted).astype('float32')
    np.savez_compressed(out/'intervention.npz',ids=ids,donors=donors,shifted=shifted)
    (out/'protocol.json').write_text(json.dumps(dict(ids=ids.tolist(),donors=donors.tolist(),sampling_seeds=[46,47,6101],
        construction='GT masks only for external background intervention; donor object replaced by RGB 0.5; target object pixels preserved',
        inference_gt_mask_inputs=False),indent=2))
    backbone,_=clip.load('ViT-L/14',device='cuda',jit=False);backbone.float().eval().requires_grad_(False)
    tokens={}
    with torch.no_grad():
        for name,images in [('original',originals),('background_swap',shifted)]:
            rows=[]
            for start in range(0,len(images),a.batch):
                x=torch.tensor(images[start:start+a.batch],device='cuda')
                with torch.autocast('cuda',dtype=torch.float16):z=backbone.encode_image(normalize(x),spatial=True)
                rows.append(z.cpu().numpy())
            tokens[name]=np.concatenate(rows)
    models={};clean=SpatialConcepts(torch.tensor(np.load(cache/'text.npy'),device='cuda')).cuda()
    clean.load_state_dict(torch.load('artifacts/spatial_clean/best.pt',map_location='cuda',weights_only=False)['model']);clean.eval()
    head=torch.load('artifacts/spatial_clean/matched_6000_continuous_head.pt',map_location='cpu',weights_only=False)
    models['clean']={}
    with torch.no_grad():
        for name in tokens:
            rows=[]
            for start in range(0,len(ids),a.batch):
                l,_,_=clean(torch.tensor(tokens[name][start:start+a.batch],device='cuda').float());rows.append(l.sigmoid().cpu().numpy())
            probs=np.concatenate(rows);logits=head_predict(head,probs);correct=logits.argmax(1)==labels[ids]
            np.save(out/('clean_'+name+'_probs.npy'),probs);np.save(out/('clean_'+name+'_correct.npy'),correct)
            models['clean'][name]=dict(accuracy=float(correct.mean()),**native_metrics(probs,np.array(arrays['attrs'][ids])))
    for path in a.runs:
        run=Path(path);checkpoint=torch.load(run/'best.pt',map_location='cuda',weights_only=False);cfg=checkpoint['args']
        noise=ConditionalRGBNoise(SpatialConcepts(torch.tensor(np.load(cache/'text.npy'),device='cuda')),cfg['independent_heads'],cfg.get('concept_local',False)).cuda()
        noise.load_state_dict(checkpoint['model']);head=torch.load(run/'head.pt',map_location='cpu',weights_only=False)
        modes=[cfg['mode']]+(['mean'] if cfg['mode']=='noise' else [])
        for mode in modes:
            key=run.name+('_same_checkpoint_mean' if mode!=cfg['mode'] else '')
            models[key]={}
            for name,images in [('original',originals),('background_swap',shifted)]:
                probs=real_view_predictions(noise,backbone,images,tokens[name],mode,cfg['suppress'],a.batch)
                logits=head_predict(head,probs);correct=logits.argmax(1)==labels[ids]
                np.save(out/(key+'_'+name+'_probs.npy'),probs);np.save(out/(key+'_'+name+'_correct.npy'),correct)
                models[key][name]=dict(accuracy=float(correct.mean()),**native_metrics(probs,np.array(arrays['attrs'][ids])))
                print(json.dumps(dict(model=key,condition=name,**models[key][name])),flush=True)
    comparisons={}
    for key in models:
        original=np.load(out/(key+'_original_correct.npy'));swapped=np.load(out/(key+'_background_swap_correct.npy'))
        comparisons[key+'_swap_minus_original']=pair_ci(swapped,original)
        if key!='clean':comparisons[key+'_swap_minus_clean']=pair_ci(swapped,np.load(out/'clean_background_swap_correct.npy'))
    result=dict(n=len(ids),models=models,paired=comparisons,strict_external_intervention=True)
    (out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
