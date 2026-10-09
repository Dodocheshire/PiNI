"""Where does sampled noise actually affect native recognition?

Fixed validation images; same epsilon across full, foreground-only and
background-only perturbations. Native GT foreground constructs the external
intervention only. The learned generator always receives its ordinary inputs.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from clip import clip
from .data import mask_from_uri,normalize
from .models import ConditionalRGBNoise,SpatialConcepts
from .train_noise import fixed_subset
from .train_spatial import batch_tensor,load_cache


def evaluate(run,backbone,arrays,splits,labels,records,args):
    run=Path(run);cache=Path(args.cache)
    ck=torch.load(run/'best.pt',map_location='cuda',weights_only=False);cfg=ck['args']
    model=ConditionalRGBNoise(SpatialConcepts(torch.tensor(np.load(cache/'text.npy'),device='cuda')),
                              cfg['independent_heads'],cfg.get('concept_local',False)).cuda()
    model.load_state_dict(ck['model']);model.eval()
    eligible=splits['val'][np.array(arrays['mask_valid'][splits['val'],18]).astype(bool)]
    selected=fixed_subset(eligible,labels,args.images,seed=2038);ids=[];masks=[]
    for i in selected:
        path=Path(records[i]['path'])
        with path.with_name(path.stem+'_attr_dict.pkl').open('rb') as f:gt=pickle.load(f)
        mask=np.asarray(mask_from_uri(gt['entire-object']).resize((224,224),Image.Resampling.NEAREST))>0
        if not mask.any() or mask.all():continue
        ids.append(int(i));masks.append(mask)
    ids=np.array(ids);masks=np.stack(masks);n=len(ids);assert n>0
    rgb=np.load(cache/'rgb.npy',mmap_mode='r');modes=['all','foreground','background']
    samples={key:np.empty((args.draws,n,18),dtype=np.float32) for key in modes}
    baseline=np.empty((n,18),dtype=np.float32);energies=np.empty((n,18,2),dtype=np.float32)
    torch.manual_seed(args.seed)
    def recognize(x,k,original_tokens):
        with torch.autocast('cuda',dtype=torch.float16):
            tokens=backbone.encode_image(normalize(x),spatial=True)
            logits,_,_=model.recognize(tokens,original_tokens)
            return logits[torch.arange(len(k),device='cuda'),k].float().cpu().numpy()
    with torch.no_grad():
        flat=np.arange(n*18)
        for start in range(0,len(flat),args.batch):
            index=flat[start:start+args.batch];pos=index//18;ks=index%18;ii=ids[pos]
            tokens=batch_tensor(arrays['tokens'],ii);images=batch_tensor(rgb,ii)/255
            k=torch.tensor(ks,device='cuda');fg=torch.tensor(masks[pos,None],device='cuda')
            mean,_,sigma,_=model.perturb(images,tokens,k,mode='mean',suppress=cfg['suppress'])
            baseline[pos,ks]=recognize(mean,k,tokens)
            s2=sigma.square();energies[pos,ks,0]=(s2*fg).sum((1,2,3)).cpu().numpy()
            energies[pos,ks,1]=(s2*(~fg)).sum((1,2,3)).cpu().numpy()
            for draw in range(args.draws):
                delta=sigma*torch.randn_like(images)
                for key,location in [('all',1),('foreground',fg),('background',~fg)]:
                    samples[key][draw,pos,ks]=recognize(mean+delta*location,k,tokens)
            print(json.dumps(dict(run=str(run),views_done=min(start+args.batch,n*18),total=n*18)),flush=True)
    stats={}
    for key,value in samples.items():
        prob=1/(1+np.exp(-np.clip(value,-30,30)));base=1/(1+np.exp(-np.clip(baseline,-30,30)))
        variance=value.var(0,ddof=1)
        stats[key]=dict(mean_native_logit_variance=float(variance.mean()),
                       RMS_native_probability_fluctuation=float(np.sqrt(prob.var(0,ddof=1).mean())),
                       mean_absolute_probability_shift_from_mean=float(np.abs(prob.mean(0)-base).mean()))
    result=dict(run=str(run),checkpoint_sha256=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(),
                n_validation_images=n,n_native_views=n*18,draws=args.draws,seed=args.seed,ids=ids.tolist(),modes=stats,
                mean_foreground_RGB_noise_energy=float(energies[:,:,0].mean()),
                mean_background_RGB_noise_energy=float(energies[:,:,1].mean()),
                background_over_foreground_output_variance=stats['background']['mean_native_logit_variance']/max(1e-12,stats['foreground']['mean_native_logit_variance']),
                annotations='Native foreground masks construct external noise-location interventions only; unchanged generator receives tokens and concept IDs, never GT masks.',
                interpretation='Fixed validation influence diagnostic, not test accuracy or proof of causality in natural backgrounds. Same epsilon for all locations. Output variance depends on both injected energy and recognition sensitivity; high sigma on background does not itself prove background is ignored.')
    out=Path('reports/spatial_noise_v1');out.mkdir(parents=True,exist_ok=True)
    (out/(run.name+'_noise_region_intervention.json')).write_text(json.dumps(result,indent=2))
    np.savez_compressed(run/'noise_region_intervention_logits.npz',baseline=baseline,ids=ids,**samples)
    print(result,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--cache',default='artifacts/rival_vitl14');p.add_argument('--images',type=int,default=64)
    p.add_argument('--draws',type=int,default=8);p.add_argument('--batch',type=int,default=128);p.add_argument('--seed',type=int,default=9916)
    args=p.parse_args();assert args.draws>1;torch.set_num_threads(4)
    arrays,splits,labels,records=load_cache(args.cache)
    backbone,_=clip.load('ViT-L/14',device='cuda',jit=False);backbone.float().eval().requires_grad_(False)
    for run in args.runs:evaluate(run,backbone,arrays,splits,labels,records,args)


if __name__=='__main__':main()
