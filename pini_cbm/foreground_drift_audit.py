"""Check whether RGB adaptation preserves the pretrained foreground blocker."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .models import ConditionalRGBNoise,SpatialConcepts
from .train_spatial import load_cache


def main():
    torch.set_num_threads(4);cache=Path('artifacts/rival_vitl14');arrays,splits,_,_=load_cache(cache)
    ids=splits['val'][np.array(arrays['mask_valid'][splits['val'],18]).astype(bool)]
    gt=np.array(arrays['masks'][ids,18],dtype='float32');fgmass=gt.sum((1,2));bgmass=(1-gt).sum((1,2))
    names=['spatial_clean','spatial_local','spatial_region','mean_sequential','noise_sequential','mean_separate_heads','noise_separate_heads',
           'local_mean','local_noise','full_joint_mean','full_joint_noise','region_mean','region_noise','fixed_mean','fixed_noise','region_seed43_mean','region_seed43_noise']
    destination=Path('reports/spatial_noise_v1/foreground_drift.json')
    prior=json.loads(destination.read_text()) if destination.exists() else {'models':{}}
    protocol=hashlib.sha256(Path(__file__).read_bytes()+Path('pini_cbm/models.py').read_bytes()+
        (cache/'COMPLETE.json').read_bytes()+ids.tobytes()+gt.tobytes()+
        repr([(k,arrays[k].shape,Path(arrays[k].filename).stat().st_size,Path(arrays[k].filename).stat().st_mtime_ns) for k in arrays]).encode()).hexdigest()
    result=dict(n_validation=len(ids),all_foreground_IoU=float(gt.mean((1,2)).mean()),models={},
                protocol='Fixed valid validation foreground masks. Fractional native area gives exact IoU for nearest-upsampled16x16 binary predictions. GT is an audit reference only; detector receives cached image tokens.',
                interpretation='Generator foreground branch on original inputs, before RGB perturbation; no class/threshold selection.')
    for name in names:
        run=Path('artifacts')/name
        if not (run/'results.json').exists():continue
        checkpoint_hash=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()
        old=prior['models'].get(name,{})
        if old.get('checkpoint_sha256')==checkpoint_hash and old.get('protocol_sha256')==protocol:
            result['models'][name]=old;print(name,'validated cached foreground audit',flush=True);continue
        ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        concepts=SpatialConcepts(torch.tensor(np.load(cache/'text.npy')))
        model=None
        if 'concepts.texts' in ck['model']:
            cfg=ck['args'];model=ConditionalRGBNoise(concepts,cfg['independent_heads'],cfg.get('concept_local',False))
            model.load_state_dict(ck['model']);concepts=model.concepts
        else:concepts.load_state_dict(ck['model'])
        concepts.eval();rows=[]
        with torch.no_grad():
            for start in range(0,len(ids),128):
                tokens=torch.tensor(np.array(arrays['tokens'][ids[start:start+128]]),dtype=torch.float32)
                p=model.blocker(tokens) if model is not None and model.fixed_foreground else concepts.spatial(tokens)[-1].sigmoid().reshape(-1,16,16)
                rows.append(p.numpy())
        prob=np.concatenate(rows);pred=prob>=.5;intersection=(gt*pred).sum((1,2));union=(gt+pred-gt*pred).sum((1,2))
        valid=(fgmass>0)&(bgmass>0)
        result['models'][name]=dict(checkpoint_sha256=checkpoint_hash,protocol_sha256=protocol,
            foreground_IoU=float((intersection/np.maximum(union,1e-6)).mean()),
            foreground_probability_on_true_foreground=float(((prob*gt).sum((1,2))[valid]/fgmass[valid]).mean()),
            foreground_probability_on_true_background=float(((prob*(1-gt)).sum((1,2))[valid]/bgmass[valid]).mean()),
            false_foreground_fraction_of_background=float(((pred*(1-gt)).sum((1,2))[valid]/bgmass[valid]).mean()))
        print(name,result['models'][name],flush=True)
    destination.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
