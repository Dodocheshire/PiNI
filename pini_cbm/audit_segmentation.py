"""Exact native mask area IoU for the frozen clean checkpoint, CPU only."""
import json
from pathlib import Path
import numpy as np
import torch
from .models import SpatialConcepts
from .train_spatial import load_cache


def main():
    torch.set_num_threads(2);cache=Path('artifacts/rival_vitl14');out=Path('artifacts/spatial_clean')
    arrays,splits,labels,records=load_cache(cache)
    model=SpatialConcepts(torch.tensor(np.load(cache/'text.npy')))
    model.load_state_dict(torch.load(out/'best.pt',map_location='cpu',weights_only=False)['model']);model.eval()
    result=json.loads((out/'results.json').read_text());audit={}
    with torch.no_grad():
        for split,ids in splits.items():
            objects=[];parts=[]
            for start in range(0,len(ids),128):
                ii=ids[start:start+128];tokens=torch.tensor(np.array(arrays['tokens'][ii]),dtype=torch.float32)
                l,m,f=model(tokens);pred=(torch.cat([m,f[:,None]],1)>0).float()
                gt=torch.tensor(np.array(arrays['masks'][ii]),dtype=torch.float32)
                valid=torch.tensor(np.array(arrays['mask_valid'][ii]));present=torch.tensor(np.array(arrays['attrs'][ii]),dtype=torch.bool)&valid[:,:18]
                inter=(gt*pred).sum((-1,-2));union=(gt+pred-gt*pred).sum((-1,-2)).clamp_min(1e-6)
                iou=inter/union;objects.extend(iou[:,18][valid[:,18]].tolist());parts.extend(iou[:,:18][present].tolist())
            audit[split]=dict(fg_IoU=float(np.mean(objects)),concept_mask_IoU=float(np.mean(parts)))
            result['concepts'][split].update(audit[split]);print(split,audit[split],flush=True)
    audit['definition']='Exact area of original 224x224 native binary mask against nearest-upsampled binary 16x16 predicted mask, computed via fractional patch occupancy; includes tiny concept regions.'
    (out/'segmentation_audit.json').write_text(json.dumps(audit,indent=2));result['mask_iou_definition']=audit['definition']
    (out/'results.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
