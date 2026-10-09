"""RGB-only inference for a trained spatial-noise CBM, no annotations/cache."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from clip import clip
from clip.model import convert_weights

from .data import ATTRS,CLASSES,normalize
from .models import ConditionalRGBNoise,SpatialConcepts


class SpatialNoiseCBM:
    def __init__(self,run,device='cuda'):
        self.device=torch.device(device);self.run=Path(run)
        ck=torch.load(self.run/'best.pt',map_location='cpu',weights_only=False);state=ck['model'];cfg=ck['args']
        texts=state['concepts.texts'];width=state['concepts.patch.weight'].shape[0]
        self.noise=ConditionalRGBNoise(SpatialConcepts(texts,width),cfg['independent_heads'],cfg.get('concept_local',False))
        self.noise.load_state_dict(state);self.noise.to(self.device).eval()
        if cfg['order']=='independent':self.noise.concepts.requires_grad_(False)
        # Match archived evaluation's parameter metadata. FP16 autocast caches
        # eligible gradient-bearing weight casts; toggling requires_grad here
        # changes that numerical path even inside no_grad. predict() disables
        # autograd and never updates weights.
        self.head={k:v.to(self.device) for k,v in torch.load(self.run/'head.pt',map_location='cpu',weights_only=False).items()}
        assert self.head['weight'].shape==(10,18) and self.head['center'].shape==(18,) and self.head['scale'].shape==(18,)
        self.suppress=cfg['suppress']
        self.backbone,_=clip.load('ViT-L/14',device=self.device,jit=False)
        self.backbone.float().eval().requires_grad_(False)
        # The immutable original-token cache used CLIP's default selective
        # FP16 conversion. Perturbed views used FP32 masters + FP16 autocast.
        # Reproduce both protocols; model.half() also casts LayerNorm weights
        # and is not equivalent to CLIP's original convert_weights().
        self.original_backbone=copy.deepcopy(self.backbone)
        if self.device.type=='cuda':convert_weights(self.original_backbone)

    def encode(self,rgb,original=False):
        if original:
            # The cache's source batch was 64. On this FP16 backend changing
            # batch shape changes rounding even with identical image bytes.
            # Padding is annotation-free and prevents serving-batch drift.
            rows=[]
            with torch.autocast(self.device.type,enabled=False):
                for start in range(0,len(rgb),64):
                    x=rgb[start:start+64];n=len(x)
                    if n<64:x=torch.cat([x,x[-1:].expand(64-n,-1,-1,-1)])
                    rows.append(self.original_backbone.encode_image(normalize(x),spatial=True)[:n])
            return torch.cat(rows)
        with torch.autocast(self.device.type,dtype=torch.float16,enabled=self.device.type=='cuda'):
            return self.backbone.encode_image(normalize(rgb),spatial=True)

    @torch.no_grad()
    def predict(self,rgb,mode='noise',draws=1,batch=128,seed=8101):
        assert mode in ['mean','noise'] and draws>=1 and rgb.ndim==4 and rgb.shape[1:]==(3,224,224)
        assert batch==128,'Use the validated 128-view numerical protocol'
        byte_input=rgb.dtype==torch.uint8
        rgb=rgb.to(self.device,dtype=torch.float32).contiguous()
        # Cache/evaluation divides uint8 values on CUDA. CPU float division
        # can differ by one ulp, amplified by subsequent FP16 rounding.
        if byte_input:rgb=rgb/255
        assert torch.isfinite(rgb).all() and rgb.min()>=0 and rgb.max()<=1
        original=self.encode(rgb,original=True).float();concepts=torch.empty((len(rgb),18),device=self.device)
        torch.manual_seed(seed)
        for start in range(0,len(rgb)*18,batch):
            index=torch.arange(start,min(start+batch,len(rgb)*18),device=self.device);n=len(index)
            if n<batch:index=torch.cat([index,index[-1:].expand(batch-n)])
            pos=index//18;k=index%18
            tokens=original[pos];values=[]
            for _ in range(draws):
                x,_,_,_=self.noise.perturb(rgb[pos],tokens,k,mode=mode,suppress=self.suppress)
                with torch.autocast(self.device.type,dtype=torch.float16,enabled=self.device.type=='cuda'):
                    logits,_,_=self.noise.recognize(self.encode(x),tokens)
                values.append(logits[torch.arange(len(k),device=self.device),k].float().sigmoid())
            concepts[pos[:n],k[:n]]=torch.stack(values).mean(0)[:n]
        # This is the complete classification path. The original/noisy image
        # embeddings and foreground probabilities are not class-head inputs.
        standardized=(concepts-self.head['center'])/self.head['scale']
        classes=standardized@self.head['weight'].T+self.head['bias']
        return concepts.cpu().numpy(),classes.cpu().numpy()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('images',nargs='+')
    p.add_argument('--run',default='artifacts/region_noise');p.add_argument('--device',default='cuda')
    p.add_argument('--mode',choices=['mean','noise'],default='noise');p.add_argument('--draws',type=int,default=1)
    p.add_argument('--seed',type=int,default=8101);p.add_argument('--out');a=p.parse_args();torch.set_num_threads(4)
    rows=[]
    for name in a.images:
        with Image.open(name) as image:rows.append(np.asarray(image.convert('RGB').resize((224,224),Image.Resampling.BICUBIC)).copy())
    rgb=torch.tensor(np.stack(rows).transpose(0,3,1,2),dtype=torch.uint8)
    model=SpatialNoiseCBM(a.run,a.device);probs,logits=model.predict(rgb,a.mode,a.draws,seed=a.seed)
    output=dict(run=a.run,mode=a.mode,draws=a.draws,inference_inputs='RGB images only; no labels, masks, or token cache',
        numeric_protocol='CUDA original tokens: CLIP selective FP16, padded batch 64; perturbed views: FP32 masters with FP16 autocast, padded batch 128. CPU uses FP32. Partial-batch padding is a serving protocol; archived incomplete tail batches may round differently.',
        class_head_inputs='18 native concept probabilities only',images=[dict(path=name,predicted_class=CLASSES[int(scores.argmax())],
        concepts={attr:float(value) for attr,value in zip(ATTRS,values)}) for name,values,scores in zip(a.images,probs,logits)])
    if a.out:Path(a.out).write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':main()
