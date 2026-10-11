"""Image-only shared-view inference: one original and one perturbed encoding."""
import argparse
import json

import numpy as np
import torch
from PIL import Image

from clip import clip
from pini_cbm.data import normalize
from .model import SharedSpatialCBM


@torch.no_grad()
def infer_rgb(model,backbone,rgb,mode='noise',seed=8101):
    model.eval();backbone.eval()
    devices=[rgb.device.index] if rgb.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        with torch.autocast(rgb.device.type,dtype=torch.float16,enabled=rgb.is_cuda):
            original=backbone.encode_image(normalize(rgb),spatial=True)
        processed,mu,sigma,fg=model.perturb(rgb,original,mode)
        with torch.autocast(rgb.device.type,dtype=torch.float16,enabled=rgb.is_cuda):
            features=model.encode_view(backbone,normalize(processed),fg)
        concepts,_,_=model.recognize(features,foreground_prior=fg)
    return model.classify(concepts),concepts.sigmoid(),{'rgb':processed,'mean':mu,'std':sigma,'foreground':fg}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('images',nargs='+');p.add_argument('--checkpoint',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=8101)
    a=p.parse_args();torch.set_num_threads(6)
    checkpoint=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    model=SharedSpatialCBM.from_checkpoint(checkpoint).to(a.device).eval()
    backbone,_=clip.load('ViT-B/16',device=a.device,jit=False)
    backbone.float().eval().requires_grad_(False)
    rgb=[]
    for name in a.images:
        with Image.open(name) as im:
            rgb.append(np.asarray(im.convert('RGB').resize((224,224),Image.Resampling.BICUBIC)).transpose(2,0,1).copy())
    images=torch.as_tensor(np.stack(rgb),device=a.device).float()/255
    logits,concepts,_=infer_rgb(model,backbone,images,checkpoint['method'],a.seed)
    for index,name in enumerate(a.images):
        print(json.dumps({'image':name,'class_index':int(logits[index].argmax()),
                          'concepts':concepts[index].cpu().tolist(),'image_encodes':2,
                          'spatial_noise_prior':model.spatial_noise_prior}))


if __name__=='__main__':main()
