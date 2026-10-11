"""Conditional background leakage and measured inference cost from fitted models."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F

from pini_cbm.data import normalize
from .data import write_json
from .infer import infer_rgb


def background_leakage(destination,arrays,splits,seed=340):
    """Foreground-held-out linear probe of nuisance background in concept residuals."""
    destination=Path(destination)
    conditions=['correlated','independent','reversed']
    scores=np.stack([np.load(destination/f'{c}_concepts.npy') for c in conditions],1)
    targets=np.stack([np.array(arrays[f'background_{c}'][splits['test']]) for c in conditions],1)
    # Each foreground's true concepts remain fixed while its background changes.
    residual=scores-scores.mean(1,keepdims=True)
    ids=np.arange(len(scores))
    train,test=train_test_split(ids,test_size=.5,random_state=seed,stratify=arrays['labels'][splits['test']])
    assert not set(train)&set(test)
    x=torch.as_tensor(residual[train].reshape(-1,scores.shape[-1]),device='cuda').float()
    y=torch.as_tensor(targets[train].reshape(-1),device='cuda').long()
    center=x.mean(0);scale=x.std(0).clamp_min(.001);x=(x-center)/scale
    v=torch.as_tensor(residual[test].reshape(-1,scores.shape[-1]),device='cuda').float()
    v=(v-center)/scale
    torch.manual_seed(seed)
    domain_count=int(targets.max())+1
    probe=nn.Linear(x.shape[1],domain_count).cuda()
    optimizer=torch.optim.AdamW(probe.parameters(),lr=.02,weight_decay=.01)
    for _ in range(400):
        loss=F.cross_entropy(probe(x),y)
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    predicted=probe(v).argmax(1).cpu().numpy()
    accuracy=balanced_accuracy_score(targets[test].reshape(-1),predicted)
    record={'background_probe_balanced_accuracy':float(accuracy),'chance':1/domain_count,
            'train_foregrounds':len(train),'test_foregrounds':len(test),'backgrounds_per_foreground':3,
            'concepts':scores.shape[-1],'split_seed':seed,'foreground_groups_disjoint':True,
            'feature':'concept scores minus the same foreground mean over three changed backgrounds',
            'meaning':'how easily a held-out foreground background domain can be read from concept scores'}
    np.savez(destination/'background_probe.npz',weight=probe.weight.detach().cpu().numpy(),
             bias=probe.bias.detach().cpu().numpy(),test_ids=test,train_ids=train,
             targets=targets[test].reshape(-1),predicted=predicted,center=center.cpu().numpy(),scale=scale.cpu().numpy())
    write_json(destination/'background_leakage.json',record)
    return record


@torch.no_grad()
def benchmark(model,method,backbone,rgb,destination):
    """Measure the full RGB path, including every encoder call, with CUDA events."""
    calls=[]
    hook=backbone.visual.register_forward_hook(lambda *unused:calls.append(1))
    def run():
        if method in ['mean','noise']:
            return infer_rgb(model,backbone,rgb,method)
        z=backbone.encode_image(normalize(rgb),spatial=True)
        if method=='linear_probe':return model(F.normalize(z[:,0].float(),dim=-1))
        c,_=model.recognize(z)
        return model.classify(c)
    model.eval();backbone.eval()
    with torch.autocast('cuda',dtype=torch.float16):
        run();run()
        calls.clear()
        first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
        first.record()
        for _ in range(20):run()
        last.record();torch.cuda.synchronize()
    hook.remove()
    expected=2 if method in ['mean','noise'] else 1
    assert len(calls)==20*expected,(method,len(calls))
    seconds=first.elapsed_time(last)/1000
    record={'batch':len(rgb),'repetitions':20,'images_per_second':20*len(rgb)/seconds,
            'milliseconds_per_image':1000*seconds/(20*len(rgb)),'encoder_calls_per_image':expected,
            'backbone':'CLIP ViT-B/16','cached_original_features':False,
            'parameters':sum(p.numel() for p in model.parameters()),
            'visual_backbone_parameters':sum(p.numel() for p in backbone.visual.parameters())}
    write_json(Path(destination)/'efficiency.json',record)
    return record
