"""16-shot OxfordPets reproduction using exact official PiNI model classes."""
import argparse
import hashlib
import json
import random
import time
import types
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from clip import clip
from clip.model import convert_weights
from .data import MEAN, STD
from .official import load_official


def mean_image_sample(self,mu,variance,num=1):
    return mu.expand(num,*mu.size()).transpose(0,1)


def mean_prompt_sample(self,mu,variance,num=1):
    return mu.expand(num,*mu.size())


def set_sampling_mode(model,kind,mode):
    """Only the declared mean control changes sampling; original code intact."""
    if kind=='image':generators=[(model.vpnGenerator,mean_image_sample)]
    else:generators=[(model.imageVpnGenerator,mean_image_sample),(model.promptVpnGenerator,mean_prompt_sample)]
    for generator,replacement in generators:
        if not hasattr(generator,'_original_sample'):generator._original_sample=generator.sample
        generator.sample=types.MethodType(replacement,generator) if mode=='mean' else generator._original_sample


def split_pets(root, seed):
    root=Path(root); random.seed(seed)
    def read(name):
        rows=[]
        for line in (root/'annotations'/(name+'.txt')).read_text().splitlines():
            if line.startswith('#') or not line.strip(): continue
            stem,label,*_=line.split()
            rows.append(dict(path=str(root/'images'/(stem+'.jpg')),label=int(label)-1,
                             classname=stem.rsplit('_',1)[0].lower()))
        return rows
    trainval=read('trainval');test=read('test');by=defaultdict(list)
    for row in trainval:by[row['label']].append(row)
    train=[];val=[]
    for rows in by.values():
        # Exact official operation: shuffle indices before partition.
        indices=list(range(len(rows)));random.shuffle(indices);n=round(len(rows)*.2)
        val.extend(rows[i] for i in indices[:n]);train.extend(rows[i] for i in indices[n:])
    def fewshot(rows,n):
        by=defaultdict(list)
        for row in rows:by[row['label']].append(row)
        return [r for rr in by.values() for r in random.sample(rr,n)]
    return dict(train=fewshot(train,16),val=fewshot(val,4),test=test)


class PetsImages(Dataset):
    def __init__(self, rows, train=False):
        self.rows=rows
        geom=[T.RandomResizedCrop(224,interpolation=T.InterpolationMode.BICUBIC),T.RandomHorizontalFlip()] if train else [T.Resize(224,interpolation=T.InterpolationMode.BICUBIC),T.CenterCrop(224)]
        self.transform=T.Compose(geom+[T.ToTensor(),T.Normalize(MEAN,STD)])
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        return self.transform(Image.open(row['path']).convert('RGB')),row['label']


@torch.no_grad()
def evaluate(model, loader, kind, seed=1301, draws=1):
    torch.manual_seed(seed);model.eval();logits=[];labels=[]
    for x,y in loader:
        x=x.cuda(non_blocking=True)
        with torch.autocast('cuda',dtype=torch.float16):
            samples=[]
            for _ in range(draws):
                # Official image forward labels are only used by an unused
                # local variable. Inference receives zero placeholders.
                z=model(x,torch.zeros(len(x),device='cuda',dtype=torch.long)) if kind=='image' else model(x)
                samples.append(z.float().softmax(-1))
            z=torch.stack(samples).mean(0)
        logits.append(z.cpu().numpy());labels.append(y.numpy())
    probs=np.concatenate(logits);labels=np.concatenate(labels)
    return dict(accuracy=float((probs.argmax(1)==labels).mean()),NLL=float(-np.log(probs[np.arange(len(labels)),labels].clip(1e-12)).mean()),n=len(labels)),probs


@torch.no_grad()
def zero_shot(backbone, loader, names):
    prompts=['a photo of a '+n.replace('_',' ')+', a type of pet.' for n in names]
    text=F.normalize(backbone.encode_text(clip.tokenize(prompts).cuda()).float(),dim=-1)
    out=[];labels=[]
    for x,y in loader:
        features=F.normalize(backbone.encode_image(x.cuda()).float(),dim=-1)
        out.append((backbone.logit_scale.exp()*features@text.T).softmax(-1).cpu().numpy());labels.append(y.numpy())
    probs=np.concatenate(out);labels=np.concatenate(labels)
    return dict(accuracy=float((probs.argmax(1)==labels).mean()),n=len(labels)),probs


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='artifacts/data/oxford_pets/oxford-iiit-pet')
    p.add_argument('--kind',choices=['main','image'],default='main');p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--batch',type=int,default=32);p.add_argument('--lr',type=float,default=.002)
    p.add_argument('--seed',type=int,default=1);p.add_argument('--out')
    p.add_argument('--noise-mode',choices=['noise','mean'],default='noise')
    p.add_argument('--precision',choices=['amp','fp16'],default='amp')
    p.add_argument('--eval-batch',type=int,default=128);a=p.parse_args()
    torch.set_num_threads(6);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    suffix=('_mean' if a.noise_mode=='mean' else '')+('_fp16' if a.precision=='fp16' else '')
    out=Path(a.out or 'artifacts/pets_'+a.kind+suffix+'_seed'+str(a.seed));out.mkdir(parents=True,exist_ok=True)
    splits=split_pets(a.root,a.seed); encoded=json.dumps(splits,indent=2);(out/'split.json').write_text(encoded)
    names={r['label']:r['classname'] for r in splits['test']};names=[names[i] for i in range(37)]
    loaders={s:DataLoader(PetsImages(rows,s=='train'),batch_size=a.batch if s=='train' else a.eval_batch,
                         shuffle=s=='train',num_workers=4,pin_memory=True,drop_last=s=='train') for s,rows in splits.items()}
    classes,provenance=load_official('image' if a.kind=='image' else 'main')
    # Official construction initializes prompt embeddings on CPU, then moves
    # the complete model to CUDA. Keep this order for CPU token indices.
    backbone,_=clip.load('ViT-B/16',device='cpu',jit=False);backbone.float().eval().requires_grad_(False)
    if a.precision=='fp16':convert_weights(backbone)
    cfg=NS(DATASET=NS(NAME='OxfordPets'),TRAINER=NS(CLIP_VPN=NS(REDUCTION=1)))
    model=classes.CustomCLIP(cfg,names,backbone).cuda()
    set_sampling_mode(model,a.kind,a.noise_mode)
    params=[]
    for name,param in model.named_parameters():
        active=('vpnGenerator' in name) if a.kind=='image' else ('imageVpnGenerator' in name or 'promptVpnGenerator' in name)
        param.requires_grad_(active)
        if active:params.append(param)
    optim=torch.optim.SGD(params,lr=a.lr,momentum=.9,weight_decay=.0005)
    scaler=torch.amp.GradScaler('cuda',enabled=a.precision=='amp')
    history=[];best=-1;t=time.time()
    def checkpoint(epoch,path):
        state={k:v for k,v in model.state_dict().items() if 'vpnGenerator' in k or 'imageVpnGenerator' in k or 'promptVpnGenerator' in k}
        torch.save(dict(state=state,epoch=epoch,args=vars(a),provenance=provenance,classnames=names),path)
    for epoch in range(a.epochs):
        model.train();backbone.eval();losses=[]
        lr=1e-5 if epoch==0 else a.lr*.5*(1+np.cos(np.pi*(epoch-1)/(a.epochs-1)))
        for group in optim.param_groups:group['lr']=lr
        for x,y in loaders['train']:
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);optim.zero_grad()
            with torch.autocast('cuda',dtype=torch.float16,enabled=a.precision=='amp'):
                logits=model(x,y) if a.kind=='image' else model(x)
                loss=F.cross_entropy(logits.float() if a.precision=='amp' else logits,y)
            assert torch.isfinite(loss).item(), f'Nonfinite loss at epoch {epoch+1}; precision={a.precision}'
            scaler.scale(loss).backward();scaler.step(optim);scaler.update();losses.append(float(loss.detach()))
        row=dict(epoch=epoch+1,train_CE=float(np.mean(losses)),lr=lr,elapsed_s=round(time.time()-t,1))
        if (epoch+1)%10==0 or epoch==0 or epoch+1==a.epochs:
            val,_=evaluate(model,loaders['val'],a.kind);row['val']=val
            if val['accuracy']>best:best=val['accuracy'];checkpoint(epoch+1,out/'best_val.pt')
        checkpoint(epoch+1,out/'last.pt');history.append(row)
        (out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
    # Published last-step convention: no choice based on test scores.
    test,preds=evaluate(model,loaders['test'],a.kind)
    np.save(out/'test_MC1_probs.npy',preds)
    if a.noise_mode=='mean':
        # Repeating a fully deterministic sampler yields the same probabilities.
        mc8=dict(test);mc8_preds=preds.copy()
    else:mc8,mc8_preds=evaluate(model,loaders['test'],a.kind,draws=8)
    np.save(out/'test_MC8_probs.npy',mc8_preds)
    # float backbone requires autocast because it was made float for AMP.
    with torch.autocast('cuda',dtype=torch.float16):clean,probs=zero_shot(backbone,loaders['test'],names)
    np.save(out/'test_zero_shot_probs.npy',probs)
    results=dict(kind=a.kind,last_epoch=a.epochs,MC1=test,MC8=mc8,zero_shot=clean,
                 official_paper_reference=(.8951 if a.kind=='image' else .9297) if a.noise_mode=='noise' else None,
                 noise_mode=a.noise_mode,
                 seed=a.seed,shots=16,provenance=provenance,manifest_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
                 precision='fp32 master weights + torch AMP fp16; only original generators optimized' if a.precision=='amp' else 'original fp16 model/generators, direct fp16 CE/SGD, no scaler',
                 evaluation_batch=a.eval_batch,
                 lr_configuration='Pets exact LR unpublished; fixed 0.002, no test tuning',
                 split_configuration='official split algorithm seeded 1; original published split file unavailable',
                 changed_core_equations=a.noise_mode=='mean')
    if a.noise_mode=='noise':
        set_sampling_mode(model,a.kind,'mean')
        mean_stats,mean_preds=evaluate(model,loaders['test'],a.kind)
        np.save(out/'test_mean_probs.npy',mean_preds);results['same_checkpoint_mean']=mean_stats
        set_sampling_mode(model,a.kind,'noise')
    (out/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)


if __name__=='__main__':main()
