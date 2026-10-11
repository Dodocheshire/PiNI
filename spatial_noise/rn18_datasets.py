"""Full RIVAL10 and CUB/Places Waterbirds inputs for the shared ResNet18 CBM."""
import argparse
import copy
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision.models import resnet18,ResNet18_Weights

from .cub_aligned import BirdBackbone,SimpleCBM,normalize,tensor,fit_classifier
from .data import write_json
from .grounding import BIRD_PHRASES,FrozenGrounder
from pini_cbm.data import ATTRS,CLASSES,mask_from_uri

RIVAL_PHRASES=['animal','vehicle','head','wing','wheel','tail','ear','beak','eye','horn','leg','body']
CUB=Path('artifacts/cub_aligned_rn18')
RIVAL_SOURCE=Path('artifacts/rival_vitl14')


def link(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():os.symlink(source.resolve(),destination)


class RivalBackbone(nn.Module):
    def __init__(self,path):
        super().__init__();network=resnet18(weights=None)
        network.fc=nn.Linear(512,10)
        network.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'])
        self.spatial=nn.Sequential(*list(network.children())[:-2])
        self.eval().requires_grad_(False)

    def forward(self,rgb):return self.spatial(normalize(rgb))


class Experiment:
    def __init__(self,name):
        self.name=name;self.root=Path('artifacts')/(name+'_aligned_rn18')
        self.k=18 if name=='rival10' else 112
        self.classes=10 if name=='rival10' else 2
        self.phrases=RIVAL_PHRASES if name=='rival10' else BIRD_PHRASES
        self.detector_precision='fp16' if name=='rival10' else 'fp32'

    def model(self):return SimpleCBM(self.k,self.classes)
    def backbone(self):return RivalBackbone(self.root/'backbone/refit/best.pt') if self.name=='rival10' else BirdBackbone()
    def load(self):
        cache=self.root/'cache';protocol=json.loads((cache/'protocol.json').read_text())
        splits={k:np.asarray(v,dtype='int64') for k,v in protocol['splits'].items()}
        arrays={k:np.load(cache/(k+'.npy'),mmap_mode='r') for k in ['labels','attrs','rgb_original','foreground','features_original']}
        if (cache/'background_eligible.npy').exists():
            arrays['background_eligible']=np.load(cache/'background_eligible.npy',mmap_mode='r')
        return arrays,splits,protocol


class RivalMasks(Dataset):
    def __init__(self,records):self.records=records
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        path=Path(self.records[index]['path'])
        with path.with_name(path.stem+'_attr_dict.pkl').open('rb') as f:annotation=pickle.load(f)
        if 'entire-object' in annotation:
            foreground=np.asarray(mask_from_uri(annotation['entire-object']).resize((224,224),Image.Resampling.NEAREST))>0
        else:
            with Image.open(path.with_name(path.stem+'_merged_mask.JPEG')) as im:
                foreground=np.asarray(im.convert('L').resize((224,224),Image.Resampling.NEAREST))>127
        return index,torch.from_numpy(foreground.copy())


def prepare_waterbirds(experiment):
    root=experiment.root;cache=root/'cache';cache.mkdir(parents=True,exist_ok=True)
    if (cache/'COMPLETE.json').exists():return
    source=CUB/'cache';protocol=json.loads((source/'protocol.json').read_text())
    species=json.loads(Path('artifacts/cross_dataset/cub/cache/protocol.json').read_text())['classes']
    water=['Albatross','Auklet','Cormorant','Frigatebird','Fulmar','Gull','Jaeger','Kittiwake',
           'Pelican','Puffin','Tern','Gadwall','Grebe','Mallard','Merganser','Guillemot','Pacific_Loon']
    mapping=np.asarray([int(any(term.lower() in name.lower() for term in water)) for name in species])
    labels=mapping[np.load(source/'labels.npy')]
    np.save(cache/'labels.npy',labels)
    for filename in ['attrs.npy','native_ids.npy','concepts.json','foreground.npy','rgb_original.npy','rgb_changed.npy',
                     'features_original.npy','features_changed.npy','background_protocol.json','BACKGROUND_COMPLETE.json']:
        link(source/filename,cache/filename)
    for view in ['original','changed']:link(CUB/'regions'/view,root/'regions'/view)
    protocol.update(dataset='Waterbirds: complete CUB birds grouped as landbird/waterbird + independent Places background swaps',
                    classes=['landbird','waterbird'],concept_count=112,
                    species_to_binary=mapping.tolist(),waterbird_terms=water,
                    paired_cache_source=str(CUB),region_phrases=experiment.phrases,
                    split_label_counts={s:np.bincount(labels[ids],minlength=2).tolist() for s,ids in protocol['splits'].items()})
    write_json(cache/'protocol.json',protocol);write_json(cache/'COMPLETE.json',{'images':len(labels)})
    print(json.dumps({'stage':'waterbirds_inputs','counts':protocol['counts'],'label_counts':protocol['split_label_counts']}),flush=True)


def prepare_rival(experiment):
    cache=experiment.root/'cache';cache.mkdir(parents=True,exist_ok=True)
    if (cache/'RAW_COMPLETE.json').exists():return
    records=json.loads((RIVAL_SOURCE/'manifest.json').read_text());n=len(records)
    labels=np.asarray([r['label'] for r in records],dtype='int64')
    splits={s:np.flatnonzero(np.asarray([r['split']==s for r in records])).tolist() for s in ['train','val','test']}
    for f in ['rgb.npy','attrs.npy']:link(RIVAL_SOURCE/f,cache/('rgb_original.npy' if f=='rgb.npy' else f))
    np.save(cache/'labels.npy',labels);write_json(cache/'concepts.json',ATTRS)
    foreground=np.lib.format.open_memmap(cache/'foreground.npy',mode='w+',dtype='bool',shape=(n,224,224))
    loader=DataLoader(RivalMasks(records),batch_size=128,num_workers=8)
    for ids,fg in loader:foreground[ids.numpy()]=fg.numpy()
    foreground.flush()
    np.save(cache/'background_eligible.npy',foreground.reshape(n,-1).any(1))
    protocol={'dataset':'RIVAL10','classes':CLASSES,'concept_count':18,'concepts':ATTRS,
              'counts':{k:len(v) for k,v in splits.items()},'splits':splits,
              'backbone':'ImageNet ResNet18, task pretrained on RIVAL10 train+val, then frozen',
              'preprocess':'RGB bicubic resize to 224x224; ImageNet mean/std',
              'source_manifest_sha256':hashlib.sha256((RIVAL_SOURCE/'manifest.json').read_bytes()).hexdigest(),
              'foreground':'Native entire-object annotation, nearest resize to 224',
              'region_phrases':experiment.phrases,'missing_foreground_masks':int((~foreground.reshape(n,-1).any(1)).sum())}
    write_json(cache/'protocol.json',protocol);write_json(cache/'RAW_COMPLETE.json',{'images':n})
    print(json.dumps({'stage':'rival_inputs','counts':protocol['counts'],'missing_masks':protocol['missing_foreground_masks']}),flush=True)


def pretrain_rival(experiment,epochs=15,batch=128,seed=42):
    cache=experiment.root/'cache';protocol=json.loads((cache/'protocol.json').read_text())
    if (experiment.root/'backbone/refit/COMPLETE.json').exists():return
    images=torch.as_tensor(np.array(np.load(cache/'rgb_original.npy',mmap_mode='r')),device='cuda')
    labels=torch.as_tensor(np.load(cache/'labels.npy'),device='cuda')
    splits={k:torch.tensor(v,device='cuda') for k,v in protocol['splits'].items()}
    for stage in ['selection','refit']:
        destination=experiment.root/'backbone'/stage;destination.mkdir(parents=True,exist_ok=True)
        if (destination/'COMPLETE.json').exists():continue
        torch.manual_seed(seed);model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).cuda()
        model.fc=nn.Linear(512,10).cuda()
        ids=splits['train'] if stage=='selection' else torch.cat([splits['train'],splits['val']])
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
        scaler=torch.amp.GradScaler('cuda',init_scale=256)
        history=[];best=-float('inf');started=time.time()
        for epoch in range(1,epochs+1):
            model.train();order=ids[torch.randperm(len(ids),device='cuda')]
            total=0;correct=0;loss_sum=0
            for offset in range(0,len(order),batch):
                selected=order[offset:offset+batch];x=images[selected].float()/255
                flips=torch.rand(len(selected),device='cuda')<.5
                x[flips]=x[flips].flip(-1)
                with torch.autocast('cuda',dtype=torch.float16):
                    y=model(normalize(x));loss=F.cross_entropy(y,labels[selected])
                optimizer.zero_grad(set_to_none=True);scaler.scale(loss).backward()
                scaler.step(optimizer);scaler.update()
                correct+=int((y.argmax(1)==labels[selected]).sum());loss_sum+=float(loss.detach())*len(selected);total+=len(selected)
            model.eval();validation=0
            if stage=='selection':
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                    for offset in range(0,len(splits['val']),batch):
                        selected=splits['val'][offset:offset+batch]
                        validation+=int((model(normalize(images[selected].float()/255)).argmax(1)==labels[selected]).sum())
            score=validation/len(splits['val']) if stage=='selection' else -loss_sum/total
            if score>best:
                best=score;selected_epoch=epoch
                torch.save({'model':model.state_dict(),'seed':seed,'stage':stage,'epoch':epoch,
                            'fit_images':len(ids),'initialization':'ImageNet resnet18-f37072fd'},destination/'best.pt')
            entry={'stage':'rival_backbone_'+stage,'epoch':epoch,'loss':loss_sum/total,
                   'train_accuracy':correct/total,'validation_accuracy':validation/len(splits['val']) if stage=='selection' else None,
                   'seconds':time.time()-started}
            history.append(entry);write_json(destination/'history.json',history);print(json.dumps(entry),flush=True)
            scheduler.step()
        write_json(destination/'COMPLETE.json',{'selected_epoch':selected_epoch,'epochs':epochs,'fit_images':len(ids),
                    'selection':'validation accuracy' if stage=='selection' else 'training loss','seconds':time.time()-started})
    del images,model;torch.cuda.empty_cache()


@torch.no_grad()
def features(experiment,batch=128):
    cache=experiment.root/'cache'
    if (cache/'COMPLETE.json').exists():return
    rgb=np.load(cache/'rgb_original.npy',mmap_mode='r');n=len(rgb);backbone=experiment.backbone().cuda()
    out=np.lib.format.open_memmap(cache/'features_original.npy',mode='w+',dtype='float16',shape=(n,512,7,7))
    for offset in range(0,n,batch):
        with torch.autocast('cuda',dtype=torch.float16):z=backbone(tensor(rgb[offset:offset+batch])/255)
        out[offset:offset+len(z)]=z.cpu().numpy()
    out.flush();write_json(cache/'COMPLETE.json',{'images':n})


def simple(experiment,seed=42,steps=1000):
    arrays,splits,protocol=experiment.load()
    # Mean features are pooled on the CPU to avoid holding large spatial banks in GPU memory.
    maps=arrays['features_original'];x=torch.empty((len(maps),512),device='cuda')
    for offset in range(0,len(maps),256):x[offset:offset+256]=tensor(maps[offset:offset+256]).mean((2,3))
    target=tensor(arrays['attrs']);labels=tensor(arrays['labels']).long()
    for stage in ['simple','simple_refit']:
        destination=experiment.root/stage/f'seed{seed}';destination.mkdir(parents=True,exist_ok=True)
        if (destination/'COMPLETE.json').exists():continue
        refit=stage=='simple_refit';ids=np.concatenate([splits['train'],splits['val']]) if refit else splits['train']
        assert not set(ids)&set(splits['test'])
        torch.manual_seed(seed);model=experiment.model().cuda()
        model.feature_center.copy_(x[ids].mean(0));model.feature_scale.copy_(x[ids].std(0).clamp_min(.03))
        priors=target[ids].mean(0).clamp(.02,.98);pos=((1-priors)/priors).clamp(1,10)
        optimizer=torch.optim.AdamW(model.concepts.parameters(),lr=.01,weight_decay=.001)
        best=-float('inf');history=[];started=time.time()
        for step in range(1,steps+1):
            c=model.concept_logits(x[ids]);loss=F.binary_cross_entropy_with_logits(c,target[ids],pos_weight=pos)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if step%25==0:
                with torch.no_grad():
                    use=ids if refit else splits['val'];v=model.concept_logits(x[use])
                    metric=float(((v>0)==target[use].bool()).float().mean()) if refit else -float(F.binary_cross_entropy_with_logits(v,target[use],pos_weight=pos))
                history.append({'step':step,'loss':float(loss.detach()),'selection_score':metric})
                if metric>best:best=metric;state=copy.deepcopy(model.concepts.state_dict());selected=step
        model.concepts.load_state_dict(state)
        with torch.no_grad():concepts=model.concept_logits(x)
        use=ids if refit else splits['val']
        info=fit_classifier(model,{'train':concepts[ids],'val':concepts[use]},
                             {'train':labels[ids],'val':labels[use]},steps)
        if refit:info['training_accuracy']=info.pop('validation_accuracy')
        result={'dataset':experiment.name,'seed':seed,'fit_images':len(ids),'concept_selected_step':selected,
                'classifier':info,'seconds':time.time()-started,'concepts':experiment.k,'classes':experiment.classes,
                'trainable_parameters':sum(p.numel() for p in model.parameters())}
        with torch.no_grad():
            for split,selection in splits.items():
                c=concepts[selection];y=model.classify(c)
                result[split]={'accuracy':float((y.argmax(1)==labels[selection]).float().mean()),
                               'concept_accuracy':float(((c>0)==target[selection].bool()).float().mean()),'images':len(selection)}
                np.save(destination/f'{split}_concepts.npy',c.cpu().numpy());np.save(destination/f'{split}_logits.npy',y.cpu().numpy())
        torch.save({'model':model.state_dict(),'seed':seed,'architecture':'SimpleCBM','concepts':experiment.k,
                    'classes':experiment.classes,'protocol':str(experiment.root/'cache/protocol.json')},destination/'best.pt')
        write_json(destination/'history.json',history);write_json(destination/'results.json',result)
        write_json(destination/'COMPLETE.json',result);print(json.dumps({'stage':stage,'results':result}),flush=True)


def backgrounds(experiment):
    from . import cub_regions
    cub_regions.backgrounds(experiment=experiment)


def regions(experiment,batch=16):
    from . import cub_regions
    cub_regions.regions(batch,experiment=experiment)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,choices=['rival10','waterbirds'])
    parser.add_argument('--phase',choices=['prepare','regions','train','all'],default='all')
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--backbone-epochs',type=int,default=15);args=parser.parse_args();torch.set_num_threads(6)
    e=Experiment(args.dataset)
    if args.phase in ['prepare','all']:
        if e.name=='waterbirds':prepare_waterbirds(e)
        else:prepare_rival(e);pretrain_rival(e,args.backbone_epochs,seed=args.seed);features(e)
        simple(e,args.seed);backgrounds(e)
    if args.phase in ['regions','all']:regions(e)
    if args.phase in ['train','all']:
        from .cub_noise import train,verify,simple_evaluation
        verify(args.seed,experiment=e)
        for method in ['clean','mean','noise']:
            train(method,args.seed,args.epochs,experiment=e)
            train(method,args.seed,args.epochs,refit=True,experiment=e)
        simple_evaluation(args.seed,experiment=e)


if __name__=='__main__':main()
