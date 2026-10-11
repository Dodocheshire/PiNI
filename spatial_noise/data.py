"""Small native-concept subsets and paired class-correlated background shifts."""
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from clip import clip
from pini_cbm.data import ATTRS, CLASSES, mask_from_uri, normalize

ROOT = Path('artifacts/simplified_spatial_noise')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def per_class(ids, labels, count, rng):
    return np.sort(np.concatenate([rng.permutation(ids[labels[ids] == k])[:count]
                                   for k in np.unique(labels)]))


def background_crop(rgb, foreground):
    """Largest grid-aligned rectangle containing no foreground pixels."""
    occupied = foreground.reshape(16, 14, 16, 14).any((1, 3))
    best = (0, 0, 0, 0, 0)
    for top in range(16):
        free = np.ones(16, dtype=bool)
        for bottom in range(top, 16):
            free &= ~occupied[bottom]
            ends = np.flatnonzero(np.diff(np.r_[False, free, False].astype(int)))
            for left, right in zip(ends[::2], ends[1::2]):
                area = (bottom + 1 - top) * (right - left)
                if area > best[0]:
                    best = (area, top, bottom + 1, left, right)
    area, top, bottom, left, right = best
    if area < 12:
        return None
    assert not foreground[top*14:bottom*14, left*14:right*14].any()
    crop = rgb[:, top*14:bottom*14, left*14:right*14].transpose(1, 2, 0)
    return np.asarray(Image.fromarray(crop).resize((224, 224), Image.Resampling.BICUBIC)).transpose(2, 0, 1).copy()


def places_banks(destination,rng):
    """Copy a small, disjoint Places subset for the Waterbirds background recipe."""
    places=Path('/home/pengcheng/VLG-CBM/datasets/places365_torch/data_256')
    categories=[['b/bamboo_forest','f/forest/broadleaf'],['o/ocean','l/lake/natural']]
    banks={'train':[[],[]],'val':[[],[]]};sources={'train':[[],[]],'val':[[],[]]}
    for domain,groups in enumerate(categories):
        for group in groups:
            files=sorted((places/group).glob('*.jpg'))
            assert len(files)>=80,group
            choices=rng.permutation(len(files))[:80]
            for offset,j in enumerate(choices):
                split='train' if offset<60 else 'val'
                path=files[int(j)]
                with Image.open(path) as im:
                    resized=im.convert('RGB').resize((224,224),Image.Resampling.BICUBIC)
                    out=destination/'background_images'/split/str(domain)/(group.replace('/','_')+'_'+path.stem+'.png')
                    out.parent.mkdir(parents=True,exist_ok=True);resized.save(out)
                    banks[split][domain].append(np.asarray(resized).transpose(2,0,1).copy())
                sources[split][domain].append(str(path))
    for split in banks:
        banks[split]=[np.stack(v) for v in banks[split]]
    assert not {p for rows in sources['train'] for p in rows}&{p for rows in sources['val'] for p in rows}
    write_json(destination/'background_sources.json',{'source':'Places365 local copy','categories':categories,
               'files':sources,'recipe':'https://github.com/kohpangwei/group_DRO/blob/master/dataset_scripts/generate_waterbirds.py'})
    return banks


def prepare(name, backbone, batch=64):
    destination = ROOT/name/'cache'
    if (destination/'COMPLETE.json').exists():
        return destination
    started = time.time()
    source = Path('artifacts/rival_vitl14' if name == 'rival10' else 'artifacts/cross_dataset/cub/cache')
    records = json.loads((source/'manifest.json').read_text())
    labels = np.asarray([r['label'] for r in records])
    if name=='waterbirds':
        water=['Albatross','Auklet','Cormorant','Frigatebird','Fulmar','Gull','Jaeger','Kittiwake',
               'Pelican','Puffin','Tern','Gadwall','Grebe','Mallard','Merganser','Guillemot','Pacific_Loon']
        labels=np.asarray([int(any(term.lower() in r['relative_path'].split('/')[0].lower() for term in water))
                           for r in records])
    rng = np.random.default_rng(310)
    foreground_valid=np.load(source/'mask_valid.npy',mmap_mode='r')[:,-1]
    split_source = {s:np.flatnonzero(np.asarray([r['split'] == s for r in records]) & foreground_valid)
                    for s in ['train','val','test']}
    count = ({'train':200,'val':100,'test':300} if name=='waterbirds' else
             {'train':200 if name=='rival10' else 10, 'val':50 if name=='rival10' else 3,
              'test':100 if name=='rival10' else 5})
    selected = {s:per_class(ids,labels,count[s],rng) for s,ids in split_source.items()}
    original_ids = np.concatenate(list(selected.values()))
    splits = {}
    offset = 0
    for s, ids in selected.items():
        splits[s] = np.arange(offset, offset + len(ids))
        offset += len(ids)
    destination.mkdir(parents=True, exist_ok=True)
    rgb = np.array(np.load(source/'rgb.npy',mmap_mode='r')[original_ids])
    attrs = np.array(np.load(source/'attrs.npy',mmap_mode='r')[original_ids])
    masks16 = np.array(np.load(source/'masks.npy',mmap_mode='r')[original_ids],dtype='float32')
    masks = F.interpolate(torch.from_numpy(masks16),(14,14),mode='area').numpy().astype('float16')
    valid = np.array(np.load(source/'mask_valid.npy',mmap_mode='r')[original_ids])
    if name=='rival10':
        names = ATTRS
        classes = CLASSES
        fg = np.empty((len(original_ids),224,224),dtype=bool)
        for j,i in enumerate(original_ids):
            p=Path(records[int(i)]['path'])
            with p.with_name(p.stem+'_attr_dict.pkl').open('rb') as f:
                annotations=pickle.load(f)
            fg[j]=np.asarray(mask_from_uri(annotations['entire-object']).resize((224,224),Image.Resampling.NEAREST))>0
        prompts=['a photo showing '+s.replace('-',' ')+'.' for s in names]
    else:
        names=json.loads((source/'concepts.json').read_text())
        classes=json.loads((source/'protocol.json').read_text())['classes']
        if name=='waterbirds':classes=['landbird','waterbird']
        fg=np.array(np.load(source/'fg224.npy',mmap_mode='r')[original_ids])
        prompts=[]
        for s in names:
            group,value=s.split('::')
            prompts.append('a photo of a bird with '+value.replace('_',' ')+' '+group.replace('has_','').replace('_',' ')+'.')
    labels=labels[original_ids]
    np.save(destination/'rgb_original.npy',rgb)
    np.save(destination/'attrs.npy',attrs)
    np.save(destination/'masks.npy',masks)
    np.save(destination/'mask_valid.npy',valid)
    np.save(destination/'foreground.npy',fg)
    np.save(destination/'labels.npy',labels)
    with torch.no_grad():
        text=F.normalize(backbone.encode_text(clip.tokenize(prompts).cuda()).float(),dim=-1)
    np.save(destination/'text.npy',text.cpu().numpy())
    write_json(destination/'concepts.json',names)
    # Banks are made only from the selected training/validation images.
    banks = places_banks(destination,rng) if name=='waterbirds' else {}
    domain_count=2 if name=='waterbirds' else 10
    for split in ['train','val']:
        if name=='waterbirds':continue
        domains=[]
        for domain in range(10):
            crops=[]
            ids=rng.permutation(splits[split][labels[splits[split]]%10==domain])
            for i in ids:
                crop=background_crop(rgb[i],fg[i])
                if crop is not None:
                    crops.append(crop)
                if len(crops)==12:
                    break
            assert crops,(name,split,domain)
            domains.append(np.stack(crops))
        banks[split]=domains
        write_json(destination/f'{split}_background_bank.json',{'counts':[len(x) for x in domains],
            'crop_rule':'largest object-free rectangle; at least 12/256 of donor image',
            'source_split':split})
    protocol={'dataset':name,'backbone':'OpenAI CLIP ViT-B/16','classes':classes,'concepts':len(names),
              'split_seed':310,'per_class_budget':count,'counts':{s:len(v) for s,v in splits.items()},
              'splits':{s:v.tolist() for s,v in splits.items()},'source_indices':original_ids.tolist(),
              'source_manifest_sha256':hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
              'background_domains':domain_count,'background_match_draw_probability':.95,
              'expected_training_background_class_correlation':.95,
              'test_domains':['original','correlated','independent','reversed'],
              'native_concept_labels':True,'eligibility':'native entire-object mask available',
              'train_view_encoding':'one cached original encoding; one perturbed encoding',
              'inference_encodes_per_image':2,'native_foreground_preserved_exactly':True}
    write_json(destination/'protocol.json',protocol)
    for condition in protocol['test_domains']:
        images=np.lib.format.open_memmap(destination/f'rgb_{condition}.npy',mode='w+',dtype='uint8',shape=rgb.shape)
        domains=np.full(len(rgb),-1,dtype='int64')
        for split,ids in splits.items():
            for i in ids:
                if condition=='original':
                    images[i]=rgb[i]
                    continue
                domain=int(labels[i]%domain_count)
                if condition=='independent':
                    domain=int(rng.integers(domain_count))
                elif condition=='correlated' and rng.random()>.95:
                    domain=(domain+int(rng.integers(1,domain_count)))%domain_count
                elif condition=='reversed':
                    domain=(domain+1)%domain_count
                bank=banks['train' if split=='train' else 'val'][domain]
                background=bank[int(rng.integers(len(bank)))]
                images[i]=np.where(fg[i][None],rgb[i],background)
                domains[i]=domain
                assert np.array_equal(images[i,:,fg[i]],rgb[i,:,fg[i]])
        images.flush()
        np.save(destination/f'background_{condition}.npy',domains)
        if condition=='correlated':
            protocol['background_correlated_actual_match_fraction']={s:float((domains[ids]==labels[ids]%domain_count).mean())
                                                                     for s,ids in splits.items()}
            write_json(destination/'protocol.json',protocol)
        feature_path=destination/f'tokens_{condition}.npy'
        tokens=np.lib.format.open_memmap(feature_path,mode='w+',dtype='float16',shape=(len(rgb),197,512))
        for start in range(0,len(rgb),batch):
            x=torch.as_tensor(np.array(images[start:start+batch]),device='cuda').float()/255
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                z=backbone.encode_image(normalize(x),spatial=True)
            tokens[start:start+len(z)]=z.cpu().numpy()
            if start%(batch*20)==0 or start+batch>=len(rgb):
                print(json.dumps({'stage':'cache','dataset':name,'condition':condition,
                                 'done':min(start+batch,len(rgb)),'n':len(rgb),'elapsed_s':round(time.time()-started,1)}),flush=True)
        tokens.flush()
    write_json(destination/'COMPLETE.json',{'seconds':time.time()-started,'n':len(rgb),'feature_shape':[197,512]})
    return destination


def load(name):
    root=ROOT/name/'cache'
    assert (root/'COMPLETE.json').exists()
    protocol=json.loads((root/'protocol.json').read_text())
    arrays={key:np.load(root/(key+'.npy'),mmap_mode='r') for key in ['attrs','masks','mask_valid','foreground','labels','text']}
    for condition in protocol['test_domains']:
        for prefix in ['tokens','rgb','background']:
            arrays[prefix+'_'+condition]=np.load(root/f'{prefix}_{condition}.npy',mmap_mode='r')
    splits={key:np.asarray(ids,dtype='int64') for key,ids in protocol['splits'].items()}
    return root,arrays,splits,protocol


@torch.no_grad()
def white_background(name, backbone, arrays, protocol, batch=64):
    """Train and evaluate on native foreground pixels over an all-white background."""
    destination=ROOT/name/'white_background_cache'
    destination.mkdir(parents=True,exist_ok=True)
    if not (destination/'COMPLETE.json').exists():
        original=arrays['rgb_original'];n=len(original)
        foreground=np.array(arrays['foreground'])
        if name in ['cub','waterbirds']:
            records=json.loads(Path('artifacts/cross_dataset/cub/cache/manifest.json').read_text())
            for j,index in enumerate(protocol['source_indices']):
                path=Path('artifacts/datasets/local/segmentations')/Path(records[index]['relative_path']).with_suffix('.png')
                with Image.open(path) as im:
                    foreground[j]=np.asarray(im.convert('L').resize((224,224),Image.Resampling.NEAREST))>127
        np.save(destination/'foreground.npy',foreground)
        rgb=np.lib.format.open_memmap(destination/'rgb.npy',mode='w+',dtype='uint8',shape=original.shape)
        tokens=np.lib.format.open_memmap(destination/'tokens.npy',mode='w+',dtype='float16',shape=(n,197,512))
        for start in range(0,n,batch):
            end=min(start+batch,n)
            rgb[start:end]=np.where(foreground[start:end,None],original[start:end],255)
            x=torch.as_tensor(np.array(rgb[start:end]),device='cuda').float()/255
            with torch.autocast('cuda',dtype=torch.float16):
                z=backbone.encode_image(normalize(x),spatial=True)
            tokens[start:end]=z.cpu().numpy()
        rgb.flush();tokens.flush()
        write_json(destination/'COMPLETE.json',{'n':n,'background_rgb':[255,255,255],
                   'foreground':'native segmentation >127 for CUB; native binary mask for RIVAL10',
                   'splits':'same train/validation/test indices as source cache'})
    updated=dict(arrays)
    updated['foreground']=np.load(destination/'foreground.npy',mmap_mode='r')
    fg14=F.interpolate(torch.from_numpy(np.array(updated['foreground'])).float()[:,None],(14,14),mode='area').numpy()
    updated['masks']=np.array(arrays['masks'])
    updated['masks'][:,:-1]*=fg14
    updated['masks'][:,-1]=fg14[:,0]
    rgb=np.load(destination/'rgb.npy',mmap_mode='r');tokens=np.load(destination/'tokens.npy',mmap_mode='r')
    for condition in protocol['test_domains']:
        updated['rgb_'+condition]=rgb;updated['tokens_'+condition]=tokens
        updated['background_'+condition]=np.full(len(rgb),-1,dtype='int64')
    return updated


@torch.no_grad()
def prepare_salf(name, backbone, batch=128):
    """SALF paper's 7x7 red-circle visual prompts, radius 32, cached once."""
    root, arrays, splits, protocol=load(name)
    destination=root/'salf_teacher.npy'
    if (root/'SALF_TEACHER_COMPLETE.json').exists():
        return
    ids=np.r_[splits['train'],splits['val']]
    teacher=np.lib.format.open_memmap(destination,mode='w+',dtype='float32',
                                    shape=(len(arrays['labels']),len(arrays['text']),49))
    yy,xx=torch.meshgrid(torch.arange(224,device='cuda'),torch.arange(224,device='cuda'),indexing='ij')
    rings=torch.stack([((yy-y).square()+(xx-x).square()).sqrt().sub(32).abs()<2
                       for y in range(28,197,28) for x in range(28,197,28)])
    text=torch.as_tensor(np.array(arrays['text']),device='cuda')
    started=time.time()
    for start in range(0,len(ids)*49,batch):
        flat=np.arange(start,min(start+batch,len(ids)*49))
        source_ids=ids[flat//49]; positions=flat%49
        x=torch.as_tensor(np.array(arrays['rgb_correlated'][source_ids]),device='cuda').float()/255
        mask=rings[torch.as_tensor(positions,device='cuda')]
        x[:,0][mask]=1.;x[:,1][mask]=0.;x[:,2][mask]=0.
        with torch.autocast('cuda',dtype=torch.float16):
            z=F.normalize(backbone.encode_image(normalize(x)).float(),dim=-1)
        scores=(z@text.T).cpu().numpy()
        teacher[source_ids,:,positions]=scores
        if start%(batch*80)==0 or start+batch>=len(ids)*49:
            print(json.dumps({'stage':'salf_teacher','dataset':name,'done':int(flat[-1])+1,
                             'total':len(ids)*49,'elapsed_s':round(time.time()-started,1)}),flush=True)
    teacher.flush()
    write_json(root/'SALF_TEACHER_COMPLETE.json',{'grid':[7,7],'radius':32,'ring_width':4,
               'ids':ids.tolist(),'forward_encodes':len(ids)*49,'seconds':time.time()-started,
               'source':'https://arxiv.org/html/2502.20134v1#S3.SS2'})
