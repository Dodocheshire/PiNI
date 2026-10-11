"""Paired background data and cached frozen DINO inputs for aligned CUB experiments."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from .cub_aligned import ROOT, BirdBackbone, load, tensor
from .data import write_json
from .grounding import FrozenGrounder, BIRD_PHRASES


@torch.no_grad()
def backgrounds(batch=128, experiment=None):
    root = ROOT if experiment is None else experiment.root
    cache = root/'cache'
    if (cache/'BACKGROUND_COMPLETE.json').exists():
        return
    arrays, splits, protocol = load() if experiment is None else experiment.load()
    rng = np.random.default_rng(42)
    places = Path('/home/pengcheng/VLG-CBM/datasets/places365_torch/data_256')
    groups = ['b/bamboo_forest', 'f/forest/broadleaf', 'o/ocean', 'l/lake/natural']
    sources = {s: [] for s in splits}; banks = {s: [] for s in splits}
    for group in groups:
        paths = sorted((places/group).glob('*.jpg'))
        selected = rng.permutation(len(paths))[:180]
        assert len(selected) == 180
        for j, index in enumerate(selected):
            split = ['train', 'val', 'test'][j//60]
            path = paths[index]
            with Image.open(path) as image:
                pixels = np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)).transpose(2, 0, 1).copy()
            sources[split].append(str(path)); banks[split].append(pixels)
    assert all(not set(sources[a]) & set(sources[b]) for a,b in [('train','val'),('train','test'),('val','test')])
    n = len(arrays['labels'])
    rgb = np.lib.format.open_memmap(cache/'rgb_changed.npy', mode='w+', dtype='uint8', shape=(n,3,224,224))
    feature = np.lib.format.open_memmap(cache/'features_changed.npy', mode='w+', dtype='float16', shape=(n,512,7,7))
    assignment = np.empty(n, dtype='int64')
    backbone = (BirdBackbone() if experiment is None else experiment.backbone()).cuda()
    for split, ids in splits.items():
        bank = np.stack(banks[split])
        for offset in range(0, len(ids), batch):
            selection = ids[offset:offset+batch]
            choice = rng.integers(len(bank), size=len(selection)); assignment[selection] = choice
            mask = np.asarray(arrays['foreground'][selection])[:, None]
            original = np.asarray(arrays['rgb_original'][selection])
            changed = np.where(mask, original, bank[choice])
            eligible = mask.reshape(len(selection),-1).any(1)
            changed[~eligible] = original[~eligible]
            assert np.array_equal(changed*mask, original*mask)
            rgb[selection] = changed
            with torch.autocast('cuda', dtype=torch.float16):
                feature[selection] = backbone(tensor(changed)/255).cpu().numpy()
        print(json.dumps({'stage': 'backgrounds', 'split': split, 'images': len(ids)}), flush=True)
    rgb.flush(); feature.flush(); np.save(cache/'background_index.npy', assignment)
    write_json(cache/'background_protocol.json', {'seed':42, 'sources': sources,
        'recipe': 'Native foreground >127 retained exactly; uniform independent Places backgrounds, disjoint banks per split',
        'sources_per_split': {s:len(v) for s,v in sources.items()},
        'training': 'original and background-changed inputs mixed equally in every adaptation arm'})
    write_json(cache/'BACKGROUND_COMPLETE.json', {'images': n})


@torch.no_grad()
def regions(batch=4, experiment=None):
    root = ROOT if experiment is None else experiment.root
    phrases = BIRD_PHRASES if experiment is None else experiment.phrases
    cache = root/'cache'; n = len(np.load(cache/'labels.npy'))
    precision='fp32' if experiment is None else experiment.detector_precision
    grounder = FrozenGrounder(phrases=phrases,image_size=800,precision=precision)
    for view in ['original', 'changed']:
        destination = root/'regions'/view; destination.mkdir(parents=True, exist_ok=True)
        if (destination/'COMPLETE.json').exists():
            continue
        rgb = np.load(cache/f'rgb_{view}.npy', mmap_mode='r')
        progress = json.loads((destination/'progress.json').read_text()) if (destination/'progress.json').exists() else {'done':0, 'detector_seconds':0}
        count = len(phrases)
        specs = {'features': ((n,256,14,14),'float16'), 'maps': ((n,count,14,14),'float16'),
                 'boxes': ((n,count,4),'float32'), 'scores': ((n,count),'float32'), 'valid': ((n,count),'bool')}
        arrays = {k:np.lib.format.open_memmap(destination/(k+'.npy'), mode='r+' if (destination/(k+'.npy')).exists() else 'w+',
                   dtype=dtype, shape=shape) for k,(shape,dtype) in specs.items()}
        started = time.time(); start = progress['done']
        for offset in range(start,n,batch):
            end = min(offset+batch,n)
            images = [Image.fromarray(np.asarray(row).transpose(1,2,0)) for row in rgb[offset:end]]
            result = grounder(images)
            for key,value in arrays.items():
                value[offset:end] = result[key].cpu().numpy()
            progress['detector_seconds'] += result['detector_forward_s']; progress['done'] = end
            if (end-start) % (batch*25) == 0 or end == n:
                for value in arrays.values(): value.flush()
                write_json(destination/'progress.json', progress)
                print(json.dumps({'stage':'regions', 'view':view, 'done':end, 'total':n,
                    'seconds':round(time.time()-started,1)}), flush=True)
        write_json(destination/'COMPLETE.json', {**progress, 'image_size':800, 'phrases':phrases,
            'caption':grounder.caption, 'threshold':.25, 'precision':precision,'batch':batch,
            'one_detector_encode_per_image':True})
    grounder.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['backgrounds','regions','all'], default='all')
    parser.add_argument('--batch',type=int,default=4)
    args=parser.parse_args(); torch.set_num_threads(6)
    if args.phase in ['backgrounds','all']: backgrounds()
    if args.phase in ['regions','all']: regions(args.batch)


if __name__ == '__main__': main()
