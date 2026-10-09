"""Cache frozen ViT-L/14 spatial features and immutable RIVAL10 split."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from clip import clip
from .data import ATTRS, CLASSES, RivalRaw, manifest, normalize


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/home/pengcheng/VLG-CBM/datasets/RIVAL10')
    p.add_argument('--out', default='artifacts/rival_vitl14')
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    torch.set_num_threads(6)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / 'COMPLETE.json').exists():
        print('Cache already complete', flush=True); return
    records = manifest(a.root)
    encoded = json.dumps(records, ensure_ascii=False, indent=2)
    (out / 'manifest.json').write_text(encoded)
    n = len(records)
    specs = {'tokens': ((n, 257, 768), 'float16'), 'rgb': ((n, 3, 224, 224), 'uint8'),
             'attrs': ((n, 18), 'float32'), 'masks': ((n, 19, 16, 16), 'float16'), 'mask_valid': ((n, 19), 'bool')}
    arrays = {k: np.lib.format.open_memmap(out / (k + '.npy'), mode='w+', dtype=d, shape=s) for k, (s, d) in specs.items()}
    model, _ = clip.load('ViT-L/14', device='cuda', jit=False)
    model.eval().requires_grad_(False)
    with torch.no_grad():
        texts = model.encode_text(clip.tokenize(['a photo showing ' + x.replace('-', ' ') + '.' for x in ATTRS]).cuda())
        np.save(out / 'text.npy', torch.nn.functional.normalize(texts.float(), dim=-1).cpu().numpy())
    loader = DataLoader(RivalRaw(records), batch_size=a.batch, num_workers=a.workers, pin_memory=True, shuffle=False)
    t = time.time()
    with torch.inference_mode():
        for batch, (ids, rgb, attrs, masks, valid) in enumerate(loader):
            features = model.encode_image(normalize(rgb.cuda(non_blocking=True).float() / 255), spatial=True)
            inds = ids.numpy()
            arrays['tokens'][inds] = features.cpu().numpy()
            arrays['rgb'][inds] = rgb.numpy()
            arrays['attrs'][inds] = attrs.numpy()
            arrays['masks'][inds] = masks.numpy()
            arrays['mask_valid'][inds] = valid.numpy()
            if batch % 20 == 0:
                print(json.dumps({'stage': 'cache', 'done': int(ids[-1]) + 1, 'total': n, 'elapsed_s': round(time.time()-t, 1)}), flush=True)
    for arr in arrays.values(): arr.flush()
    info = dict(n=n, counts={s:sum(r['split']==s for r in records) for s in ['train','val','test']},
                attrs=ATTRS, classes=CLASSES, backbone='OpenAI CLIP ViT-L/14 224',
                transform='RGB bicubic resize 224x224; same geometry for masks; CLIP normalize; no augmentation',
                manifest_sha256=hashlib.sha256(encoded.encode()).hexdigest(), seed=42)
    (out / 'COMPLETE.json').write_text(json.dumps(info, indent=2))
    print(info, flush=True)


if __name__ == '__main__': main()
