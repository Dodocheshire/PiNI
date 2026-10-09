"""RIVAL10 native labels/masks; annotation-free inference uses only images."""
import base64
import io
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

# Exact ordering from mmoayeri/RIVAL10/datasets/local_rival10.py.
ATTRS = ['long-snout', 'wings', 'wheels', 'text', 'horns', 'floppy-ears',
         'ears', 'colored-eyes', 'tail', 'mane', 'beak', 'hairy', 'metallic',
         'rectangular', 'wet', 'long', 'tall', 'patterned']
CLASSES = ['truck', 'car', 'plane', 'ship', 'cat', 'dog', 'equine', 'deer', 'frog', 'bird']
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def normalize(rgb):
    return (rgb - rgb.new_tensor(MEAN)[None, :, None, None]) / rgb.new_tensor(STD)[None, :, None, None]


def manifest(root, seed=42):
    root = Path(root)
    meta = root if (root / 'label_mappings.json').exists() else root / 'meta'
    mappings = json.loads((meta / 'label_mappings.json').read_text())
    wnids = json.loads((meta / 'wnid_to_class.json').read_text())
    records = []
    for split in ['train', 'test']:
        for p in sorted((root / split / 'ordinary').glob('*.JPEG')):
            if '_merged_mask' in p.name:
                continue
            label = mappings[wnids[p.name.split('_')[0]]][1]
            records.append(dict(path=str(p), label=label, split=split))
    rng = np.random.default_rng(seed)
    for k in range(10):
        ids = np.array([i for i, r in enumerate(records) if r['split'] == 'train' and r['label'] == k])
        rng.shuffle(ids)
        for i in ids[:round(len(ids) * .1)]:
            records[int(i)]['split'] = 'val'
    return records


def mask_from_uri(uri):
    im = Image.open(io.BytesIO(base64.b64decode(uri)))
    rgba = np.asarray(im.convert('RGBA'))
    # Native PNGs have transparent zero and green foreground. Explicit RGB
    # avoids interpreting opaque black as foreground in other PNG encodings.
    return Image.fromarray((np.any(rgba[..., :3] > 0, axis=-1) * 255).astype('uint8'))


class RivalRaw(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        p = Path(self.records[i]['path'])
        rgb = np.asarray(Image.open(p).convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)).copy()
        attrs = np.load(p.with_name(p.stem + '_attr_labels.npy'))
        assert attrs.shape == (2, 18) and not attrs[1].any(), 'Ordinary images only'
        with p.with_name(p.stem + '_attr_dict.pkl').open('rb') as f:
            d = pickle.load(f)  # Trusted local dataset annotations, never downloaded executable code.
        masks = np.zeros((19, 16, 16), dtype='float32')
        valid = np.r_[attrs[0] == 0, False]
        for k, v in d.items():
            j = 18 if k == 'entire-object' else ATTRS.index(k)
            mask = mask_from_uri(v).resize((224, 224), Image.Resampling.NEAREST)
            masks[j] = TF.to_tensor(mask)[0].reshape(16, 14, 16, 14).mean((1, 3)).numpy()
            valid[j] = True
        return i, torch.from_numpy(rgb).permute(2, 0, 1), torch.from_numpy(attrs[0].astype('float32')), torch.from_numpy(masks), torch.from_numpy(valid)
