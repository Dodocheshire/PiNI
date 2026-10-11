"""CUB-trained ResNet18 and official Koh-112 splits, before regional/noise adaptation."""
import argparse
import copy
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pytorchcv.model_provider import get_model

from .data import write_json

ROOT = Path('artifacts/cub_aligned_rn18')
OFFICIAL = Path('/home/pengcheng/SOTA-CBM/_shared_datasets/CUB_processed/class_attr_data_10')
SOURCE = Path('artifacts/cross_dataset/cub/cache')


def normalize(rgb):
    mean = rgb.new_tensor([.485, .456, .406])[None, :, None, None]
    std = rgb.new_tensor([.229, .224, .225])[None, :, None, None]
    return (rgb - mean) / std


class BirdBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        network = get_model('resnet18_cub', pretrained=True)
        self.spatial = nn.Sequential(*list(network.features.children())[:-1])
        self.original_classifier = network.output
        self.eval().requires_grad_(False)

    def forward(self, rgb):
        return self.spatial(normalize(rgb))


class BirdImages(Dataset):
    def __init__(self, records):
        self.records = records
        self.resize = transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR)
        self.crop = transforms.CenterCrop(224)
        self.mask_resize = transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record['path']) as im:
            rgb = np.asarray(self.crop(self.resize(im.convert('RGB')))).transpose(2, 0, 1).copy()
        mask_path = Path('artifacts/datasets/local/segmentations')/Path(record['relative_path']).with_suffix('.png')
        with Image.open(mask_path) as im:
            foreground = np.asarray(self.crop(self.mask_resize(im.convert('L')))) > 127
        return torch.from_numpy(rgb), torch.from_numpy(foreground.copy())


@torch.no_grad()
def prepare(backbone, batch=128):
    cache = ROOT/'cache'
    if (cache/'COMPLETE.json').exists():
        return
    cache.mkdir(parents=True, exist_ok=True)
    source = json.loads((SOURCE/'manifest.json').read_text())
    source_ids = np.load(SOURCE/'native_image_ids.npy')
    by_id = {int(i): record for i, record in zip(source_ids, source)}
    records, labels, attributes, native_ids, splits, hashes = [], [], [], [], {}, {}
    for split in ['train', 'val', 'test']:
        path = OFFICIAL/(split+'.pkl')
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open('rb') as stream:
            rows = pickle.load(stream)
        start = len(records)
        for row in rows:
            record = by_id[row['id']]
            assert record['label'] == row['class_label']
            records.append(record); labels.append(row['class_label'])
            attributes.append(row['attribute_label']); native_ids.append(row['id'])
        splits[split] = list(range(start, len(records)))
    assert len(records) == len(set(native_ids)) == 11788
    attributes = np.asarray(attributes, dtype='float32'); labels = np.asarray(labels, dtype='int64')
    assert attributes.shape == (11788, 112)
    assert all(len(np.unique(attributes[labels == k], axis=0)) == 1 for k in range(200))
    np.save(cache/'labels.npy', labels); np.save(cache/'attrs.npy', attributes)
    np.save(cache/'native_ids.npy', np.asarray(native_ids))
    (cache/'concepts.json').write_text((SOURCE/'concepts.json').read_text())
    rgb = np.lib.format.open_memmap(cache/'rgb_original.npy', mode='w+', dtype='uint8', shape=(len(records), 3, 224, 224))
    foreground = np.lib.format.open_memmap(cache/'foreground.npy', mode='w+', dtype='bool', shape=(len(records), 224, 224))
    features = np.lib.format.open_memmap(cache/'features_original.npy', mode='w+', dtype='float16', shape=(len(records), 512, 7, 7))
    logits = np.empty((len(records), 200), dtype='float32')
    started = time.time(); offset = 0
    loader = DataLoader(BirdImages(records), batch_size=batch, num_workers=8, pin_memory=True)
    for images, masks in loader:
        end = offset + len(images)
        rgb[offset:end] = images.numpy(); foreground[offset:end] = masks.numpy()
        with torch.autocast('cuda', dtype=torch.float16):
            feat = backbone(images.cuda(non_blocking=True).float()/255)
            y = backbone.original_classifier(feat.mean((2, 3)))
        features[offset:end] = feat.cpu().numpy(); logits[offset:end] = y.float().cpu().numpy()
        offset = end
        if offset % (batch*20) == 0 or offset == len(records):
            print(json.dumps({'stage': 'aligned_cache', 'done': offset, 'total': len(records),
                              'seconds': round(time.time()-started, 1)}), flush=True)
    rgb.flush(); foreground.flush(); features.flush()
    np.save(cache/'pretrained_logits.npy', logits)
    protocol = {'dataset': 'CUB-200-2011', 'concepts': 'Koh official class-level majority-vote 112',
                'backbone': 'pytorchcv resnet18_cub, CUB-trained, frozen',
                'backbone_sha256': hashlib.sha256(Path('/root/.torch/models/resnet18_cub-2333-200d8b9c.pth').read_bytes()).hexdigest(),
                'official_source': str(OFFICIAL), 'official_pickle_sha256': hashes,
                'counts': {k: len(v) for k, v in splits.items()}, 'splits': splits,
                'preprocess': 'RGB conversion, Resize(256) bilinear, CenterCrop(224), ImageNet mean/std',
                'foreground': 'native segmentation >127, same geometric resize/crop as RGB',
                'pretrained_classifier_accuracy': {k: float((logits[v].argmax(1) == labels[v]).mean()) for k,v in splits.items()},
                'seconds': time.time()-started}
    write_json(cache/'protocol.json', protocol)
    write_json(cache/'COMPLETE.json', {'images': len(records)})
    print(json.dumps({'stage': 'pretrained_classifier', 'accuracy': protocol['pretrained_classifier_accuracy']}), flush=True)


def load():
    cache = ROOT/'cache'
    protocol = json.loads((cache/'protocol.json').read_text())
    splits = {k: np.asarray(v) for k, v in protocol['splits'].items()}
    arrays = {k: np.load(cache/(k+'.npy'), mmap_mode='r') for k in
              ['labels', 'attrs', 'rgb_original', 'foreground', 'features_original']}
    return arrays, splits, protocol


def tensor(array, ids=None):
    return torch.as_tensor(np.array(array if ids is None else array[ids]), device='cuda').float()


class SimpleCBM(nn.Module):
    def __init__(self, concepts=112, classes=200):
        super().__init__()
        self.register_buffer('feature_center', torch.zeros(512))
        self.register_buffer('feature_scale', torch.ones(512))
        self.concepts = nn.Linear(512, concepts)
        self.register_buffer('concept_center', torch.zeros(concepts))
        self.register_buffer('concept_scale', torch.ones(concepts))
        self.classifier = nn.Linear(concepts, classes)

    def concept_logits(self, feature_maps):
        features = feature_maps.float().mean((-2, -1)) if feature_maps.ndim == 4 else feature_maps.float()
        return self.concepts((features.float()-self.feature_center)/self.feature_scale)

    def classify(self, concepts):
        return self.classifier((concepts-self.concept_center)/self.concept_scale)

    def forward(self, feature_maps):
        c = self.concept_logits(feature_maps)
        return self.classify(c), c


def fit_classifier(model, features, labels, steps=1000):
    """Fit on frozen concept logits, choose checkpoint using validation accuracy."""
    model.concept_center.copy_(features['train'].mean(0))
    model.concept_scale.copy_(features['train'].std(0).clamp_min(.03))
    nn.init.zeros_(model.classifier.weight); nn.init.zeros_(model.classifier.bias)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=.01, weight_decay=.001)
    best = -1; state = None
    for step in range(steps):
        loss = F.cross_entropy(model.classify(features['train']), labels['train'])
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if (step+1) % 25 == 0:
            with torch.no_grad():
                accuracy = float((model.classify(features['val']).argmax(1) == labels['val']).float().mean())
            if accuracy > best:
                best = accuracy; state = copy.deepcopy(model.classifier.state_dict()); selected = step+1
    model.classifier.load_state_dict(state)
    return {'selected_step': selected, 'validation_accuracy': best}


def baseline(seed=42, steps=1000):
    destination = ROOT/'simple'/f'seed{seed}'; destination.mkdir(parents=True, exist_ok=True)
    if (destination/'COMPLETE.json').exists():
        return
    torch.manual_seed(seed)
    arrays, splits, protocol = load()
    x = {k: tensor(arrays['features_original'], v).mean((2,3)) for k,v in splits.items()}
    labels = {k: tensor(arrays['labels'], v).long() for k,v in splits.items()}
    target = {k: tensor(arrays['attrs'], v) for k,v in splits.items()}
    model = SimpleCBM().cuda()
    model.feature_center.copy_(x['train'].mean(0)); model.feature_scale.copy_(x['train'].std(0).clamp_min(.03))
    priors = target['train'].mean(0).clamp(.02, .98); pos = ((1-priors)/priors).clamp(1,10)
    optimizer = torch.optim.AdamW(model.concepts.parameters(), lr=.01, weight_decay=.001)
    best = float('inf'); history = []; started = time.time()
    for step in range(steps):
        loss = F.binary_cross_entropy_with_logits(model.concept_logits(x['train']), target['train'], pos_weight=pos)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if (step+1) % 25 == 0:
            with torch.no_grad():
                validation = float(F.binary_cross_entropy_with_logits(model.concept_logits(x['val']), target['val'], pos_weight=pos))
            history.append({'step': step+1, 'train_bce': float(loss.detach()), 'validation_bce': validation})
            if validation < best:
                best = validation; state = copy.deepcopy(model.concepts.state_dict()); selected = step+1
    model.concepts.load_state_dict(state)
    with torch.no_grad():
        concepts = {k: model.concept_logits(v) for k,v in x.items()}
    head_info = fit_classifier(model, concepts, labels, steps)
    result = {'seed': seed, 'concept_selected_step': selected, 'classifier': head_info,
              'seconds': time.time()-started, 'trainable_parameters': sum(p.numel() for p in model.parameters())}
    with torch.no_grad():
        for split, c in concepts.items():
            logits = model.classify(c)
            result[split] = {'accuracy': float((logits.argmax(1) == labels[split]).float().mean()),
                             'concept_accuracy': float(((c > 0) == target[split].bool()).float().mean()),
                             'images': len(c)}
            np.save(destination/f'{split}_concepts.npy', c.cpu().numpy())
            np.save(destination/f'{split}_logits.npy', logits.cpu().numpy())
    torch.save({'model': model.state_dict(), 'seed': seed, 'architecture': 'SimpleCBM',
                'protocol': str(ROOT/'cache/protocol.json')}, destination/'best.pt')
    write_json(destination/'history.json', history); write_json(destination/'results.json', result)
    write_json(destination/'COMPLETE.json', result)
    print(json.dumps({'stage': 'simple_cbm', 'results': result}), flush=True)


def refit(seed=42, steps=1000):
    """Refit fresh heads on train+val; training metrics select the checkpoint."""
    destination = ROOT/'simple_refit'/f'seed{seed}'
    if (destination/'COMPLETE.json').exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    arrays, splits, protocol = load()
    development = np.concatenate([splits['train'], splits['val']])
    assert len(development) == 5994 and not set(development) & set(splits['test'])
    x = tensor(arrays['features_original']).mean((2, 3))
    target = tensor(arrays['attrs']); labels = tensor(arrays['labels']).long()
    torch.manual_seed(seed)
    model = SimpleCBM().cuda()
    model.feature_center.copy_(x[development].mean(0))
    model.feature_scale.copy_(x[development].std(0).clamp_min(.03))
    priors = target[development].mean(0).clamp(.02, .98)
    pos = ((1-priors)/priors).clamp(1, 10)
    optimizer = torch.optim.AdamW(model.concepts.parameters(), lr=.01, weight_decay=.001)
    best = -1; history = []; started = time.time()
    for step in range(steps):
        c = model.concept_logits(x[development])
        loss = F.binary_cross_entropy_with_logits(c, target[development], pos_weight=pos)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if (step+1) % 25 == 0:
            with torch.no_grad():
                c = model.concept_logits(x[development])
                accuracy = float(((c > 0) == target[development].bool()).float().mean())
            history.append({'step': step+1, 'concept_accuracy': accuracy, 'loss': float(loss.detach())})
            if accuracy > best:
                best = accuracy; state = copy.deepcopy(model.concepts.state_dict()); selected = step+1
    model.concepts.load_state_dict(state)
    with torch.no_grad():
        concepts = model.concept_logits(x)
    head_info = fit_classifier(model, {'train': concepts[development], 'val': concepts[development]},
                               {'train': labels[development], 'val': labels[development]}, steps)
    head_info['training_accuracy'] = head_info.pop('validation_accuracy')
    result = {'seed': seed, 'fit_images': len(development), 'concept_selected_step': selected,
              'classifier': head_info, 'seconds': time.time()-started,
              'selection': 'maximum train+val concept accuracy, then maximum train+val task accuracy; first tie',
              'configuration': {'lr': .01, 'weight_decay': .001, 'max_steps': steps},
              'trainable_parameters': sum(p.numel() for p in model.parameters())}
    with torch.no_grad():
        for split, ids in {**splits, 'trainval': development}.items():
            c = concepts[ids]; logits = model.classify(c)
            result[split] = {'accuracy': float((logits.argmax(1) == labels[ids]).float().mean()),
                             'concept_accuracy': float(((c > 0) == target[ids].bool()).float().mean()),
                             'images': len(ids)}
            np.save(destination/f'{split}_concepts.npy', c.cpu().numpy())
            np.save(destination/f'{split}_logits.npy', logits.cpu().numpy())
    torch.save({'model': model.state_dict(), 'seed': seed, 'architecture': 'SimpleCBM',
                'protocol': str(ROOT/'cache/protocol.json'), 'refit': True}, destination/'best.pt')
    write_json(destination/'history.json', history); write_json(destination/'results.json', result)
    write_json(destination/'COMPLETE.json', result)
    print(json.dumps({'stage': 'simple_cbm_refit', 'results': result}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['baseline', 'refit'], default='baseline')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=1000)
    args = parser.parse_args()
    torch.set_num_threads(6)
    backbone = BirdBackbone().cuda()
    prepare(backbone)
    baseline(args.seed, args.steps)
    if args.phase == 'refit':
        refit(args.seed, args.steps)


if __name__ == '__main__':
    main()
