"""Same-backbone, 18-concept CBM baselines on the fixed RIVAL10 split.

Sequential supervised CBM: https://proceedings.mlr.press/v119/koh20a.html
LF-CBM projection: https://github.com/Trustworthy-ML-Lab/Label-free-CBM
Both use this experiment's validation-selected proximal-L1 classification head.
The LF projection keeps the predefined 18 native concepts and uses the original
center/cube/normalize similarity; no native concept labels enter its optimizer.
"""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

from .train_noise import fixed_subset
from .train_spatial import load_cache, fit_head, head_predict


ROOT = Path('artifacts/paper_baselines')
SEEDS = [42, 43, 44]


def cos_cubed(teacher, predicted):
    # Same algebra as upstream cos_similarity_cubed_single.
    a = (teacher.float() - teacher.float().mean(0, keepdim=True)).pow(3)
    b = (predicted - predicted.mean(0, keepdim=True)).pow(3)
    return (F.normalize(a, dim=0) * F.normalize(b, dim=0)).sum(0)


def features(arrays):
    destination = ROOT / 'original_cls.npy'
    if not destination.exists():
        values = np.empty((len(arrays['tokens']), 768), dtype='float32')
        for start in range(0, len(values), 512):
            z = torch.tensor(np.array(arrays['tokens'][start:start+512, 0]), dtype=torch.float32)
            values[start:start+len(z)] = F.normalize(z, dim=-1).numpy()
        np.save(destination, values)
    return torch.from_numpy(np.load(destination))


def train():
    torch.set_num_threads(4)
    ROOT.mkdir(parents=True, exist_ok=True)
    arrays, splits, labels, _ = load_cache('artifacts/rival_vitl14')
    x = features(arrays)
    text = torch.from_numpy(np.load('artifacts/rival_vitl14/text.npy'))
    teacher = x @ text.T
    targets = torch.from_numpy(np.array(arrays['attrs']))
    head_train = fixed_subset(splits['train'], labels, 6000)
    for method in ['supervised_cbm', 'lf_cbm']:
        for seed in SEEDS:
            out = ROOT / f'{method}_seed{seed}'
            out.mkdir(exist_ok=True)
            if (out/'original_results.json').exists():
                print('Reusing completed baseline', method, seed, flush=True)
                continue
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            projection = nn.Linear(768, 18, bias=(method=='supervised_cbm'))
            history = []
            best = -float('inf'); state = None
            if method == 'supervised_cbm':
                optimizer = torch.optim.AdamW(projection.parameters(), lr=.003, weight_decay=.001)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 40)
                for epoch in range(40):
                    order = np.random.default_rng(seed+epoch).permutation(splits['train'])
                    for start in range(0, len(order), 256):
                        ids = order[start:start+256]
                        loss = F.binary_cross_entropy_with_logits(projection(x[ids]), targets[ids])
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                    scheduler.step()
                    with torch.no_grad():
                        scores = projection(x[splits['val']]).sigmoid().numpy()
                        score = float(average_precision_score(
                            np.array(arrays['attrs'][splits['val']]), scores, average='macro'))
                    history.append(dict(epoch=epoch+1, validation_mAP=score))
                    if score > best:
                        best = score; state = copy.deepcopy(projection.state_dict())
            else:
                optimizer = torch.optim.Adam(projection.parameters(), lr=.001)
                population = splits['train'].tolist()
                for step in range(1000):
                    ids = random.sample(population, min(4096, len(population)))
                    loss = -cos_cubed(teacher[ids], projection(x[ids])).mean()
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    if step % 50 == 0 or step == 999:
                        with torch.no_grad():
                            score = float(cos_cubed(teacher[splits['val']], projection(x[splits['val']])).mean())
                        history.append(dict(step=step, validation_cos_cubed=score))
                        if score > best:
                            best = score; state = copy.deepcopy(projection.state_dict())
                        else:
                            break
            projection.load_state_dict(state)
            with torch.no_grad():
                scores = projection(x)
                if method == 'supervised_cbm': scores = scores.sigmoid()
                scores = scores.numpy()
            head, curve, selected = fit_head(scores[head_train], labels[head_train],
                scores[splits['val']], labels[splits['val']], device='cpu')
            torch.save(dict(method=method, seed=seed, model=state, head=head), out/'model.pt')
            logits = head_predict(head, scores[splits['test']])
            metrics = dict(accuracy=float((logits.argmax(1)==labels[splits['test']]).mean()),
                NEC=curve[selected]['NEC'], mAP=float(average_precision_score(
                    np.array(arrays['attrs'][splits['test']]),scores[splits['test']],average='macro')))
            np.save(out/'test_scores.npy', scores[splits['test']])
            np.save(out/'test_class_logits.npy', logits)
            result = dict(method=method, seed=seed, original=metrics, history=history,
                head_grid=curve, head_selected=selected, concept_train_n=len(splits['train']),
                head_train_n=len(head_train), validation_n=len(splits['val']), test_n=len(splits['test']),
                concept_supervision=method=='supervised_cbm', region_supervision=False,
                class_head='same proximal-L1 grid as spatial noise CBM',
                retained_concepts=18, backbone='frozen CLIP ViT-L/14 normalized CLS 768')
            (out/'original_results.json').write_text(json.dumps(result, indent=2))
            print(method, seed, json.dumps(metrics), flush=True)


def evaluate():
    from clip import clip
    from .data import normalize
    torch.set_num_threads(4)
    arrays, _, labels, _ = load_cache('artifacts/rival_vitl14')
    saved = np.load('artifacts/background_intervention_region/intervention.npz')
    ids, donors, images = saved['ids'], saved['donors'], saved['shifted']
    fingerprint = hashlib.sha256(ids.tobytes()+donors.tobytes()+images.tobytes()).hexdigest()
    tokens_path = ROOT/'background_cls.npy'
    metadata_path = ROOT/'background_features.json'
    if tokens_path.exists() and metadata_path.exists() and json.loads(metadata_path.read_text())['intervention_sha256']==fingerprint:
        x = torch.from_numpy(np.load(tokens_path))
    else:
        backbone, _ = clip.load('ViT-L/14', device='cuda', jit=False)
        backbone.float().eval().requires_grad_(False)
        rows = []
        with torch.no_grad():
            for start in range(0, len(ids), 128):
                rgb = torch.tensor(images[start:start+128], device='cuda')
                with torch.autocast('cuda', dtype=torch.float16):
                    tokens = backbone.encode_image(normalize(rgb), spatial=True)
                rows.append(F.normalize(tokens[:, 0].float(), dim=-1).cpu())
        x = torch.cat(rows)
        np.save(tokens_path, x.numpy())
        metadata_path.write_text(json.dumps(dict(intervention_sha256=fingerprint,
            n=len(ids), ids=ids.tolist(), donors=donors.tolist(), frozen_backbone=True,
            source='artifacts/background_intervention_region/intervention.npz'), indent=2))
    truth = np.array(arrays['attrs'][ids])
    results = []
    for method in ['supervised_cbm', 'lf_cbm']:
        for seed in SEEDS:
            out = ROOT/f'{method}_seed{seed}'
            ck = torch.load(out/'model.pt', map_location='cpu', weights_only=False)
            projection = nn.Linear(768,18,bias=(method=='supervised_cbm'))
            projection.load_state_dict(ck['model']); projection.eval()
            with torch.no_grad():
                scores = projection(x)
                if method=='supervised_cbm': scores=scores.sigmoid()
                scores=scores.numpy()
            logits=head_predict(ck['head'],scores)
            metrics=dict(accuracy=float((logits.argmax(1)==labels[ids]).mean()),
                mAP=float(average_precision_score(truth,scores,average='macro')))
            np.save(out/'background_scores.npy',scores)
            np.save(out/'background_class_logits.npy',logits)
            record=json.loads((out/'original_results.json').read_text())
            record.update(background=metrics,intervention_sha256=fingerprint)
            (out/'results.json').write_text(json.dumps(record,indent=2)); results.append(record)
            print(method,seed,'background',json.dumps(metrics),flush=True)
    (ROOT/'results.json').write_text(json.dumps(dict(seeds=SEEDS,models=results,
        intervention_sha256=fingerprint),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['train','evaluate'],required=True)
    args=p.parse_args()
    train() if args.phase=='train' else evaluate()


if __name__=='__main__':main()
