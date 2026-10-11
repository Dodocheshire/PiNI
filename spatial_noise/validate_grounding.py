"""Validate frozen CUB boxes, paired background stability and a one-CLIP gradient path."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .data import ROOT, write_json
from .grounding import (BIRD_PARTS, BIRD_PHRASES, DEFAULT_DETECTOR, FrozenGrounder,
                        GroundedNoiseGenerator, GroundedConceptReadout, cub_concept_regions)


def box_iou(a, b):
    intersection = np.maximum(0, np.minimum(a[..., 2:], b[..., 2:]) -
                              np.maximum(a[..., :2], b[..., :2])).prod(-1)
    area_a = np.maximum(0, a[..., 2:] - a[..., :2]).prod(-1)
    area_b = np.maximum(0, b[..., 2:] - b[..., :2]).prod(-1)
    return intersection / np.maximum(area_a + area_b - intersection, 1e-8)


def metrics(boxes, valid, target_boxes, parts):
    iou = box_iou(boxes[:, 0], target_boxes) * valid[:, 0]
    bird_area = (target_boxes[:, 2:] - target_boxes[:, :2]).prod(-1)
    output = {'bird_box_coverage': float(valid[:, 0].mean()), 'bird_box_mean_iou': float(iou.mean()),
              'bird_box_recall_iou50': float((iou >= .5).mean()), 'parts': {}}
    for k, ids in enumerate(BIRD_PARTS):
        if not ids:
            continue
        points = parts[:, ids, :2]
        visible = parts[:, ids, 2] > 0
        hit = ((points >= boxes[:, k, None, :2]) & (points <= boxes[:, k, None, 2:])).all(-1)
        hit &= valid[:, k, None]
        area_ratio = (boxes[:, k, 2:] - boxes[:, k, :2]).prod(-1) / np.maximum(bird_area, 1e-8)
        output['parts'][BIRD_PHRASES[k]] = {
            'coverage': float(valid[:, k].mean()), 'visible_points': int(visible.sum()),
            'visible_point_recall': float((hit & visible).sum() / max(visible.sum(), 1)),
            'localized_point_recall_area_le_half_bird': float(
                (hit & visible & (area_ratio <= .5)[:, None]).sum() / max(visible.sum(), 1)),
            'median_box_to_bird_area': float(np.median(area_ratio[valid[:, k]])) if valid[:, k].any() else None}
    return output


def load_targets(indices, protocol):
    source = Path('artifacts/cross_dataset/cub/cache')
    root = Path('artifacts/datasets/local/CUB_200_2011')
    records = json.loads((source/'manifest.json').read_text())
    source_ids = np.asarray(protocol['source_indices'])[indices]
    native_ids = np.load(source/'native_image_ids.npy')[source_ids].astype(int) - 1
    boxes = np.loadtxt(root/'bounding_boxes.txt')[:, 1:][native_ids]
    boxes[:, 2:] += boxes[:, :2]
    parts = np.array(np.load(source/'parts.npy', mmap_mode='r')[source_ids])
    for j, index in enumerate(source_ids):
        with Image.open(records[index]['path']) as im:
            width, height = im.size
        boxes[j] /= [width, height, width, height]
        parts[j, :, :2] /= [width, height]
    return boxes, parts


def visualize(destination, records, rgb, target_boxes, parts):
    chosen = list(range(min(6, len(records['indices']))))
    palette = ['#1f77b4', '#e69f00', '#cc79a7', '#009e73']
    shown = [0, 1, 2, 3]
    fig, axes = plt.subplots(2, len(chosen), figsize=(3*len(chosen), 7), squeeze=False)
    for row, condition in enumerate(['original', 'reversed']):
        for col, j in enumerate(chosen):
            ax = axes[row, col]
            ax.imshow(rgb[condition][j].transpose(1, 2, 0))
            for k, color in zip(shown, palette):
                if not records[condition]['valid'][j, k]:
                    continue
                x1, y1, x2, y2 = records[condition]['boxes'][j, k] * 224
                ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=2))
            x1, y1, x2, y2 = target_boxes[j] * 224
            ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='white', linestyle='--'))
            visible = parts[j, :, 2] > 0
            ax.scatter(parts[j, visible, 0]*224, parts[j, visible, 1]*224, s=9, c='white', edgecolors='black', linewidths=.3)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"Validation image {records['indices'][j]}", fontsize=10)
        axes[row, 0].set_ylabel('Original' if row == 0 else 'Changed background', fontsize=12)
    handles = [Rectangle((0, 0), 1, 1, fill=False, edgecolor=c, linewidth=2) for c in palette]
    fig.legend(handles, [BIRD_PHRASES[k] for k in shown], loc='upper center', ncol=4, bbox_to_anchor=(.5, .96))
    fig.suptitle('Frozen Grounding DINO: bird and part boxes | white: native annotation', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .89))
    fig.savefig(destination, dpi=170, bbox_inches='tight'); plt.close(fig)


def gradient_audit(grounder, rgb, cache, index, destination):
    from clip import clip
    from pini_cbm.data import normalize
    torch.manual_seed(42)
    backbone, _ = clip.load('ViT-B/16', device=grounder.device, jit=False)
    backbone.float().eval().requires_grad_(False)
    concepts = json.loads((cache/'concepts.json').read_text())
    head = GroundedConceptReadout(torch.from_numpy(np.load(cache/'text.npy')),
                                  cub_concept_regions(concepts)).to(grounder.device)
    generator = GroundedNoiseGenerator(spatial_noise_prior=False).to(grounder.device)
    calls = {'clip': 0}
    def count(module, args, output):
        calls['clip'] += 1
    hook = backbone.visual.register_forward_hook(count)
    try:
        before = grounder.image_encodes
        grounded = grounder([Image.fromarray(rgb.transpose(1, 2, 0))])
        image = torch.from_numpy(np.array(rgb)).to(grounder.device).float()[None]/255
        processed, mu, sigma = generator(image, grounded)
        with torch.autocast('cuda', dtype=torch.float16):
            features = backbone.encode_image(normalize(processed), spatial=True)
        scores = head(features, grounded)
        target = torch.as_tensor(np.array(np.load(cache/'attrs.npy', mmap_mode='r')[index]),
                                 device=grounder.device).float()[None]
        loss = F.binary_cross_entropy_with_logits(scores, target)
        loss.backward()
        gradients = generator.decoder[-1].weight.grad
        result = {'stage': 'untrained bridge gradient verification',
                  'dino_image_encodes': grounder.image_encodes-before, 'clip_image_encodes': calls['clip'],
                  'mean_gradient_norm': float(gradients[:3].norm()), 'std_gradient_norm': float(gradients[3].norm()),
                  'frozen_dino': all(p.grad is None for p in grounder.detector.parameters()),
                  'frozen_clip': all(p.grad is None for p in backbone.parameters()),
                  'generator_parameters': sum(p.numel() for p in generator.parameters()),
                  'concept_readout_parameters': sum(p.numel() for p in head.parameters()),
                  'frozen_detector_parameters': sum(p.numel() for p in grounder.detector.parameters()),
                  'detector_forward_ms': 1000*grounded['detector_forward_s'],
                  'feature_shape': list(grounded['features'].shape), 'concept_count': len(concepts),
                  'spatial_noise_prior': False, 'initial_concept_bce': float(loss.detach())}
        assert result['dino_image_encodes'] == result['clip_image_encodes'] == 1
        assert result['mean_gradient_norm'] > 0 and result['std_gradient_norm'] > 0
        assert result['frozen_dino'] and result['frozen_clip']
        write_json(destination/'bridge_gradient_audit.json', result)
        print(json.dumps(result), flush=True)
    finally:
        hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--image-size', type=int, default=800)
    parser.add_argument('--threshold', type=float, default=.25)
    parser.add_argument('--model', type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument('--output', type=Path, default=ROOT/'grounding_cub_boxes_v1')
    args = parser.parse_args()
    if args.limit < 1 or args.batch < 1:
        parser.error('--limit and --batch must be positive.')
    torch.set_num_threads(6)
    cache = ROOT/'cub/cache'
    protocol = json.loads((cache/'protocol.json').read_text())
    indices = np.random.default_rng(42).permutation(protocol['splits']['val'])[:args.limit]
    target_boxes, parts = load_targets(indices, protocol)
    grounder = FrozenGrounder(model_path=args.model, image_size=args.image_size, threshold=args.threshold)
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((args.model/'PROVENANCE.json').read_text())
    write_json(args.output/'protocol.json', {'dataset': 'CUB', 'split': 'validation', 'indices': indices.tolist(),
               'phrases': BIRD_PHRASES, 'box_threshold': args.threshold,
               'score': 'mean token probability for each complete phrase; top box per phrase',
               'detector_image_size': args.image_size, 'model': provenance,
               'conditions': ['original', 'reversed'], 'source_cache': str(cache),
               'features': 'stride-8 projected image map, 256 channels, pooled to 14x14 in same detector pass'})
    records, images, results = {'indices': indices.tolist()}, {}, {}
    for condition in ['original', 'reversed']:
        rgb = np.load(cache/f'rgb_{condition}.npy', mmap_mode='r')
        images[condition] = np.array(rgb[indices])
        chunks, timings = [], []
        for start in range(0, len(indices), args.batch):
            batch = [Image.fromarray(x.transpose(1, 2, 0)) for x in images[condition][start:start+args.batch]]
            output = grounder(batch)
            chunks.append({key: output[key].cpu().numpy() for key in ['boxes', 'scores', 'valid']})
            timings.append(output['detector_forward_s']/len(batch))
            if start % 20 == 0:
                print(json.dumps({'condition': condition, 'done': start+len(batch), 'total': len(indices),
                                  'detector_ms_per_image': 1000*timings[-1]}), flush=True)
        records[condition] = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
        np.savez(args.output/f'{condition}.npz', indices=indices, **records[condition],
                 native_boxes=target_boxes, native_parts=parts)
        results[condition] = metrics(records[condition]['boxes'], records[condition]['valid'], target_boxes, parts)
        results[condition]['detector_ms_per_image'] = float(np.mean(timings[1:] or timings)*1000)
        write_json(args.output/'results.json', results)
    paired = box_iou(records['original']['boxes'], records['reversed']['boxes'])
    paired *= records['original']['valid'] & records['reversed']['valid']
    results['paired_box_iou_missing_is_zero'] = dict(zip(BIRD_PHRASES, paired.mean(0).tolist()))
    write_json(args.output/'results.json', results)
    figure = Path('reports/main/grounding_cub_boxes_v1.png')
    visualize(figure, records, images, target_boxes, parts)
    # Use a training image for the gradient audit, separate from box validation.
    train_index = protocol['splits']['train'][0]
    rgb = np.load(cache/'rgb_correlated.npy', mmap_mode='r')[train_index]
    gradient_audit(grounder, rgb, cache, train_index, args.output)
    write_json(args.output/'COMPLETE.json', {'images': len(indices), 'conditions': 2, 'figure': str(figure)})
    grounder.close()


if __name__ == '__main__':
    main()
