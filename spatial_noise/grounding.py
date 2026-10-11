"""Frozen box annotations and image features from the same Grounding DINO pass."""
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from .model import SharedSpatialCBM


DEFAULT_DETECTOR = Path('../D-Pi/artifacts/models/grounding-dino-tiny')
BIRD_PHRASES = ['bird', 'bird head', 'bird beak', 'bird wing', 'bird tail',
                'bird eye', 'bird leg', 'bird belly', 'bird breast', 'bird back',
                'bird throat', 'bird crown']
# Zero-based native CUB part indices. These maps evaluate anatomy, not attributes.
BIRD_PARTS = [[], [4, 5, 6, 9, 10], [1], [8, 12], [13], [6, 10], [7, 11],
              [2], [3], [0], [14], [4]]


def box_maps(boxes, scores, valid, grid=14):
    """Rasterize normalized xyxy boxes by patch overlap, weighted by confidence."""
    edges = torch.linspace(0, 1, grid + 1, device=boxes.device)
    x_overlap = (torch.minimum(boxes[..., 2, None], edges[1:]) -
                 torch.maximum(boxes[..., 0, None], edges[:-1])).clamp_min(0) * grid
    y_overlap = (torch.minimum(boxes[..., 3, None], edges[1:]) -
                 torch.maximum(boxes[..., 1, None], edges[:-1])).clamp_min(0) * grid
    return y_overlap[..., :, None] * x_overlap[..., None, :] * (scores * valid)[..., None, None]


class FrozenGrounder:
    def __init__(self, phrases=BIRD_PHRASES, model_path=DEFAULT_DETECTOR,
                 device='cuda', image_size=800, threshold=.25, precision='fp32'):
        self.phrases = list(phrases)
        self.caption = ' . '.join(self.phrases) + ' .'
        self.device, self.image_size, self.threshold = device, image_size, threshold
        if precision not in ['fp32','fp16']:
            raise ValueError(precision)
        self.precision=precision
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path, local_files_only=True).to(device).eval().requires_grad_(False)
        tokenized = self.processor.tokenizer(self.caption, return_offsets_mapping=True)
        if len(tokenized['input_ids']) > self.detector.config.max_text_len:
            raise ValueError('Caption exceeds the detector text budget; shorten the phrase bank.')
        cursor = 0
        self.phrase_tokens = []
        for phrase in self.phrases:
            start = self.caption.index(phrase, cursor)
            end = start + len(phrase)
            ids = [i for i, (a, b) in enumerate(tokenized['offset_mapping'])
                   if b > start and a < end and b > a]
            assert ids, phrase
            self.phrase_tokens.append(ids)
            cursor = end
        self._image_feature = None
        self.image_encodes = 0
        self._hook = self.detector.model.input_proj_vision[0].register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.image_encodes += 1
        # Reuse the detector's stride-8, 256-channel feature map; no second image pass.
        self._image_feature = F.adaptive_avg_pool2d(output.detach(), (14, 14))

    @torch.no_grad()
    def __call__(self, images):
        if any(image.width != image.height for image in images):
            raise ValueError('Supply square RGB views so box and feature coordinates share the noise image grid.')
        self._image_feature = None
        inputs = self.processor(images=images, text=[self.caption] * len(images), return_tensors='pt',
                                size={'shortest_edge': self.image_size, 'longest_edge': self.image_size}).to(self.device)
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()
        started = time.perf_counter()
        before = self.image_encodes
        with torch.autocast(self.device.split(':')[0],dtype=torch.float16,enabled=self.precision=='fp16'):
            outputs = self.detector(**inputs)
        probabilities = outputs.logits.sigmoid()
        phrase_scores = torch.stack([probabilities[..., ids].mean(-1) for ids in self.phrase_tokens], -1)
        scores, queries = phrase_scores.max(1)
        boxes = outputs.pred_boxes.gather(1, queries[..., None].expand(-1, -1, 4))
        center, extent = boxes[..., :2], boxes[..., 2:]
        boxes = torch.cat((center - extent / 2, center + extent / 2), -1).clamp(0, 1)
        valid = scores >= self.threshold
        maps = box_maps(boxes, scores, valid)
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        assert self._image_feature is not None and self.image_encodes - before == 1
        result = {'features': self._image_feature.clone(), 'boxes': boxes, 'scores': scores,
                  'valid': valid, 'maps': maps, 'detector_forward_s': elapsed,
                  'input_shape': list(inputs.pixel_values.shape), 'image_encodes': 1}
        self._image_feature = None
        return result

    def close(self):
        self._hook.remove()


class GroundedNoiseGenerator(nn.Module):
    """One tiny decoder conditioned on DINO image features and fixed box maps."""
    def __init__(self, regions=len(BIRD_PHRASES), width=32, mean_limit=.08, std_limit=.8,
                 spatial_noise_prior=False):
        super().__init__()
        self.decoder = nn.Sequential(nn.Conv2d(256 + regions, width, 1), nn.GELU(),
                                     nn.Conv2d(width, 4, 1))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        self.decoder[-1].bias.data[3] = -1
        self.mean_limit, self.std_limit = mean_limit, std_limit
        self.spatial_noise_prior = spatial_noise_prior

    def distribution(self, grounded):
        features = grounded['features'].detach().float()
        features = F.layer_norm(features.permute(0, 2, 3, 1), (256,)).permute(0, 3, 1, 2)
        maps = grounded['maps'].detach().float()
        decoded = self.decoder(torch.cat((features, maps), 1))
        mu = self.mean_limit * decoded[:, :3].tanh()
        scale = self.std_limit * decoded[:, 3:4].sigmoid()
        if self.spatial_noise_prior:
            foreground = maps[:, :1]
            mu = mu * foreground * (.25 + .75 * maps[:, 1:].amax(1, keepdim=True))
            scale = scale * (1 - foreground)
        up = lambda v: F.interpolate(v, (224, 224), mode='bilinear', align_corners=False)
        return up(mu), up(.002 + scale)

    def forward(self, rgb, grounded, mode='noise', epsilon=None):
        mu, sigma = self.distribution(grounded)
        if mode == 'mean':
            return rgb + mu, mu, sigma * 0
        if mode != 'noise':
            raise ValueError(mode)
        if epsilon is None:
            epsilon = SharedSpatialCBM.sample_noise(rgb)
        return rgb + mu + sigma * epsilon, mu, sigma


def cub_concept_regions(concepts):
    """Map native attribute groups to anatomy; no image-specific labels are used."""
    lookup = [('bill', 2), ('wing', 3), ('tail', 4), ('eye', 5), ('leg', 6),
              ('belly', 7), ('breast', 8), ('back', 9), ('throat', 10), ('crown', 11),
              ('head', 1), ('forehead', 1), ('nape', 1)]
    return [next((index for word, index in lookup if word in c.split('::')[0]), 0)
            for c in concepts]


class GroundedConceptReadout(nn.Module):
    """Learn attribute scores on DINO box-pooled CLIP features, without a region predictor."""
    def __init__(self, texts, region_indices):
        super().__init__()
        self.register_buffer('region_indices', torch.tensor(region_indices, dtype=torch.long))
        self.weight = nn.Parameter(F.normalize(texts.float(), dim=-1) * 4)
        self.bias = nn.Parameter(torch.zeros(len(texts)))

    def forward(self, clip_tokens, grounded):
        maps = grounded['maps'][:, self.region_indices].detach().flatten(2)
        # Missing part boxes use the bird box; a missing bird uses the full image.
        bird = grounded['maps'][:, :1].detach().flatten(2)
        bird = torch.where(bird.sum(-1, keepdim=True) > 0, bird, torch.ones_like(bird))
        maps = torch.where(maps.sum(-1, keepdim=True) > 0, maps, bird)
        attention = maps / maps.sum(-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum('bkp,bpd->bkd', attention, F.normalize(clip_tokens[:, 1:].float(), dim=-1))
        return (pooled * self.weight).sum(-1) + self.bias
