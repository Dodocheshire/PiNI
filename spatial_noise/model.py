"""One shared RGB view supervises every concept; inference needs two encodes."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class SharedSpatialCBM(nn.Module):
    def __init__(self, texts, classes, width=32, mean_limit=.08, std_limit=.8,
                 spatial_noise_prior=False):
        super().__init__()
        texts = F.normalize(texts.float(), dim=-1)
        k, d = texts.shape
        self.register_buffer('texts', texts)
        # Fixed semantic coordinates replace two learned projection networks.
        width=min(width,k)
        with torch.no_grad():
            _,_,directions=torch.linalg.svd(texts,full_matrices=False)
        self.register_buffer('semantic_projection',directions[:width].T.contiguous())
        self.foreground = nn.Linear(d, 1)
        self.region_bias = nn.Parameter(torch.zeros(k))
        self.region_scale = nn.Parameter(torch.tensor(4.0))
        self.readout = nn.Parameter(texts.clone() * 4)
        self.concept_bias = nn.Parameter(torch.zeros(k))
        self.generator = nn.Linear(width, 4)
        nn.init.zeros_(self.generator.weight)
        nn.init.zeros_(self.generator.bias)
        self.generator.bias.data[3] = -1.0
        self.classifier = nn.Linear(k, classes)
        self.register_buffer('concept_center', torch.zeros(k))
        self.register_buffer('concept_scale', torch.ones(k))
        self.mean_limit = mean_limit
        self.std_limit = std_limit
        self.spatial_noise_prior = bool(spatial_noise_prior)
        self.k = k

    def configuration(self):
        return {'width': self.semantic_projection.shape[1], 'mean_limit': self.mean_limit,
                'std_limit': self.std_limit, 'spatial_noise_prior': self.spatial_noise_prior}

    @classmethod
    def from_checkpoint(cls, checkpoint):
        # Existing v3 checkpoints were trained with the spatial prior enabled.
        configuration = checkpoint.get('model_config', {'spatial_noise_prior': True})
        model = cls(checkpoint['model']['texts'], checkpoint['classes'], **configuration)
        model.load_state_dict(checkpoint['model'])
        return model

    def regions(self, tokens):
        z = F.normalize(tokens[:, 1:].float(), dim=-1)
        h = F.gelu(z @ self.semantic_projection)
        q = F.normalize(self.texts @ self.semantic_projection, dim=-1)
        maps = self.region_scale * torch.einsum('bpd,kd->bkp', z, self.texts)
        maps = maps + self.region_bias[None, :, None]
        foreground = self.foreground(z).squeeze(-1)
        return z, h, q, maps, foreground

    def recognize(self, tokens, foreground_prior=None):
        z, h, q, maps, foreground = self.regions(tokens)
        fg = foreground.sigmoid() if foreground_prior is None else foreground_prior
        relevance = maps.sigmoid() * fg[:, None]
        attention = relevance / relevance.sum(-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum('bkp,bpd->bkd', attention, z)
        logits = (pooled * self.readout[None]).sum(-1) + self.concept_bias
        return logits, maps, foreground

    def distribution(self, original_tokens):
        z, h, queries, maps, fg_logits = self.regions(original_tokens)
        concept_logits, _, _ = self.recognize(original_tokens)
        fg = fg_logits.sigmoid().detach()
        relevance = maps.sigmoid().detach() * concept_logits.sigmoid().detach()[:, :, None]
        weights = relevance / relevance.sum(1, keepdim=True).clamp_min(1e-6)
        # All concept texts jointly condition ONE pixel decoder.
        context = torch.einsum('bkp,kw->bpw', weights, queries.detach())
        decoded = self.generator(F.gelu(h.detach() + context))
        mu = self.mean_limit * decoded[:, :, :3].tanh()
        scale = self.std_limit * decoded[:, :, 3].sigmoid()
        if self.spatial_noise_prior:
            support = fg * (.25 + .75 * relevance.amax(1))
            mu = mu * support[:, :, None]
            scale = scale * (1 - fg)
        sigma = .002 + scale
        grid = math.isqrt(mu.shape[1])
        assert grid * grid == mu.shape[1]
        up = lambda x: F.interpolate(x, (224, 224), mode='bilinear', align_corners=False)
        mu = up(mu.transpose(1, 2).reshape(-1, 3, grid, grid))
        sigma = up(sigma.reshape(-1, 1, grid, grid))
        return mu, sigma, fg

    def perturbation_penalty(self, mu, sigma, foreground, mode):
        # The uniform magnitude penalty carries no foreground/background preference.
        penalty = .01 * mu.square().mean()
        if mode == 'noise' and self.spatial_noise_prior:
            grid = math.isqrt(foreground.shape[1])
            background = 1 - F.interpolate(foreground.reshape(-1, 1, grid, grid),
                                          sigma.shape[-2:], mode='bilinear', align_corners=False)
            ratio = (sigma / .18).clamp_min(1e-3)
            kl = ((ratio.square() - 1 - 2 * ratio.log()) * background).sum()
            penalty = penalty + .02 * kl / background.sum().clamp_min(1)
        return penalty

    def perturb(self, rgb, original_tokens, mode='noise', epsilon=None):
        mu, sigma, fg = self.distribution(original_tokens)
        if mode == 'clean':
            return rgb, mu * 0, sigma * 0, fg
        if mode == 'mean':
            return rgb + mu, mu, sigma * 0, fg
        if mode != 'noise':
            raise ValueError(mode)
        epsilon = self.sample_noise(rgb) if epsilon is None else epsilon
        return rgb + mu + sigma * epsilon, mu, sigma, fg

    @staticmethod
    def sample_noise(rgb):
        """Correlated Gaussian noise with unit variance at every RGB pixel."""
        size=14
        coarse=torch.randn(len(rgb),3,size,size,device=rgb.device,dtype=rgb.dtype)
        smooth=F.interpolate(coarse,rgb.shape[-2:],mode='bilinear',align_corners=False)
        # A bilinear weighted sum has variance equal to the sum of squared weights.
        coordinate=((torch.arange(rgb.shape[-1],device=rgb.device)+.5)*size/rgb.shape[-1]-.5).clamp(0,size-1)
        fraction=coordinate-coordinate.floor()
        variance=(1-fraction).square()+fraction.square()
        smooth=smooth/(variance[:,None]*variance[None,:]).sqrt()[None,None]
        return .95**.5*smooth+.05**.5*torch.randn_like(rgb)

    def encode_view(self, backbone, processed, foreground):
        """Suppress background keys inside the same second ViT encode."""
        bias=2*foreground.detach().clamp_min(.01).log()
        bias=F.pad(bias,(1,0),value=0)
        return backbone.encode_image(processed,spatial=True,key_padding_mask=bias)

    def classify(self, concept_logits):
        return self.classifier((concept_logits.sigmoid() - self.concept_center) / self.concept_scale)

    def freeze_recognizer(self):
        for name, parameter in self.named_parameters():
            if not name.startswith(('generator.', 'classifier.', 'readout', 'concept_bias')):
                parameter.requires_grad_(False)


class SALFConcepts(nn.Module):
    """CVPR2025 SALF projection, red-circle concept alignment, softmax pooling."""
    def __init__(self, texts, classes):
        super().__init__()
        self.projection = nn.Linear(texts.shape[1], len(texts), bias=False)
        self.projection.weight.data.copy_(F.normalize(texts.float(), dim=-1) * 4)
        self.bias = nn.Parameter(torch.zeros(len(texts)))
        self.classifier = nn.Linear(len(texts), classes)
        self.register_buffer('concept_center', torch.zeros(len(texts)))
        self.register_buffer('concept_scale', torch.ones(len(texts)))

    def recognize(self, tokens):
        patches=F.normalize(tokens[:, 1:].float(), dim=-1).transpose(1,2).reshape(-1,512,14,14)
        patches=F.interpolate(patches,(7,7),mode='bilinear',align_corners=False).flatten(2).transpose(1,2)
        maps = self.projection(patches).transpose(1, 2)
        scores = (maps * maps.softmax(-1)).sum(-1) + self.bias
        return scores, maps

    def classify(self, concepts):
        return self.classifier((concepts - self.concept_center) / self.concept_scale)
