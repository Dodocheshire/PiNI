"""Spatial concepts and genuinely sampled RGB noise; no image-to-class bypass."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SpatialConcepts(nn.Module):
    def __init__(self, texts, width=128, region_readout=False):
        super().__init__()
        d = texts.shape[1]; k = len(texts)
        self.register_buffer('texts', texts)
        self.patch = nn.Linear(d, width)
        self.query = nn.Linear(d, width, bias=False)
        self.context = nn.Linear(d, width)
        self.fg = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        self.mask_bias = nn.Linear(d, k)
        self.logit_weights = nn.Parameter(torch.randn(k, d * 2) * .01)
        self.logit_bias = nn.Parameter(torch.zeros(k))
        self.scale = nn.Parameter(torch.tensor(4.0))
        self.k = k
        self.region_readout = region_readout
        # Keep optional readout architecture in the checkpoint itself, so all
        # existing noise/galleries restore it without relying on caller flags.
        self.register_buffer('_region_readout',torch.tensor(region_readout,dtype=torch.bool))

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        key=prefix+'_region_readout'
        if key not in state_dict:
            # Historical V1/V2 checkpoints did not contain this optional flag.
            state_dict[key]=self._region_readout.detach().clone()
        self.region_readout=bool(state_dict[key].item())
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def spatial(self, tokens):
        # Normalized CLIP embeddings; text-conditioned trainable dot products.
        z = F.normalize(tokens.float(), dim=-1)
        h = F.gelu(self.patch(z[:, 1:]) + self.context(z[:, :1]))
        q = F.normalize(self.query(self.texts), dim=-1)
        scores = self.scale * torch.einsum('bpw,kw->bkp', F.normalize(h, dim=-1), q)
        masks = scores + self.mask_bias(z[:, 0]).unsqueeze(-1)
        foreground = self.fg(h).squeeze(-1)
        return z, h, scores, masks, foreground

    def forward(self, tokens):
        z, h, scores, masks, foreground = self.spatial(tokens)
        if self.region_readout:
            # Make the supervised, predicted region part of recognition itself.
            # Contextual CLIP patches may still contain global information, but
            # there is no direct CLS-vector contribution to the concept logit.
            relevance = masks.sigmoid() * foreground[:, None].sigmoid()
            attention = relevance / relevance.sum(-1, keepdim=True).clamp_min(1e-6)
        else:
            attention = (scores * 2).softmax(-1)
        pooled = torch.einsum('bkp,bpd->bkd', attention, z[:, 1:])
        context = z[:, :1].expand(-1, self.k, -1)
        if self.region_readout:
            # A nearly absent predicted region should not be normalized into a
            # confident full-strength token. Max relevance supplies confidence.
            pooled = pooled * relevance.amax(-1, keepdim=True)
            context = torch.zeros_like(context)
        features = torch.cat([context, pooled], -1)
        logits = torch.einsum('bkd,kd->bk', features, self.logit_weights) + self.logit_bias
        return logits, masks.reshape(-1, self.k, 16, 16), foreground.reshape(-1, 16, 16)

    def recognize_global(self, global_features, concept_ids):
        # Used on a real, individually perturbed RGB view, one concept per view.
        z = F.normalize(global_features.float(), dim=-1)
        # The native clean model's recognition mixes CLS and spatial pooling;
        # this helper is only for explicit CLS-only noise variants.
        w = self.logit_weights[concept_ids]
        return (torch.cat([z, z], -1) * w).sum(-1) + self.logit_bias[concept_ids]


class ConditionalRGBNoise(nn.Module):
    """Shared text-conditioned decoder or independent per-concept RGB heads.

    All gates are predicted from image tokens. Ground-truth masks are never
    forward inputs. Noise protection/energy is an explicit architectural prior,
    not an assertion that CE spontaneously discovers foreground segmentation.
    """
    def __init__(self, concepts, independent=False, concept_local=False, mean_budget=0.0):
        super().__init__()
        self.concepts = concepts
        w = concepts.patch.out_features
        k = concepts.k
        self.independent = independent
        self.concept_local = concept_local
        self.mean_budget = float(mean_budget)
        self.register_buffer('_mean_budget', torch.tensor(self.mean_budget))
        self.mean = nn.Linear(w, 3 * k if independent else 3)
        self.std = nn.Linear(w, k if independent else 1)
        self.film = nn.Linear(concepts.texts.shape[1], w * 2)
        nn.init.zeros_(self.mean.weight); nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.std.weight); nn.init.zeros_(self.std.bias)
        if self.mean_budget > 0:
            nn.init.normal_(self.mean.weight, std=.01)

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        key=prefix+'_mean_budget'
        if key not in state_dict:
            state_dict[key]=self._mean_budget.detach().clone()
        self.mean_budget=float(state_dict[key].item())
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def distribution(self, tokens, concept_ids):
        z, h, scores, masks, fg = self.concepts.spatial(tokens)
        batch = torch.arange(len(tokens), device=tokens.device)
        relevance = masks[batch, concept_ids].sigmoid()
        object_gate = fg.sigmoid()
        # Preserve the foreground globally while modulating concept locality.
        gate = object_gate * (.5 + .5 * relevance)
        if self.independent:
            mean = self.mean(h).reshape(len(tokens), 256, self.concepts.k, 3)[batch, :, concept_ids]
            std = self.std(h)[batch, :, concept_ids]
        else:
            gamma, beta = self.film(self.concepts.texts[concept_ids]).chunk(2, -1)
            conditioned = F.gelu(h * (1 + gamma[:, None]) + beta[:, None])
            mean = self.mean(conditioned)
            std = self.std(conditioned).squeeze(-1)
        mean_gate = object_gate * relevance if self.concept_local else gate
        mu = .15 * mean.tanh() * mean_gate[:, :, None]
        if self.mean_budget > 0:
            # Fixed PiNI-inspired RMS within predicted relevant support. An
            # absent/tiny concept does not carry a whole-image energy budget.
            raw = mean.tanh() * mean_gate[:, :, None]
            mass = mean_gate.square().sum(1).clamp_min(1e-6)
            conditional_rms = (raw.square().sum((1,2))/(3*mass)).sqrt().clamp_min(1e-4)
            mu = self.mean_budget * raw / conditional_rms[:,None,None]
        # RGB standard deviations, a protected foreground and a budgeted
        # high-variance background. Report these imposed bounds explicitly.
        sigma = .008 + (1 - object_gate) * (.08 + .12 * std.sigmoid())
        if self.concept_local:
            sigma = sigma + object_gate * (1 - relevance) * .025
            # Global object preservation before concept-specific mean/noise.
            gate = object_gate
        up = lambda v: F.interpolate(v, (224, 224), mode='bilinear', align_corners=False)
        mu = up(mu.transpose(1, 2).reshape(-1, 3, 16, 16))
        sigma = up(sigma.reshape(-1, 1, 16, 16)).expand(-1, 3, -1, -1)
        gate = up(gate.reshape(-1, 1, 16, 16))
        return mu, sigma, gate

    def perturb(self, rgb, tokens, concept_ids, mode='noise', epsilon=None, suppress=False):
        mu, sigma, gate = self.distribution(tokens, concept_ids)
        if mode == 'clean':
            return rgb, mu * 0, sigma * 0, gate
        base = gate * rgb + (1-gate) * .5 if suppress else rgb
        if mode in ['mask', 'mean', 'noise']:
            base = base + (mu if mode in ['mean', 'noise'] else 0)
        if mode == 'noise':
            epsilon = torch.randn_like(rgb) if epsilon is None else epsilon
            base = base + sigma * epsilon
        # No clamp before backbone; display clipping is measured separately.
        return base, mu, sigma, gate
