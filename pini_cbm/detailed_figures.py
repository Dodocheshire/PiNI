"""Detailed, implementation-matched paper figures and measured tensor shapes."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.path import Path as PlotPath
import numpy as np
import torch
from torch.nn import functional as F

from .models import SpatialConcepts, ConditionalRGBNoise


OUT = Path('reports/main')
BLUE = ('#DBEDFF', '#174A78')
GREEN = ('#E4F6E8', '#27613B')
RED = ('#FFE4E4', '#C62727')
GRAY = ('#F3F3F3', '#555555')


class Paper:
    def __init__(self, title, height=116):
        self.fig = plt.figure(figsize=(20, height/10), facecolor='white')
        self.ax = self.fig.add_axes([.012,.012,.976,.976])
        self.ax.set(xlim=(0,200), ylim=(0,height)); self.ax.axis('off')
        self.text(100,height-4,title,24,bold=True)

    def text(self,x,y,value,size=13,bold=False,color='black',ha='center'):
        return self.ax.text(x,y,value,fontsize=size,ha=ha,va='center',
            fontweight='bold' if bold else 'normal',color=color,linespacing=1.2,zorder=5)

    def frame(self,bottom,top,title):
        self.ax.add_patch(Rectangle((1,bottom),198,top-bottom,fill=False,lw=1.15))
        self.text(3,top-3.7,title,18,True,ha='left')

    def box(self,x,y,w,h,label,shape='',theme=BLUE,size=13):
        self.ax.add_patch(Rectangle((x,y),w,h,facecolor=theme[0],edgecolor=theme[1],lw=1.3,zorder=2))
        if shape:
            self.text(x+w/2,y+h*.65,label,size,True)
            self.text(x+w/2,y+h*.23,shape,11,color=theme[1])
        else:self.text(x+w/2,y+h/2,label,size,True)

    def arrow(self,points,red=False):
        color=RED[1] if red else 'black'
        path=PlotPath(points,[PlotPath.MOVETO]+[PlotPath.LINETO]*(len(points)-1))
        self.ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=14,
            lw=1.35,color=color,linestyle=(0,(5,3)) if red else '-',zorder=3))

    def save(self,name):
        for extension in ['png','svg','pdf']:
            self.fig.savefig(OUT/f'{name}.{extension}',dpi=180,facecolor='white')
        plt.close(self.fig)


def measured_shapes():
    torch.set_num_threads(2)
    ck=torch.load('artifacts/region_noise/best.pt',map_location='cpu',weights_only=False)
    cfg=ck['args'];state=ck['model']
    model=ConditionalRGBNoise(SpatialConcepts(state['concepts.texts']),cfg['independent_heads'],cfg['concept_local'])
    model.load_state_dict(state);model.eval()
    cached=np.load('artifacts/rival_vitl14/tokens.npy',mmap_mode='r')
    tokens=torch.tensor(np.array(cached[:2]),dtype=torch.float32)
    rgb=torch.tensor(np.array(np.load('artifacts/rival_vitl14/rgb.npy',mmap_mode='r')[:2]),dtype=torch.float32)/255
    k=torch.tensor([2,6]);record={}
    def add(name,value):record[name]=list(value.shape)
    with torch.no_grad():
        z,h,scores,masks,fg=model.concepts.spatial(tokens)
        q=F.normalize(model.concepts.query(model.concepts.texts),dim=-1)
        r=masks.sigmoid()*fg.sigmoid()[:,None]
        alpha=r/r.sum(-1,keepdim=True).clamp_min(1e-6)
        pooled=torch.einsum('bkp,bpd->bkd',alpha,z[:,1:])*r.amax(-1,keepdim=True)
        gamma,beta=model.film(model.concepts.texts[k]).chunk(2,-1)
        conditioned=F.gelu(h*(1+gamma[:,None])+beta[:,None])
        mu,sigma,gate=model.distribution(tokens,k)
        perturbed,_,_,_=model.perturb(rgb,tokens,k,mode='noise',suppress=True)
        logits,region,foreground=model.concepts(tokens)
        for name,value in [('RGB',rgb),('Z',z),('H',h),('T',model.concepts.texts),('Q',q),
            ('scores',scores),('region_logits',region),('foreground_logits',foreground),
            ('relevance',r),('attention',alpha),('pooled',pooled),('concept_logits',logits),
            ('gamma',gamma),('beta',beta),('conditioned_H',conditioned),
            ('mean_head',model.mean(conditioned)),('std_head',model.std(conditioned)),
            ('mu_RGB',mu),('sigma_RGB',sigma),('foreground_gate_RGB',gate),('noisy_RGB',perturbed)]:add(name,value)
        readout=torch.cat([torch.zeros_like(pooled),pooled],dim=-1)
        direct=(readout*model.concepts.logit_weights[None]).sum(-1)+model.concepts.logit_bias
        assert torch.allclose(logits,direct,atol=1e-6)
        assert list(mu.shape)==[2,3,224,224] and list(logits.shape)==[2,18]
        assert model.concepts.region_readout and not model.fixed_foreground
    record.update(batch_size=2,source_checkpoint='artifacts/region_noise/best.pt',
        source_sha256=hashlib.sha256(Path('pini_cbm/models.py').read_bytes()).hexdigest(),
        width=128,patch_grid=[16,16],concepts=18,classes=10,
        mean_budget=cfg['mean_budget'],region_readout=True,fixed_foreground=False,
        stage2_parameters='noise generator and concept/region predictor updated together',
        class_head_shape=[10,18],readout_identity_verified=True)
    (OUT/'tensor_shapes.json').write_text(json.dumps(record,indent=2))
    return record


def overview():
    p=Paper('Spatial Noise CBM: Detailed Training and Mean Inference',128)
    p.frame(85,117,'(1) Concept / region pretraining: 18,987 images, 40 epochs')
    p.box(4,98,19,10,'RGB image','B x 3 x 224 x 224',GRAY,12)
    p.box(29,98,25,10,'Frozen CLIP ViT-L/14','257 output tokens',size=12)
    p.box(60,98,23,10,'Normalize tokens Z','B x 257 x 768',size=12)
    p.box(90,98,36,10,'Concept & Region Predictor','hidden H: B x 256 x 128',size=13)
    p.box(134,98,26,10,'Region outputs','G: B x 1 x 16 x 16\nR: B x 18 x 16 x 16',GREEN,12)
    p.box(170,98,25,10,'Concept probabilities','B x 18',GREEN,11)
    for a,b in [(23,29),(54,60),(83,90),(126,134),(160,170)]:p.arrow([(a,103),(b,103)])
    p.box(4,87,35,7,'18 names -> Frozen text encoder','T: 18 x 768',size=11)
    p.arrow([(39,90.5),(108,90.5),(108,98)])
    p.text(151,89.5,'Concept BCE + 0.25 Region BCE + 0.5 Dice',12,color=RED[1])
    p.arrow([(130,89.5),(120,89.5),(120,98)],red=True)

    p.frame(44,83,'(2) Noise adaptation: 6,003 images, 6 epochs; one random concept k per image')
    p.box(4,60,29,12,'RGB x; tokens Z, H\nConcept text t_k\nForeground G, region R_k','t_k: B x 768',size=11)
    p.box(40,60,32,12,'Conditional Noise Generator','mu, sigma: B x 3 x 224 x 224',RED,12)
    p.box(80,60,32,12,'Sample a real RGB view','x_tilde_k: B x 3 x 224 x 224',RED,12)
    p.box(121,60,26,12,'Frozen CLIP ViT-L/14','B x 257 x 768',size=12)
    p.box(156,60,39,12,'Region-based Concept Readout','all logits: B x 18 -> select k: B',size=12)
    for a,b in [(33,40),(72,80),(112,121),(147,156)]:p.arrow([(a,66),(b,66)])
    p.text(84,54,r'$x_g=G_{224}\odot x+(1-G_{224})\cdot0.5;\quad\widetilde{x}_k=x_g+\mu_k+\sigma_k\odot\xi$',15)
    p.text(86,47.5,'View concept BCE + clean concept BCE / region BCE / Dice + mean penalty',12,color=RED[1])
    p.arrow([(53,48),(53,60)],red=True);p.arrow([(180,49),(180,60)],red=True)

    p.frame(1,42,'(3) Sparse head fitting, then mean inference: all 18 concepts are evaluated')
    p.box(4,18,24,12,'RGB -> frozen CLIP','Z: B x 257 x 768',size=12)
    p.box(34,18,32,12,'Regions + Mean Generator','enumerate k = 1 ... 18',size=12)
    p.box(72,18,32,12,'18 Mean RGB Views','B x 18 x 3 x 224 x 224',RED,12)
    p.box(111,18,34,12,'CLIP + Concept Readout','(18B) x 18 -> select diagonal',size=11)
    p.box(151,18,19,12,'Concepts c','B x 18',GREEN,12)
    p.box(176,18,19,12,'Sparse Head','B x 10 logits',GREEN,12)
    for a,b in [(28,34),(66,72),(104,111),(145,151),(170,176)]:p.arrow([(a,24),(b,24)])
    p.text(87,11.5,r'$\widehat{x}_k=x_g+\mu_k;\quad c_k=\operatorname{sigmoid}(\ell_k(\widehat{x}_k))$',16)
    p.text(85,5.5,'Head fitting uses sampled training concepts; inference reuses the fitted head with mean-view concepts.',11)
    p.text(174,9.5,r'$W_y:10\times18$',14,color=GREEN[1])
    p.save('pipeline_detailed')


def concept_region():
    p=Paper('Inside the Learnable Concept & Region Predictor',144)
    p.frame(96,133,'(a) Build image patch features and text queries')
    p.box(4,112,25,12,'Normalized CLIP\ntokens Z','B x 257 x 768',GRAY,12)
    p.box(38,115,27,10,'Patch Linear: 768 -> 128','Z[:,1:] -> B x 256 x 128',size=11)
    p.box(38,99,27,10,'Context Linear: 768 -> 128','Z[:,0] -> B x 1 x 128',size=11)
    p.box(76,110,31,13,'Broadcast sum + GELU','H: B x 256 x 128',size=12)
    p.arrow([(29,120),(38,120)]);p.arrow([(29,116),(33,116),(33,104),(38,104)])
    p.arrow([(65,120),(76,118)]);p.arrow([(65,104),(70,104),(70,113),(76,113)])
    p.box(119,108,29,13,'Text embeddings T','18 x 768',GRAY,12)
    p.box(160,108,35,13,'Query Linear + L2 norm','768 -> 128; Q: 18 x 128',size=12)
    p.arrow([(148,114.5),(160,114.5)])
    p.text(97,101,r'$H=\operatorname{GELU}(W_pZ_{patch}+W_cZ_{CLS})$',15)

    p.frame(47,94,'(b) Predict foreground G and all 18 concept regions R')
    p.box(4,69,24,13,'H and Q','B x 256 x 128\n18 x 128',GRAY,12)
    p.box(37,69,31,13,'Scaled cosine matching','S: B x 18 x 256',size=12)
    p.box(80,69,30,13,'Add CLS-based mask bias','Linear 768 -> 18',size=11)
    p.box(122,69,32,13,'Sigmoid + reshape','R: B x 18 x 16 x 16',GREEN,12)
    p.box(166,69,29,13,'Foreground MLP','128 -> 128 -> 1\nG: B x 1 x 16 x 16',GREEN,11)
    for a,b in [(28,37),(68,80),(110,122)]:p.arrow([(a,75.5),(b,75.5)])
    p.arrow([(16,69),(16,61),(180.5,61),(180.5,69)])
    p.text(75,55,r'$S_{bkp}=s\,\langle\operatorname{norm}(H_{bp}),Q_k\rangle;\quad R=\operatorname{sigmoid}(S+W_mZ_{CLS})$',14)
    p.text(159,49.8,'Foreground: Linear -> GELU -> Linear -> sigmoid',11)

    p.frame(1,45,'(c) Use the predicted regions to read native concept probabilities')
    p.box(4,21,24,12,'Region x Foreground','r = R * G\nB x 18 x 256',GREEN,12)
    p.box(36,21,27,12,'Normalize over patches','alpha: B x 18 x 256',size=11)
    p.box(72,21,32,12,'Pool original patch tokens','p_k: B x 18 x 768',size=11)
    p.box(114,21,32,12,'Scale by max relevance','p_k * max_p(r_kp)',size=11)
    p.box(155,21,40,12,'Per-concept Linear + sigmoid','[0; p_k]: B x 18 x 1536\nc: B x 18',GREEN,12)
    for a,b in [(28,36),(63,72),(104,114),(146,155)]:p.arrow([(a,27),(b,27)])
    p.text(51,13,r'$\alpha_{bkp}=r_{bkp}/\sum_p r_{bkp}$',16)
    p.text(119,13,r'$p_{bk}=\sum_p\alpha_{bkp}Z_{bp}$',16)
    p.text(101,5.5,'Readout weights: 18 x 1536; the first 768 inputs are zero in the region-readout model.',11)
    p.save('concept_region_detail')


def noise_generator():
    p=Paper('Inside the Concept-Conditional RGB Noise Generator',155)
    p.frame(101,144,'(a) Condition the patch features on the selected concept text')
    p.box(4,122,23,12,'Concept t_k','B x 768',GRAY,12)
    p.box(35,122,30,12,'FiLM Linear: 768 -> 256','split gamma, beta: B x 128',RED,11)
    p.box(76,122,34,12,'Feature modulation + GELU','H_k: B x 256 x 128',RED,12)
    p.box(122,126,32,10,'Mean Linear: 128 -> 3','M: B x 256 x 3',RED,12)
    p.box(122,108,32,10,'Std Linear: 128 -> 1','V: B x 256 x 1',RED,12)
    p.box(166,118,29,13,'Raw distribution heads','RGB mean + scalar std\nper 16 x 16 cell',RED,11)
    p.arrow([(27,128),(35,128)]);p.arrow([(65,128),(76,128)])
    p.arrow([(110,128),(116,128),(116,131),(122,131)])
    p.arrow([(110,125),(116,125),(116,113),(122,113)])
    p.arrow([(154,131),(160,131),(160,125),(166,125)])
    p.arrow([(154,113),(160,113),(160,122),(166,122)])
    p.text(43,111,'Image H: B x 256 x 128',12)
    p.arrow([(60,113),(93,113),(93,122)])
    p.text(98,104.5,r'$H_k=\operatorname{GELU}(H\odot(1+\gamma_k)+\beta_k)$',16)

    p.frame(48,99,'(b) Put mean shift on concept support; protect foreground from high variance')
    p.box(4,72,27,14,'Predicted gates','g: B x 256\nr_k: B x 256',GREEN,12)
    p.box(41,75,43,14,'Local mean + fixed support RMS','u = tanh(M) * (g * r_k)\nmu_16: B x 3 x 16 x 16',RED,12)
    p.box(99,75,43,14,'Background-weighted std','sigma_16: B x 1 x 16 x 16',RED,12)
    p.box(158,73,37,16,'Bilinear upsample 16 -> 224','mu: B x 3 x 224 x 224\nsigma: B x 3 x 224 x 224\nG_224: B x 1 x 224 x 224',RED,11)
    p.arrow([(31,82),(41,82)]);p.arrow([(31,77),(35,77),(35,69),(120.5,69),(120.5,75)])
    p.arrow([(84,83),(91,83),(91,91.5),(150,91.5),(150,83),(158,83)])
    p.arrow([(142,78),(158,78)])
    p.text(70,64,r'$\mu_{16}=b_\mu u/\sqrt{\sum u^2/(3\sum(g r_k)^2)};\quad b_\mu=0.02686$',15)
    p.text(108,54.5,r'$\sigma_{16}=0.008+(1-g)(0.08+0.12\operatorname{sigmoid}(V))+0.025g(1-r_k)$',16)

    p.frame(1,46,'(c) Sample RGB in training; keep only the learned mean during inference')
    p.box(4,20,29,13,'Original RGB + G_224','x: B x 3 x 224 x 224',GRAY,12)
    p.box(43,20,37,13,'Background suppression','x_g = G_224*x + (1-G_224)*0.5',GREEN,11)
    p.box(92,20,43,13,'Training: add mean + random noise','x_tilde_k = x_g + mu_k + sigma_k*xi',RED,11)
    p.box(150,20,45,13,'Inference: add learned mean','x_hat_k = x_g + mu_k',GREEN,12)
    p.arrow([(33,26.5),(43,26.5)]);p.arrow([(80,26.5),(92,26.5)])
    p.arrow([(80,23),(85,23),(85,13),(172,13),(172,20)])
    p.text(101,7,r'$\xi\sim\mathcal{N}(0,I),\quad\xi,\widetilde{x}_k,\widehat{x}_k\in\mathbb{R}^{B\times3\times224\times224}$',16)
    p.save('noise_generator_detail')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    measured=measured_shapes()
    with plt.rc_context({'font.family':'DejaVu Serif','mathtext.fontset':'stix',
                         'svg.fonttype':'path','pdf.fonttype':42}):
        overview();concept_region();noise_generator()
    print('Created detailed pipeline, concept/region module and noise module figures; tensor shapes checked.',flush=True)
    return measured


if __name__=='__main__':main()
