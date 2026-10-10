"""Draw the spatial CBM pipeline in the supplied paper-figure style."""
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.path import Path as MplPath
import numpy as np
import torch

from .models import ConditionalRGBNoise, SpatialConcepts


BLUE = ('#DBEDFF', '#174A78')
GREEN = ('#E4F6E8', '#27613B')
RED = ('#FFE4E4', '#C62727')


def examples():
    torch.set_num_threads(2)
    source=Path('artifacts/region_noise/figure4_tensors.npz')
    with np.load(source) as data:
        row=2;sample_id=int(data['ids'][row])
        image=data['original'][row].copy()
        sampled=data['sample'][row].copy()
        mu=np.sqrt(np.mean(data['mu'][row]**2,axis=0))
        variance=np.mean(data['sigma'][row]**2,axis=0)
    tokens=np.load('artifacts/rival_vitl14/tokens.npy',mmap_mode='r')[sample_id]
    tokens=torch.tensor(np.array(tokens),dtype=torch.float32).unsqueeze(0)
    pretrained=torch.load('artifacts/spatial_region/best.pt',map_location='cpu',weights_only=False)
    region=SpatialConcepts(pretrained['model']['texts'])
    region.load_state_dict(pretrained['model']);region.eval()
    checkpoint=torch.load('artifacts/region_noise/best.pt',map_location='cpu',weights_only=False)
    config=checkpoint['args']
    noise=ConditionalRGBNoise(SpatialConcepts(checkpoint['model']['concepts.texts']),
                             config['independent_heads'],config.get('concept_local',False))
    noise.load_state_dict(checkpoint['model']);noise.eval()
    with torch.no_grad():
        _,mask,foreground=region(tokens)
        views,_,_,_=noise.perturb(torch.tensor(image)[None].expand(18,-1,-1,-1),
            tokens.expand(18,-1,-1),torch.arange(18),mode='mean',suppress=config['suppress'])
    return dict(image=image,sampled=sampled,mu=mu,variance=variance,
        foreground=foreground[0].sigmoid().numpy(),region=mask[0,6].sigmoid().numpy(),
        views=views.numpy(),sample_id=sample_id,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())


def main():
    data=examples();out=Path('reports/main');out.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'font.family':'DejaVu Serif','mathtext.fontset':'stix',
                         'svg.fonttype':'path','pdf.fonttype':42}):
        fig=plt.figure(figsize=(16,11.9),facecolor='white')
        ax=fig.add_axes([.012,.012,.976,.976])
        ax.set_xlim(0,160);ax.set_ylim(0,119);ax.set_aspect('equal');ax.axis('off')

        def text(x,y,value,size=16,bold=False,color='black',ha='center'):
            return ax.text(x,y,value,fontsize=size,fontweight='bold' if bold else 'normal',
                           ha=ha,va='center',color=color,zorder=6,linespacing=1.12)

        def box(x,y,w,h,value=None,theme=BLUE,size=16):
            ax.add_patch(Rectangle((x,y),w,h,facecolor=theme[0],edgecolor=theme[1],lw=1.3,zorder=2))
            if value:text(x+w/2,y+h/2,value,size,True)

        def arrow(a,b,color='black',dashed=False,rad=0):
            ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=17,
                linewidth=1.45,color=color,linestyle=(0,(5,3)) if dashed else '-',
                connectionstyle=f'arc3,rad={rad}',shrinkA=0,shrinkB=0,zorder=4))

        def routed_arrow(points,color='black',dashed=False):
            codes=[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1)
            ax.add_patch(FancyArrowPatch(path=MplPath(points,codes),arrowstyle='-|>',
                mutation_scale=17,lw=1.4,color=color,linestyle=(0,(5,3)) if dashed else '-',zorder=4))

        def tile(array,x,y,w,h,cmap=None,vmax=None):
            image=array.transpose(1,2,0) if array.ndim==3 else array
            ax.imshow(np.clip(image,0,1) if cmap is None else image,extent=(x,x+w,y,y+h),
                      cmap=cmap,vmin=0 if cmap else None,vmax=vmax,zorder=3,aspect='equal')
            ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='#333333',lw=.8,zorder=4))

        def vector(x,y,w,h,items,theme=BLUE):
            n=len(items);inside=w-2.4;rh=h/n
            for i,value in enumerate(items):
                yy=y+h-(i+1)*rh
                ax.add_patch(Rectangle((x+1.2,yy),inside,rh,facecolor=theme[0],edgecolor='white',lw=1,zorder=2))
                text(x+w/2,yy+rh/2,value,17)
            for side,sign in [(x,1),(x+w,-1)]:
                path=MplPath([(side+sign*.65,y),(side,y+.6),(side,y+h-.6),(side+sign*.65,y+h)],
                             [MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4])
                ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-',lw=1.5,zorder=4))

        for bottom,top in [(77,109),(36,75),(1,34)]:
            ax.add_patch(Rectangle((1,bottom),158,top-bottom,fill=False,edgecolor='black',lw=1.1,zorder=1))
        text(80,115,'Spatial Noise CBM: Concept-Guided RGB Perturbation',27,True)

        # (a) Pretrain native concepts and the spatial readout.
        text(3,104.5,'(a) Learn Concepts and Foreground Regions',22,True,ha='left')
        tile(data['image'],4,87,11,11);text(9.5,100,'RGB image',12)
        arrow((15,92.5),(21,92.5))
        box(21,87,23,11,'Frozen CLIP\nImage Encoder\nViT-L/14',size=16)
        arrow((44,92.5),(49,92.5))
        vector(49,87,9,11,[r'$h_1$',r'$\cdots$',r'$h_P$'])
        text(53.5,100,r'Image tokens $H$',13)
        arrow((58,92.5),(65,92.5))
        box(65,86,27,13,'Learnable Concept\n& Region Predictor',size=16)
        box(4,78.5,20,5.8,'18 Concept Names',GREEN,13)
        arrow((24,81.4),(30,81.4))
        box(30,78.5,25,5.8,'Frozen CLIP Text Encoder',size=12)
        arrow((55,81.4),(60,81.4));text(61,81.4,r'$T$',17)
        routed_arrow([(63,81.4),(78.5,81.4),(78.5,86)])
        arrow((92,92.5),(98,92.5))
        box(98,86,24,13,theme=GREEN)
        tile(data['foreground'],99.5,88,9,9,cmap='Greens',vmax=1)
        tile(data['region'],111,88,9,9,cmap='Greens',vmax=1)
        text(104,100,r'Foreground $G$',12);text(115.5,100,r'Region $R_k$',12)
        arrow((122,92.5),(130,92.5))
        vector(130,87,11,11,[r'$c_1$',r'$\cdots$',r'$c_{18}$'],GREEN)
        text(146,91,'Concept\nScores',14,True)
        text(122,81,r'$L_{\mathrm{concept}}+\lambda L_{\mathrm{region}}$',20,color=RED[1])
        arrow((111,81.5),(91,86),RED[1],True,rad=-.2)

        # (b) Real RGB noise; native concept losses update the generator/readout.
        text(3,70.5,'(b) Train Concept-Conditional Noise in RGB Pixels',22,True,ha='left')
        box(4,52,24,13,'Image Tokens H\nConcept Text '+r'$t_k$'+'\nRegions '+r'$G,\ R_k$',size=15)
        arrow((28,58.5),(33,58.5))
        box(33,52,28,13,'Concept-Conditional\nNoise Generator',RED,15)
        arrow((61,58.5),(67,58.5))
        box(67,52,24,13,theme=RED)
        tile(data['mu'],68.4,53.5,9.5,9.5,cmap='magma',vmax=float(data['mu'].max()))
        tile(data['variance'],79.9,53.5,9.5,9.5,cmap='viridis',vmax=float(data['variance'].max()))
        text(73.2,66.3,r'Mean $\mu_k$',13);text(84.7,66.3,r'Variance $\sigma_k^2$',13)
        arrow((91,58.5),(98,58.5))
        tile(data['sampled'],98,52,13,13);text(104.5,66.3,r'Noisy RGB $\widetilde{x}_k$',13)
        arrow((111,58.5),(118,58.5))
        box(118,52,20,13,'Frozen CLIP\nViT-L/14',size=15)
        arrow((138,58.5),(144,58.5))
        box(144,52,12,13,'Concept\nReadout\n'+r'$c_k$',size=14)
        tile(data['image'],4,38.5,11,11);text(17.5,41,'RGB\n'+r'$x$',13)
        routed_arrow([(22,46.7),(104.5,46.7),(104.5,52)])
        text(62,49.3,r'$x_g=G\odot x+(1-G)\cdot 0.5$',17)
        text(67,42,r'$\widetilde{x}_k=x_g+\mu_k+\sigma_k\odot\xi,\quad \xi\sim\mathcal{N}(0,I)$',18)
        text(132,44.6,'Concept + Region Losses',14,True,RED[1])
        arrow((146,46.3),(149,51.8),RED[1],True)
        path=MplPath([(119,44.1),(119,38.3),(110,38.3),(103,38.3),
                      (80,38.3),(58,38.3),(45,38.3),
                      (36,38.3),(36,43),(36,52)],
                     [MplPath.MOVETO]+[MplPath.CURVE4]*9)
        ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=17,
                                    lw=1.5,color=RED[1],linestyle=(0,(5,3)),zorder=4))
        text(132,39.8,r'$\nabla L$',16,color=RED[1])

        # (c) Deterministic mean views feed a strict 18-concept class head.
        text(3,29.5,'(c) Mean Inference through the Concept Bottleneck',22,True,ha='left')
        tile(data['image'],4,12,11,11);text(9.5,9,'RGB image x',12)
        arrow((15,17.5),(22,17.5))
        box(22,11.5,27,13,'Learned Regions\n+ Mean Generator\n'+r'$G,\ \mu_k$',size=15)
        text(35.5,25.9,'Reuse (a) + (b)',12)
        arrow((49,17.5),(56,17.5))
        for offset,k in [(2,7),(1,8),(0,6)]:
            tile(data['views'][k],56+offset*.9,11.5+offset*.9,12,12)
        text(63,26,'18 Mean RGB Views',13)
        arrow((70.2,17.5),(77,17.5))
        box(77,11.5,27,13,'Frozen CLIP\n+ Concept Readout',size=15)
        arrow((104,17.5),(109,17.5))
        vector(109,11.5,10,13,[r'$c_1$',r'$\cdots$',r'$c_{18}$'])
        text(114,26,'18 Concepts',13)
        arrow((119,17.5),(124,17.5))
        box(124,11.5,20,13,'Sparse Linear\nClassifier',GREEN,15)
        text(134,8,r'$W\in\mathbb{R}^{10\times18}$',16)
        arrow((144,17.5),(149,17.5))
        box(149,12,8,11,'Cat',GREEN,17)
        text(152.5,25.9,'Prediction',12)
        text(66,5.7,r'$\widehat{x}_k=G\odot x+(1-G)\cdot 0.5+\mu_k$',19)

        for extension in ['png','svg','pdf']:
            fig.savefig(out/f'pipeline.{extension}',dpi=200,facecolor='white')
        plt.close(fig)
    metadata=dict(reference='/home/pengcheng/Reference/Style/image.png',
        reference_sha256=hashlib.sha256(Path('/home/pengcheng/Reference/Style/image.png').read_bytes()).hexdigest(),
        image_id=data['sample_id'],example_tensor_sha256=data['source_sha256'],
        stages=['Learn concepts and foreground regions','Train concept-conditional RGB noise',
                'Mean inference through 18 native concepts and sparse linear classifier'])
    print('Created reports/main/pipeline.png, .svg, .pdf',flush=True)
    return metadata


if __name__=='__main__':
    main()
