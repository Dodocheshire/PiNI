"""Main PNG figures for the aligned ResNet18 CUB experiment."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import torch

from .cub_aligned import ROOT, BirdBackbone, SimpleCBM, tensor, load
from .grounding import GroundedNoiseGenerator
from .data import write_json

OUT=Path('reports/main')
COLORS=['#357ABD','#56A76C','#A76BC3','#E58B2A']
LABELS=['Simple CBM','Background training','Region + mean','Region + mean + noise']


def read(path): return json.loads(path.read_text())


def pipeline():
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':12})
    fig,ax=plt.subplots(figsize=(17,10));ax.set_xlim(0,17);ax.set_ylim(0,10);ax.axis('off')
    blue='#DFEDFA';green='#E4F4E3';orange='#FFF0DB';red='#B92929'
    for y,h in [(7.0,2.4),(2.6,4.15),(.1,2.25)]:
        ax.add_patch(Rectangle((.05,y),16.9,h,fill=False,edgecolor='#222222',linewidth=1.3))
    def box(x,y,w,h,text,color):
        ax.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor='#29475D',linewidth=1.25))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=12,fontweight='bold',linespacing=1.45)
    def arrow(a,b,color='#222222',dashed=False):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=16,color=color,
                                    linewidth=1.6,linestyle='--' if dashed else '-'))
    ax.text(.25,9.0,'(a) Establish a simple concept bottleneck',fontsize=18,fontweight='bold')
    box(.4,7.55,2.2,.95,'CUB image\n224 x 224',orange)
    box(3.1,7.5,3.4,1.05,'CUB-trained ResNet18\nFrozen | global pooling',blue)
    box(7.1,7.5,3.4,1.05,'Linear concept head\n512 -> 112 attributes',green)
    box(11.1,7.5,2.55,1.05,'Linear classifier\n112 -> 200',green)
    box(14.25,7.55,2.3,.95,'Bird species',green)
    for a,b in [((2.6,8.03),(3.1,8.03)),((6.5,8.03),(7.1,8.03)),((10.5,8.03),(11.1,8.03)),((13.65,8.03),(14.25,8.03))]:arrow(a,b)
    ax.text(.25,6.33,'(b) Add one region-conditioned RGB perturbation',fontsize=18,fontweight='bold')
    box(.4,4.75,2.1,1.0,'RGB image x',orange)
    box(3.0,4.65,3.0,1.2,'Frozen Grounding DINO\nFeatures + 12 region maps',blue)
    box(6.65,4.65,3.0,1.2,'Small noise generator\nShared mu and sigma\n8,740 parameters',green)
    box(10.3,4.65,2.55,1.2,'Perturbed RGB\nx + mu + sigma * eps',orange)
    box(13.5,4.65,3.0,1.2,'Same simple CBM\nOne ResNet18 encode',blue)
    for a,b in [((2.5,5.25),(3.,5.25)),((6.,5.25),(6.65,5.25)),((9.65,5.25),(10.3,5.25)),((12.85,5.25),(13.5,5.25))]:arrow(a,b)
    ax.plot([1.45,1.45,11.57],[4.7,4.1,4.1],color='#333333',linewidth=1.3)
    arrow((11.57,4.1),(11.57,4.65));ax.text(6.1,4.17,'Preserve the original RGB input',ha='center',fontsize=11)
    ax.text(4.5,5.97,'Fixed part names',ha='center',fontsize=10)
    ax.text(8.5,3.58,'L = classification CE + concept BCE + cached-reference consistency + 0.01 * perturbation energy',ha='center',fontsize=12)
    arrow((15,3.25),(8.15,3.25),red,True);arrow((8.15,3.25),(8.15,4.6),red,True)
    ax.text(12.0,2.92,'All losses reach the noise generator',color=red,ha='center',fontsize=11)
    ax.text(.3,2.96,'Spatial noise prior OFF',fontsize=12,fontweight='bold')
    ax.text(.25,1.95,'(c) Select configuration, then refit on train + validation',fontsize=18,fontweight='bold')
    box(.5,.52,4.6,.95,'4,796 train | 1,198 validation\nChoose configuration',blue)
    box(6.0,.52,4.6,.95,'5,994 train + validation\nRefit with fixed configuration',green)
    box(11.5,.52,4.95,.95,'5,794 test images\nOriginal / changed background / concept drift',orange)
    arrow((5.1,1.),(6.,1.));arrow((10.6,1.),(11.5,1.))
    fig.suptitle('Region-conditioned Noise for a Simple CUB CBM',fontsize=23,fontweight='bold',y=.99)
    fig.subplots_adjust(left=.015,right=.985,bottom=.02,top=.94)
    OUT.mkdir(parents=True,exist_ok=True);fig.savefig(OUT/'cub_rn18_pipeline.png',dpi=180);plt.close(fig)
    plt.rcParams.update({'font.family':'DejaVu Sans'})


def baseline():
    protocol=read(ROOT/'cache/protocol.json')
    before=read(ROOT/'simple/seed42/results.json'); after=read(ROOT/'simple_refit/seed42/results.json')
    values=[protocol['pretrained_classifier_accuracy']['test']*100,before['test']['accuracy']*100,after['test']['accuracy']*100]
    fig,ax=plt.subplots(figsize=(9,4.8));ax.bar(range(3),values,color=[COLORS[0],COLORS[2],COLORS[1]],width=.55)
    for i,v in enumerate(values):ax.text(i,v+1.4,f'{v:.2f}%',ha='center',fontweight='bold',fontsize=14)
    ax.set_xticks(range(3),['CUB-pretrained\nResNet18 classifier','Simple CBM\n4,796 training images','Simple CBM refit\n5,994 training images'])
    ax.set_ylim(0,100);ax.set_ylabel('Test classification accuracy (%)');ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.set_title('CUB-200-2011 | 5,794 test images | 112 official concepts',pad=18)
    fig.tight_layout();fig.savefig(OUT/'cub_rn18_baseline.png',dpi=180);plt.close(fig)


def results():
    base=ROOT/'prior_off/seed42/refit'
    values=[read(ROOT/'simple_refit/seed42/paired_results.json')]+[read(base/m/'results.json')['test'] for m in ['clean','mean','noise']]
    fig,axes=plt.subplots(1,2,figsize=(13,5));x=np.arange(4)
    for i,view in enumerate(['original','changed']):
        v=[100*r[view]['accuracy'] for r in values]
        bars=axes[0].bar(x+(i-.5)*.34,v,width=.34,color=COLORS,alpha=1 if i else .42,edgecolor=COLORS,hatch='' if i else '//')
        for bar,n in zip(bars,v):axes[0].text(bar.get_x()+bar.get_width()/2,n+.5,f'{n:.2f}',ha='center',fontsize=9)
    axes[0].set_xticks(x,['Simple\nCBM','Background\ntraining','Region\n+ mean','Region\n+ noise'])
    axes[0].set_ylim(0,100);axes[0].set_ylabel('Classification accuracy (%)')
    axes[0].set_title('Hatched: original | Solid: changed background')
    drift=[100*r['concept_drift_mae'] for r in values]
    bars=axes[1].bar(x,drift,color=COLORS,width=.55)
    for bar,v in zip(bars,drift):axes[1].text(bar.get_x()+bar.get_width()/2,v+.08,f'{v:.2f}',ha='center')
    axes[1].set_xticks(x,['Simple\nCBM','Background\ntraining','Region\n+ mean','Region\n+ noise'])
    axes[1].set_ylabel('Concept probability drift (percentage points)')
    axes[1].set_title('Same bird, changed background | lower is better');axes[1].set_ylim(0,max(drift)*1.3)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    delta=100*(values[-1]['changed']['accuracy']-values[-2]['changed']['accuracy'])
    fig.suptitle(f'Full CUB | ResNet18 | train+val refit | seed 42 | Noise vs mean: {delta:+.2f} pp',fontsize=14)
    fig.tight_layout();fig.savefig(OUT/'cub_rn18_main_results.png',dpi=180);plt.close(fig)
    summary=dict(zip(['simple','clean','mean','noise'],values))
    summary['protocol']={'dataset':'CUB-200-2011','backbone':'CUB-trained frozen ResNet18',
                         'concepts':112,'trainval_images':5994,'test_images':5794,'seed':42,
                         'final_models':'refit on train+val','spatial_noise_prior':False}
    summary['noise_minus_mean']={
        'original_accuracy_pp':100*(values[-1]['original']['accuracy']-values[-2]['original']['accuracy']),
        'changed_accuracy_pp':delta,
        'concept_drift_pp':100*(values[-1]['concept_drift_mae']-values[-2]['concept_drift_mae'])}
    write_json(ROOT/'main_results.json',summary)


@torch.no_grad()
def visualization():
    arrays,splits,protocol=load();rng=np.random.default_rng(42)
    ids=np.sort(rng.choice(splits['test'],size=5,replace=False))
    changed=np.load(ROOT/'cache/rgb_changed.npy',mmap_mode='r')
    rgb=tensor(changed,ids)/255
    region={k:tensor(np.load(ROOT/'regions/changed'/(k+'.npy'),mmap_mode='r'),ids) for k in ['features','maps']}
    boxes=np.load(ROOT/'regions/changed/boxes.npy',mmap_mode='r')[ids]
    valid=np.load(ROOT/'regions/changed/valid.npy',mmap_mode='r')[ids]
    outputs={}; distributions={}
    for method in ['mean','noise']:
        state=torch.load(ROOT/'prior_off/seed42/refit'/method/'best.pt',weights_only=True)
        generator=GroundedNoiseGenerator(mean_limit=state['mean_limit'],std_limit=state['std_limit'],spatial_noise_prior=False).cuda()
        generator.load_state_dict(state['generator']);torch.manual_seed(4242)
        x,mu,sigma=generator(rgb,region,method)
        outputs[method]=x.clamp(0,1).cpu().numpy().transpose(0,2,3,1)
        distributions[method]=(mu.abs().mean(1).cpu().numpy(),sigma.square()[:,0].cpu().numpy())
    mu,var=distributions['noise'];vmax_mu=max(float(mu.max()),1e-8);vmax_var=max(float(var.max()),1e-8)
    titles=['Original image','Changed background','DINO bird / head / beak','Mean only','Mean + noise','Mean magnitude |mu|','Noise variance sigma^2']
    fig,axes=plt.subplots(len(ids),7,figsize=(19,13.5),layout='constrained')
    colors=['#FDE047','#38BDF8','#FB7185']
    for row,index in enumerate(ids):
        original=np.asarray(arrays['rgb_original'][index]).transpose(1,2,0)
        background=np.asarray(changed[index]).transpose(1,2,0)
        images=[original,background,background,outputs['mean'][row],outputs['noise'][row],mu[row],var[row]]
        for col,(ax,im) in enumerate(zip(axes[row],images)):
            if col<5:ax.imshow(im)
            else:
                heat=ax.imshow(im,cmap='magma' if col==5 else 'viridis',vmin=0,vmax=vmax_mu if col==5 else vmax_var)
                ax.contour(np.asarray(arrays['foreground'][index]),levels=[.5],colors=['white'],linewidths=.65)
                if row==len(ids)-1:fig.colorbar(heat,ax=axes[:,col],shrink=.65,pad=.025)
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(titles[col],fontsize=11,pad=8)
        for part,color in zip([0,1,2],colors):
            if valid[row,part]:
                x1,y1,x2,y2=boxes[row,part]*224
                axes[row,2].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=color,linewidth=1.8))
        axes[row,0].set_ylabel(f'Test ID {index}',fontsize=10)
    fig.suptitle('Actual learned perturbations | full CUB / ResNet18 | prior OFF\nFive randomly sampled test images (seed 42); shared heatmap scale across rows',fontsize=15)
    fig.savefig(OUT/'cub_rn18_noise_visualization.png',dpi=170);plt.close(fig)
    write_json(ROOT/'visualization.json',{'cache_indices':ids.tolist(),'selection_seed':42,'noise_seed':4242,'mean_heatmap_max':vmax_mu,'variance_heatmap_max':vmax_var})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--baseline-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);pipeline();baseline()
    if not args.baseline_only:results();visualization()


if __name__=='__main__':main()
