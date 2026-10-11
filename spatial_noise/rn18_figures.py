"""PNG summaries and real perturbations for the shared full-data ResNet18 runs."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
import numpy as np
import torch
from .cub_aligned import tensor
from .rn18_datasets import Experiment
from .grounding import GroundedNoiseGenerator
from .data import write_json

OUT=Path('reports/main');COLORS=['#357ABD','#56A76C','#A76BC3','#E58B2A']


def read(path):return json.loads(path.read_text())


def summary(e):
    values={'simple':read(e.root/'simple_refit/seed42/paired_results.json')}
    for method in ['clean','mean','noise']:values[method]=read(e.root/'prior_off/seed42/refit'/method/'results.json')['test']
    protocol=read(e.root/'cache/protocol.json')
    values['protocol']={k:protocol[k] for k in ['dataset','backbone','counts']}
    values['protocol'].update(seed=42,concepts=e.k,classes=e.classes,trainval_images=sum(protocol['counts'][s] for s in ['train','val']),
                              final_models='refit on train+val',spatial_noise_prior=False)
    values['noise_minus_mean']={
        'original_accuracy_pp':100*(values['noise']['original']['accuracy']-values['mean']['original']['accuracy']),
        'changed_accuracy_pp':100*(values['noise']['changed']['accuracy']-values['mean']['changed']['accuracy']),
        'concept_drift_pp':100*(values['noise']['concept_drift_mae']-values['mean']['concept_drift_mae'])}
    write_json(e.root/'main_results.json',values)
    fig,axes=plt.subplots(1,2,figsize=(13,5));x=np.arange(4);rows=[values[k] for k in ['simple','clean','mean','noise']]
    for j,view in enumerate(['original','changed']):
        v=[100*r[view]['accuracy'] for r in rows]
        bars=axes[0].bar(x+(j-.5)*.34,v,width=.34,color=COLORS,alpha=1 if j else .42,edgecolor=COLORS,hatch='' if j else '//')
        for bar,n in zip(bars,v):axes[0].text(bar.get_x()+bar.get_width()/2,n+.6,f'{n:.2f}',ha='center',fontsize=9)
    axes[0].set_ylim(0,105);axes[0].set_ylabel('Classification accuracy (%)');axes[0].set_title('Hatched: original | Solid: changed background')
    drift=[100*r['concept_drift_mae'] for r in rows];bars=axes[1].bar(x,drift,color=COLORS,width=.55)
    for bar,n in zip(bars,drift):axes[1].text(bar.get_x()+bar.get_width()/2,n+.08,f'{n:.2f}',ha='center')
    axes[1].set_ylim(0,max(drift)*1.3);axes[1].set_ylabel('Concept probability drift (percentage points)');axes[1].set_title('Same subject, changed background | lower is better')
    for ax in axes:
        ax.set_xticks(x,['Simple\nCBM','Background\ntraining','Region\n+ mean','Region\n+ noise'])
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    delta=values['noise_minus_mean']['changed_accuracy_pp']
    fig.suptitle(f'{e.name.upper()} | ResNet18 | train+val refit | seed 42 | Noise vs mean: {delta:+.2f} pp',fontsize=14)
    fig.tight_layout();OUT.mkdir(parents=True,exist_ok=True);fig.savefig(OUT/f'{e.name}_rn18_main_results.png',dpi=180);plt.close(fig)
    return values


def pipeline(e):
    p=read(e.root/'cache/protocol.json');blue='#DFEDFA';green='#E4F4E3';orange='#FFF0DB'
    fig,ax=plt.subplots(figsize=(16,6));ax.set_xlim(0,16);ax.set_ylim(0,6);ax.axis('off')
    for y,h in [(2.65,2.6),(.15,2.25)]:ax.add_patch(Rectangle((.05,y),15.9,h,fill=False,edgecolor='#222222',linewidth=1.3))
    def box(x,y,w,h,text,color):
        ax.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor='#29475D',linewidth=1.2))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11,fontweight='bold',linespacing=1.5)
    def arrow(a,b,color='#222222',style='-'):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color=color,linewidth=1.4,linestyle=style))
    ax.text(.3,4.8,'(a) One region-conditioned RGB view, one simple CBM',fontsize=17,fontweight='bold')
    boxes=[(.4,2.,'RGB image',orange),(2.85,2.5,'Frozen DINO\nImage features + regions',blue),(5.8,2.3,'Noise generator\nShared mu, sigma',green),
           (8.55,2.3,'RGB perturbation\nx + mu + sigma * eps',orange),(11.3,4.2,f'Frozen ResNet18 -> {e.k} concepts\n-> {e.classes} classes',blue)]
    for x,w,text,color in boxes:box(x,3.28,w,1.0,text,color)
    for a,b in [((2.4,3.78),(2.85,3.78)),((5.35,3.78),(5.8,3.78)),((8.1,3.78),(8.55,3.78)),((10.85,3.78),(11.3,3.78))]:arrow(a,b)
    ax.text(8.,2.88,'Class CE + concept BCE + cached-original consistency | spatial prior OFF',ha='center',fontsize=11)
    ax.text(.3,1.93,'(b) Configuration selection, train+val refit, paired background evaluation',fontsize=16,fontweight='bold')
    c=p['counts'];box(.5,.55,4.3,.95,f'{c["train"]:,} train | {c["val"]:,} validation\nOne fixed configuration, seed 42',blue)
    box(5.55,.55,4.3,.95,f'{c["train"]+c["val"]:,} train + validation\nRefit with fixed configuration',green)
    box(10.6,.55,4.9,.95,f'{c["test"]:,} original test images\nChanged-background accuracy + concept drift',orange)
    arrow((4.8,1.),(5.55,1.));arrow((9.85,1.),(10.6,1.))
    fig.suptitle(f'{e.name.upper()}: Shared Noise for a Simple ResNet18 CBM',fontsize=21,fontweight='bold')
    fig.subplots_adjust(left=.01,right=.99,bottom=.025,top=.88);fig.savefig(OUT/f'{e.name}_rn18_pipeline.png',dpi=180);plt.close(fig)


@torch.no_grad()
def visualization(e,device='cuda'):
    arrays,splits,p=e.load();rng=np.random.default_rng(42);eligible=np.asarray(splits['test'])
    if 'background_eligible' in arrays:eligible=eligible[np.asarray(arrays['background_eligible'][eligible])]
    ids=np.sort(rng.choice(eligible,size=5,replace=False));changed=np.load(e.root/'cache/rgb_changed.npy',mmap_mode='r')
    convert=lambda a:torch.as_tensor(np.array(a[ids]),device=device).float()
    rgb=convert(changed)/255;region={k:convert(np.load(e.root/'regions/changed'/(k+'.npy'),mmap_mode='r')) for k in ['features','maps']}
    boxes=np.load(e.root/'regions/changed/boxes.npy',mmap_mode='r')[ids];valid=np.load(e.root/'regions/changed/valid.npy',mmap_mode='r')[ids]
    outputs={};distributions={}
    for method in ['mean','noise']:
        state=torch.load(e.root/'prior_off/seed42/refit'/method/'best.pt',weights_only=True)
        generator=GroundedNoiseGenerator(regions=len(e.phrases),mean_limit=.03,std_limit=.1,spatial_noise_prior=False).to(device)
        generator.load_state_dict(state['generator']);torch.manual_seed(4242)
        x,mu,sigma=generator(rgb,region,method);outputs[method]=x.clamp(0,1).cpu().numpy().transpose(0,2,3,1)
        distributions[method]=(mu.abs().mean(1).cpu().numpy(),sigma.square()[:,0].cpu().numpy())
    mu,var=distributions['noise'];maxmu=max(float(mu.max()),1e-8);maxvar=max(float(var.max()),1e-8)
    fig,axes=plt.subplots(5,7,figsize=(19,13.5),layout='constrained')
    titles=['Original image','Changed background','DINO regions','Mean only','Mean + noise','Mean magnitude |mu|','Noise variance sigma^2']
    for row,index in enumerate(ids):
        original=np.asarray(arrays['rgb_original'][index]).transpose(1,2,0);background=np.asarray(changed[index]).transpose(1,2,0)
        images=[original,background,background,outputs['mean'][row],outputs['noise'][row],mu[row],var[row]]
        for col,(ax,im) in enumerate(zip(axes[row],images)):
            if col<5:ax.imshow(im)
            else:
                heat=ax.imshow(im,cmap='magma' if col==5 else 'viridis',vmin=0,vmax=maxmu if col==5 else maxvar)
                ax.contour(np.asarray(arrays['foreground'][index]),levels=[.5],colors=['white'],linewidths=.65)
                if row==4:fig.colorbar(heat,ax=axes[:,col],shrink=.65,pad=.025)
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(titles[col],fontsize=11,pad=8)
        parts=[0,1,2] if e.name=='waterbirds' else [0,1,2,4]
        for part,color in zip(parts,['#FDE047','#38BDF8','#FB7185','#4ADE80']):
            if valid[row,part]:
                x1,y1,x2,y2=boxes[row,part]*224;axes[row,2].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=color,linewidth=1.7))
        axes[row,0].set_ylabel(f'Test ID {index}',fontsize=10)
    fig.suptitle(f'Actual learned perturbations | {e.name.upper()} / ResNet18 | prior OFF\nFive randomly sampled test images (seed 42); shared heatmap scale across rows',fontsize=15)
    fig.savefig(OUT/f'{e.name}_rn18_noise_visualization.png',dpi=170);plt.close(fig)
    write_json(e.root/'visualization.json',{'cache_indices':ids.tolist(),'selection_seed':42,'noise_seed':4242,'mean_max':maxmu,'variance_max':maxvar})


def combined():
    roots=[Path('artifacts/cub_aligned_rn18'),Experiment('rival10').root,Experiment('waterbirds').root]
    if not all((r/'main_results.json').exists() for r in roots):return
    results=[read(r/'main_results.json') for r in roots];names=['CUB','RIVAL10','Waterbirds (CUB + Places)']
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,r,name in zip(axes,results,names):
        values=[r['mean']['changed']['accuracy']*100,r['noise']['changed']['accuracy']*100]
        ax.bar([0,1],values,color=COLORS[2:],width=.55)
        for i,v in enumerate(values):ax.text(i,v+.7,f'{v:.2f}%',ha='center',fontsize=13)
        delta=values[1]-values[0];ax.set_title(f'{name}\nNoise vs mean: {delta:+.2f} pp',fontsize=12)
        ax.set_ylim(0,105);ax.set_xticks([0,1],['Mean only','Mean + noise']);ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel('Changed-background classification accuracy (%)')
    fig.suptitle('New shared-view ResNet18 CBM | full data | train+val refit | seed 42 | prior OFF',fontsize=15)
    fig.tight_layout();fig.savefig(OUT/'rn18_three_dataset_results.png',dpi=180);plt.close(fig)
    write_json(Path('artifacts/rn18_three_dataset_results.json'),dict(zip(['cub','rival10','waterbirds'],results)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',required=True,choices=['rival10','waterbirds'])
    parser.add_argument('--summary-only',action='store_true');parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args();torch.set_num_threads(4)
    e=Experiment(args.dataset);summary(e);pipeline(e)
    if not args.summary_only:visualization(e,args.device)
    combined()


if __name__=='__main__':main()
