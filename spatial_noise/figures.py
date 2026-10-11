"""PNG figures from actual fitted distributions and saved test predictions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from clip import clip
from .data import ROOT,load
from .model import SharedSpatialCBM

COLORS={'linear_probe':'#2878b5','salf':'#38a169','mean':'#805ad5','noise':'#e77825'}
LABELS={'linear_probe':'Linear probe','salf':'SALF-CBM (2025)','mean':'Shared mean','noise':'Shared mean + noise'}
OUT=Path('reports/main')


def grounding_pipeline():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig,ax=plt.subplots(figsize=(16,5))
    ax.set_xlim(0,16);ax.set_ylim(0,5);ax.axis('off')
    def box(x,y,w,h,title,body,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.04',
                                   linewidth=1.3,edgecolor='#334155',facecolor=color))
        ax.text(x+w/2,y+h*.72,title,ha='center',va='center',fontsize=12,fontweight='bold')
        ax.text(x+w/2,y+h*.34,body,ha='center',va='center',fontsize=10,linespacing=1.4)
    def arrow(a,b,color='#334155',style='-'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,linewidth=1.6,
                                    color=color,linestyle=style))
    box(.2,1.8,1.5,1.2,'RGB image','Original colors','#f1f5f9')
    box(2.2,1.6,2.5,1.6,'Grounding DINO','Frozen\nBoxes + image features','#dceaf8')
    box(5.2,1.6,2.2,1.6,'Noise generator','Trainable, 8,740 params\nOne shared mu, sigma','#e0f0dc')
    box(7.9,1.8,1.9,1.2,'RGB view','x + mu + sigma * eps','#fff0d2')
    box(10.3,1.6,2.1,1.6,'CLIP ViT-B/16','Frozen\nOne image encode','#dceaf8')
    box(12.9,1.6,2.8,1.6,'Concept readout','Box pooling + learned scores\n112 CUB attributes','#e0f0dc')
    ax.text(.2,3.7,'Fixed part names\nbird, beak, wing, ...',fontsize=10,ha='left',va='center')
    arrow((1.75,3.6),(2.7,3.25))
    for a,b in [((1.7,2.4),(2.2,2.4)),((4.7,2.4),(5.2,2.4)),((7.4,2.4),(7.9,2.4)),
                ((9.8,2.4),(10.3,2.4)),((12.4,2.4),(12.9,2.4))]:arrow(a,b)
    ax.plot([3.45,3.45,14.3],[3.25,4.0,4.0],color='#334155',linewidth=1.4)
    arrow((14.3,4.0),(14.3,3.25))
    ax.text(9,4.12,'Reuse the same detected boxes for concept pooling',ha='center',fontsize=10)
    ax.plot([.95,.95,8.85],[1.75,1.1,1.1],color='#64748b',linewidth=1.2)
    arrow((8.85,1.1),(8.85,1.75),'#64748b')
    ax.text(3.25,1.2,'Original RGB pixels',fontsize=10,color='#64748b')
    arrow((14.3,.68),(6.3,.68),'#c0392b')
    ax.text(10.3,.3,'Concept loss backpropagates through CLIP to the noise generator',ha='center',fontsize=10,color='#c0392b')
    ax.set_title('Grounded shared-view concept model | spatial noise prior OFF by default',fontsize=16,pad=10)
    fig.tight_layout()
    destination=OUT/'grounded_shared_pipeline.png'
    fig.savefig(destination,dpi=180,bbox_inches='tight');plt.close(fig)
    return destination


def white_control_figure():
    from .data import write_json
    names=['cub','waterbirds'];summary={}
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,name in zip(axes,names):
        summary[name]={}
        for i,method in enumerate(['mean','noise']):
            path=ROOT/name/'v4_prior_off_white'/'seed42'/method/'results.json'
            result=json.loads(path.read_text())
            value=100*result['white']['accuracy']
            summary[name][method]=result['white']
            ax.bar(i,value,color=COLORS[method],width=.55)
            ax.text(i,value+1,f'{value:.2f}%',ha='center',fontsize=12)
        ax.set_title(name.upper() if name=='cub' else 'Waterbirds')
        ax.set_xticks([0,1],['Mean only','Mean + noise'])
        ax.set_ylim(0,100);ax.set_ylabel('Classification accuracy (%)')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('White background for both training and testing | prior OFF | seed 42')
    fig.tight_layout()
    destination=OUT/'shared_white_background_control.png'
    fig.savefig(destination,dpi=180,bbox_inches='tight');plt.close(fig)
    write_json(ROOT/'v4_prior_off_white_summary.json',summary)
    return destination


def grounding_quality_figure():
    data=json.loads((ROOT/'grounding_cub_boxes_v1/results.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(16,5),gridspec_kw={'width_ratios':[1,3]})
    values=[100*data['original']['bird_box_mean_iou'],100*data['reversed']['bird_box_mean_iou'],
            100*data['paired_box_iou_missing_is_zero']['bird']]
    axes[0].bar(range(3),values,color=['#2878b5','#e77825','#38a169'],width=.65)
    for i,v in enumerate(values):axes[0].text(i,v+1,f'{v:.1f}',ha='center')
    axes[0].set_xticks(range(3),['Original\nvs. native','Changed\nvs. native','Original\nvs. changed'])
    axes[0].set_title('Whole-bird box overlap (IoU %)')
    parts=list(data['original']['parts'])
    x=np.arange(len(parts))
    for i,(condition,label,color) in enumerate([('original','Original','#2878b5'),('reversed','Changed background','#e77825')]):
        values=[100*data[condition]['parts'][part]['localized_point_recall_area_le_half_bird'] for part in parts]
        axes[1].bar(x+(i-.5)*.36,values,width=.35,label=label,color=color)
        for j,v in enumerate(values):axes[1].text(j+(i-.5)*.36,v+1,f'{v:.0f}',ha='center',fontsize=8)
    axes[1].set_xticks(x,[p.replace('bird ','') for p in parts],rotation=25)
    axes[1].set_title('Visible part points inside a localized box (%)')
    axes[1].legend(loc='lower left',ncol=2,frameon=True)
    for ax in axes:
        ax.set_ylim(0,105);ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Frozen Grounding DINO Tiny | 100 CUB validation images + paired changed backgrounds',fontsize=15)
    fig.text(.5,.025,'Part boxes must cover at most half the native bird-box area; missed detections count as failures.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.055,1,.96))
    destination=OUT/'grounding_cub_quality.png'
    fig.savefig(destination,dpi=180,bbox_inches='tight');plt.close(fig)
    return destination


def prior_comparison(names=('rival10','cub','waterbirds')):
    """Single-seed matched mean/noise comparisons with and without spatial priors."""
    from .data import write_json
    settings=[('v3','mean','Prior on / mean','#b9a1d8'),
              ('v3','noise','Prior on / noise','#f3b47b'),
              ('v4_prior_off','mean','Prior off / mean','#7445a2'),
              ('v4_prior_off','noise','Prior off / noise','#d66a17')]
    fig,axes=plt.subplots(1,3,figsize=(16,4.8))
    summary={}
    for d,name in enumerate(names):
        summary[name]={}
        for j,(run,method,label,color) in enumerate(settings):
            path=ROOT/name/run/'seed42'/method
            result=json.loads((path/'results.json').read_text())
            a=np.load(path/'original_concepts.npy');b=np.load(path/'reversed_concepts.npy')
            drift=float(np.abs(a-b).mean())
            cosine=float((1-(a*b).sum(1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-8)).mean())
            row={'original_accuracy':result['original']['accuracy'],
                 'changed_background_accuracy':result['reversed']['accuracy'],
                 'concept_probability_mae':drift,'concept_cosine_distance':cosine,
                 'foreground_background_mean_ratio':result['reversed']['foreground_background_mean'],
                 'background_foreground_variance_ratio':result['reversed']['background_foreground_variance']}
            summary[name][run+'_'+method]=row
            for ax,value in zip(axes,[100*row['original_accuracy'],100*row['changed_background_accuracy'],100*drift]):
                x=d+(j-1.5)*.18
                ax.bar(x,value,width=.17,color=color,label=label if d==0 else None)
                ax.text(x,value+.4 if ax!=axes[2] else value+.06,f'{value:.1f}',ha='center',fontsize=8)
    for ax,title in zip(axes,['Original accuracy (%) ↑','Changed-background accuracy (%) ↑',
                              'Concept probability change (percentage points) ↓']):
        ax.set_title(title,fontsize=11)
        ax.set_xticks(np.arange(len(names)),['RIVAL10','CUB','Waterbirds'])
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].set_ylim(0,105);axes[1].set_ylim(0,105)
    axes[2].set_ylim(0,axes[2].get_ylim()[1]*1.10)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False)
    fig.suptitle('CLIP ViT-B/16 | one shared RGB view | seed 42',fontsize=15)
    fig.tight_layout(rect=(0,.10,1,.94))
    destination=OUT/'shared_spatial_prior_comparison.png'
    fig.savefig(destination,dpi=180,bbox_inches='tight');plt.close(fig)
    write_json(ROOT/'v4_prior_off_summary.json',summary)
    return destination


@torch.no_grad()
def distribution_figure(name,run='v3',seed=42):
    root,arrays,splits,protocol=load(name)
    checkpoint=torch.load(ROOT/name/run/f'seed{seed}/noise/best.pt',map_location='cpu',weights_only=False)
    model=SharedSpatialCBM.from_checkpoint(checkpoint).eval()
    labels=np.array(arrays['labels'])
    groups=np.unique(labels)
    wanted=groups[np.linspace(0,len(groups)-1,min(6,len(groups)),dtype=int)]
    chosen=[]
    for group in wanted:
        candidates=splits['test'][labels[splits['test']]==group]
        area=np.asarray(arrays['foreground'][candidates]).mean((1,2))
        order=np.argsort(np.abs(area-.35))
        chosen.append(int(candidates[order[0]]))
    rgb=torch.from_numpy(np.array(arrays['rgb_reversed'][chosen])).float()/255
    original=torch.from_numpy(np.array(arrays['tokens_reversed'][chosen])).float()
    with torch.random.fork_rng():
        torch.manual_seed(1207)
        processed,mu,sigma,fg=model.perturb(rgb,original,'noise')
    mean=mu.abs().mean(1).numpy();variance=sigma[:,0].square().numpy()
    mu_limit=max(float(np.percentile(mean,99.5)),1e-5)
    variance_limit=max(float(np.percentile(variance,99.5)),1e-6)
    fig,axes=plt.subplots(4,len(chosen),figsize=(3*len(chosen),10),squeeze=False)
    for col,index in enumerate(chosen):
        title=protocol['classes'][int(labels[index])].split('.',1)[-1].replace('_',' ')
        axes[0,col].set_title(title,fontsize=11)
        axes[0,col].imshow(rgb[col].permute(1,2,0).numpy())
        a=axes[1,col].imshow(mean[col],cmap='magma',vmin=0,vmax=mu_limit)
        b=axes[2,col].imshow(variance[col],cmap='viridis',vmin=0,vmax=variance_limit)
        axes[3,col].imshow(processed[col].clamp(0,1).permute(1,2,0).numpy())
        foreground=np.asarray(arrays['foreground'][index])
        for row in [1,2]:axes[row,col].contour(foreground,levels=[.5],colors='white',linewidths=.7)
        for row in range(4):axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
    for row,label in enumerate(['Changed background','Mean |mu|','Variance sigma^2','RGB + mean + noise']):
        axes[row,0].set_ylabel(label,fontsize=12)
    fig.subplots_adjust(left=.065,right=.89,top=.90,bottom=.035,wspace=.04,hspace=.1)
    fig.colorbar(a,cax=fig.add_axes([.91,.525,.012,.18])).set_label('Mean magnitude (RGB 0-1)')
    fig.colorbar(b,cax=fig.add_axes([.91,.3,.012,.18])).set_label('Variance (RGB 0-1 squared)')
    fig.suptitle(f'{name}: shared spatial noise',fontsize=17)
    OUT.mkdir(parents=True,exist_ok=True)
    destination=OUT/f'shared_{name}_{run}_visualization.png'
    fig.savefig(destination,dpi=200,facecolor='white',bbox_inches='tight');plt.close(fig)
    return destination


def result_figure(names,run='v3'):
    fig,axes=plt.subplots(len(names),3,figsize=(15,4*len(names)),squeeze=False)
    sources={}
    for row,name in enumerate(names):
        sources[name]={}
        for index,method in enumerate(COLORS):
            records=[]
            for seed in [42,43,44]:
                path=ROOT/name/run/f'seed{seed}'/method/'results.json'
                if path.exists():records.append((seed,json.loads(path.read_text()),path.parent))
            if not records:continue
            sources[name][method]=[r[0] for r in records]
            data=[[r[1]['reversed']['accuracy']*100 for r in records],
                  [r[1]['reversed']['mAP']*100 for r in records if 'mAP' in r[1]['reversed']],
                  [json.loads((r[2]/'background_leakage.json').read_text())['background_probe_balanced_accuracy']*100
                   for r in records if (r[2]/'background_leakage.json').exists()]]
            for column,values in enumerate(data):
                if not values:continue
                deviation=np.std(values,ddof=1) if len(values)>1 else 0
                axes[row,column].bar(index,np.mean(values),color=COLORS[method],width=.66,
                                     yerr=deviation,capsize=4,label=LABELS[method])
                axes[row,column].scatter(index+np.linspace(-.12,.12,len(values)),values,c='black',s=15,zorder=5)
                axes[row,column].text(index,np.mean(values)+deviation+1,f'{np.mean(values):.1f}',ha='center',fontsize=10)
        for column,title in enumerate(['Opposite background accuracy (%)','Native concept mAP (%)','Background leakage probe (%)']):
            axes[row,column].set_title(f'{name}: {title}',fontsize=11)
            axes[row,column].set_ylim(0,105);axes[row,column].set_xticks(range(4),['LP','SALF','Mean','Noise'])
            axes[row,column].grid(axis='y',alpha=.2);axes[row,column].set_axisbelow(True)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False)
    fig.tight_layout(rect=(0,0,1,.96))
    OUT.mkdir(parents=True,exist_ok=True)
    destination=OUT/f'shared_{run}_main_results.png'
    fig.savefig(destination,dpi=190,facecolor='white');plt.close(fig)
    return destination,sources


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',nargs='+',default=['rival10','cub','waterbirds']);p.add_argument('--run',default='v3')
    a=p.parse_args();torch.set_num_threads(6)
    for name in a.dataset:
        if (ROOT/name/a.run/'seed42/noise/best.pt').exists():print(distribution_figure(name,a.run))
    print(result_figure(a.dataset,a.run))


if __name__=='__main__':main()
