"""Readable, common-color plots of completed mechanism controls only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS=dict(clean='#0072B2',mask='#56B4E9',mean='#009E73',noise='#E69F00',joint='#D55E00',separate='#CC79A7')
LABELS={'clean':'Clean','mask_sequential':'Mask only','mean_sequential':'Mean only','noise_sequential':'Gaussian (MC1)',
        'local_clean':'Local clean','region_clean':'Region clean',
        'noise_sequential_same_checkpoint_mean':'Gaussian checkpoint, mean inference',
        'local_mean':'Local mean','local_noise':'Local Gaussian','local_noise_same_checkpoint_mean':'Local Gaussian checkpoint, mean',
        'full_joint_mean':'Full joint mean','full_joint_noise':'Full joint Gaussian','full_joint_noise_same_checkpoint_mean':'Full joint Gaussian checkpoint, mean',
        'region_mean':'Region mean','region_noise':'Region Gaussian','region_noise_same_checkpoint_mean':'Region Gaussian checkpoint, mean',
        'fixed_mean':'Fixed-FG mean','fixed_noise':'Fixed-FG Gaussian','fixed_noise_same_checkpoint_mean':'Fixed-FG Gaussian checkpoint, mean'}


def save(fig,path):
    for ext in ['png','svg','pdf']:fig.savefig(str(path)+'.'+ext,dpi=180)
    plt.close(fig)


def errors(ax,y,center,ci,color,label=None):
    ci=np.array(ci);center=np.asarray(center)
    ax.errorbar(center,y,xerr=np.maximum(0,np.stack([center-ci[:,0],ci[:,1]-center])),fmt='o',color=color,capsize=3,label=label)


def main():
    out=Path('reports/spatial_noise_v1');out.mkdir(parents=True,exist_ok=True)
    readout=out/'readout_path_audit.json'
    if readout.exists():
        data=json.loads(readout.read_text())['models'];fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained')
        y=np.arange(len(data))
        for offset,key,color in [(-.15,'CLS',COLORS['clean']),(0,'region',COLORS['mean']),(.15,'combined',COLORS['noise'])]:
            values=[r['mAP'][key]*100 for r in data.values()]
            ax.scatter(values,y+offset,color=color,label=key,s=55)
            for pos,value in zip(y+offset,values):ax.annotate(f'{value:.2f}',(value,pos),xytext=(4,0),textcoords='offset points',fontsize=9)
        ax.set_yticks(y,['Original clean','Locality-supervised clean'][:len(data)]);ax.invert_yaxis()
        ax.set_xlim(86,93);ax.set_xlabel('Native concept mAP (%)');ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8)
        ax.set_title('Existing fitted head: isolated additive contributions\nDiagnostic only; no retraining or test calibration',fontsize=10)
        save(fig,out/'readout_path_diagnostic')
    influence=[]
    for name,label in [('noise_sequential','Sequential'),('region_noise','Region-coupled'),('fixed_noise','Fixed foreground')]:
        path=out/(name+'_noise_region_intervention.json')
        if path.exists():influence.append((label,json.loads(path.read_text())))
    if influence:
        fig,ax=plt.subplots(figsize=(7,3.6),layout='constrained');y=np.arange(len(influence))
        for offset,key,color,label in [(-.15,'all',COLORS['clean'],'Full sampled noise'),(0,'foreground',COLORS['mean'],'Foreground-only noise'),(.15,'background',COLORS['noise'],'Background-only noise')]:
            values=[np.sqrt(r['modes'][key]['mean_native_logit_variance']) for _,r in influence]
            ax.scatter(values,y+offset,color=color,s=55,label=label)
        ax.set_yticks(y,[label for label,_ in influence]);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8)
        ax.set_xlabel('RMS native-logit fluctuation over 8 paired draws')
        ax.set_title('Where sampled noise affects recognition\nFixed validation images; GT only constructs external interventions',fontsize=10)
        save(fig,out/'noise_region_influence')
    drift=out/'foreground_drift.json'
    if drift.exists():
        data=json.loads(drift.read_text());rows=list(data['models'].items());y=np.arange(len(rows))
        fig,axes=plt.subplots(1,2,figsize=(11,max(4,len(rows)*.4)),layout='constrained')
        for j,(name,row) in enumerate(rows):
            color=COLORS['noise' if 'noise' in name else 'mean' if 'mean' in name else 'clean']
            axes[0].scatter(row['foreground_IoU']*100,j,color=color,s=50)
            axes[1].scatter(row['foreground_probability_on_true_background']*100,j,color=color,s=50)
        for ax in axes:ax.set_yticks(y,[name.replace('_',' ') for name,_ in rows]);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_xlim(0,100)
        axes[0].axvline(data['all_foreground_IoU']*100,color='gray',linestyle='--',label='Predict all pixels foreground');axes[0].legend(fontsize=8)
        axes[0].set_xlabel('Foreground IoU (%)');axes[1].set_xlabel('Mean foreground probability on true background (%)')
        fig.suptitle('Does RGB adaptation preserve the foreground blocker?',fontsize=12);save(fig,out/'foreground_drift')
    for folder,figure in [('background_intervention','background_results'),('background_intervention_local','background_local_results'),('background_intervention_region','background_region_results'),('background_intervention_fixed','background_fixed_results')]:
        bg=Path('artifacts')/folder/'results.json'
        if not bg.exists():continue
        data=json.loads(bg.read_text());models=dict(data['models'])
        extra=Path('artifacts/background_clean_models/results.json')
        baseline='local_clean' if folder=='background_intervention_local' else 'region_clean' if folder in ['background_intervention_region','background_intervention_fixed'] else None
        if extra.exists() and baseline:
            e=json.loads(extra.read_text());protocol=json.loads((bg.parent/'protocol.json').read_text())
            assert protocol['ids']==e['reference_ids'] and protocol['donors']==e['reference_donors']
            models[baseline]=e['models'][baseline]
        rows=list(models.items());y=np.arange(len(rows))
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for j,(name,row) in enumerate(rows):
            v=[row['original']['accuracy']*100,row['background_swap']['accuracy']*100]
            color=COLORS['mean' if 'mean' in name else 'noise' if 'noise' in name else 'mask' if 'mask' in name else 'clean']
            axes[0].plot(v,[j,j],color=color,alpha=.6)
            axes[0].scatter(v[0],j,facecolors='none',edgecolors=color,s=70)
            axes[0].scatter(v[1],j,color=color,s=60)
            axes[0].annotate(f'{v[0]:.1f} → {v[1]:.1f}',(v[1],j),xytext=(4,8),textcoords='offset points',fontsize=9)
            axes[1].scatter(row['background_swap']['mAP']*100,j,color=color,s=60)
            axes[1].annotate(f"{row['background_swap']['mAP']*100:.2f}",(row['background_swap']['mAP']*100,j),xytext=(5,5),textcoords='offset points',fontsize=9)
        names=[LABELS.get(n,n) for n,_ in rows]
        for ax in axes:
            ax.set_yticks(y,names);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
        axes[0].set_xlabel('Classification accuracy (%)');axes[0].set_xlim(min(90,min(r['background_swap']['accuracy']*100 for _,r in rows)-2),100.2)
        axes[0].set_title('Open: original; filled: foreign background')
        axes[1].set_title('Native concept mAP after background swap');axes[1].set_xlabel('mAP (%)')
        ap_values=[r['background_swap']['mAP']*100 for _,r in rows];axes[1].set_xlim(min(ap_values)-1,max(ap_values)+1)
        fig.suptitle(f"Fixed {data['n']}-image intervention — target foreground preserved; donor object removed",fontsize=12)
        save(fig,out/figure)
    uncertainty=out/'paired_uncertainty.json'
    native_bg=out/'background_native_uncertainty.json'
    if native_bg.exists():
        comparisons=[];native=json.loads(native_bg.read_text())['pairs']
        for folder,a,b,label in [('background_intervention','noise_sequential','mean_sequential','Sequential'),('background_intervention_local','local_noise','local_mean','Local'),('background_intervention_local','full_joint_noise','full_joint_mean','Full joint'),('background_intervention_region','region_noise','region_mean','Region seed 42'),('background_intervention_fixed','fixed_noise','fixed_mean','Fixed foreground'),('background_intervention_region_seed43','region_seed43_noise','region_seed43_mean','Region seed 43')]:
            path=Path('artifacts')/folder/'results.json';key=a+'_minus_'+b
            if path.exists() and key in native:
                c=json.loads(path.read_text()).get('direct_noise_controls',{}).get(key+'_background_swap')
                if c:comparisons.append((label,c,native[key]))
        if comparisons:
            fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained');y=np.arange(len(comparisons))
            errors(axes[0],y,[c['difference_pp'] for _,c,_ in comparisons],[c['CI95_pp'] for _,c,_ in comparisons],COLORS['noise'])
            errors(axes[1],y,[n['mAP_difference_pp'] for _,_,n in comparisons],[n['mAP_CI95_pp'] for _,_,n in comparisons],COLORS['noise'])
            for ax in axes:
                ax.axvline(0,color='black',lw=1);ax.set_yticks(y,[label for label,_,_ in comparisons]);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_xlabel('Gaussian minus matched mean-only (pp)')
            axes[0].set_title('Background-swap class accuracy');axes[1].set_title('Background-swap native concept mAP')
            fig.suptitle('Noise increment under changed backgrounds: paired 95% intervals\nFixed 1,000 images; donor object removed; all 18 native concepts',fontsize=10)
            save(fig,out/'background_noise_increment')
    if uncertainty.exists():
        data=json.loads(uncertainty.read_text());pairs=list(data['pairs'].items())
        if pairs:
            names=[key.split('_minus_')[0].replace('noise_','').replace('_noise','').replace('_',' ') for key,_ in pairs]
            fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');y=np.arange(len(pairs))
            errors(axes[0],y,[v['classification']['difference_pp'] for _,v in pairs],[v['classification']['CI95_pp'] for _,v in pairs],COLORS['noise'],'Own sparse head')
            shared=[(j,v['shared_GT_semantic_head']) for j,(_,v) in enumerate(pairs) if 'shared_GT_semantic_head' in v]
            if shared:errors(axes[0],np.array([j for j,_ in shared])+.15,[v['difference_pp'] for _,v in shared],[v['CI95_pp'] for _,v in shared],COLORS['clean'],'Shared GT-semantic head')
            errors(axes[1],y,[v['mAP_difference_pp'] for _,v in pairs],[v['mAP_CI95_pp'] for _,v in pairs],COLORS['noise'])
            for ax in axes:
                ax.axvline(0,color='black',lw=1);ax.set_yticks(y,names);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_xlabel('Gaussian − independently trained mean (pp)')
            axes[0].set_title('Classification difference, paired 95% CI');axes[0].legend(fontsize=8)
            axes[1].set_title('Native concept mAP difference, paired 95% CI')
            fig.suptitle('Noise increment with the same foreground suppression and initialization',fontsize=12)
            save(fig,out/'noise_increment')
        seq=data['pairs'].get('noise_sequential_minus_mean_sequential')
        if seq:
            rows=seq['per_concept'];y=np.arange(len(rows));d=[r['difference_AP_pp'] for r in rows]
            fig,ax=plt.subplots(figsize=(8,8),layout='constrained')
            errors(ax,y,d,[r['simultaneous_CI95_pp'] for r in rows],'#bdbdbd','Simultaneous 18-concept interval')
            errors(ax,y,d,[r['CI95_pp'] for r in rows],COLORS['noise'],'Pointwise 95% interval')
            ax.axvline(0,color='black',lw=1);ax.set_yticks(y,[r['name'] for r in rows]);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
            ax.set_xlabel('Sequential Gaussian − independently trained mean: concept AP (pp)')
            ax.set_title('All native concepts; no selection of only successful examples');ax.legend(fontsize=8)
            save(fig,out/'concept_AP_differences')
    official=Path('reports/pini_official');official.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained');any_result=False
    for ax,kind,paper in zip(axes,['main','image'],[92.97,89.51]):
        ax.axhline(paper,color='#0072B2',linestyle='--',label=f'Paper {paper:.2f}%')
        protocol_names=[]
        for j,suffix in enumerate(['seed1','fp16_seed1']):
            run=Path('artifacts')/('pets_'+kind+'_'+suffix)
            if not (run/'results.json').exists():continue
            row=json.loads((run/'results.json').read_text());any_result=True
            protocol_names.append(('AMP / batch 128' if suffix=='seed1' else 'FP16 / batch 32',j))
            for offset,key,color,label in [(-.15,'MC1','#E69F00','Gaussian MC1'),(0,'MC8','#D55E00','Gaussian MC8'),(.15,'same_checkpoint_mean','#009E73','Same checkpoint mean')]:
                if key not in row:continue
                score=row[key]['accuracy']*100;ax.scatter(j+offset,score,color=color,s=65,label=label if j==0 else None)
                ax.annotate(f'{score:.2f}',(j+offset,score),xytext=(0,7 if key!='MC1' else -13),textcoords='offset points',ha='center',fontsize=9)
            if suffix=='seed1':
                control=Path('artifacts')/('pets_'+kind+'_mean_seed1/results.json')
                if control.exists():
                    score=json.loads(control.read_text())['MC1']['accuracy']*100
                    ax.scatter(j+.3,score,marker='s',color='#56B4E9',s=50,label='Independently trained mean')
                    ax.annotate(f'{score:.2f}',(j+.3,score),xytext=(5,-8),textcoords='offset points',fontsize=9)
        ax.set_xticks([j for _,j in protocol_names],[n for n,_ in protocol_names]);ax.set_ylim(83,95);ax.set_xlim(-.5,1.5)
        ax.set_ylabel('Test accuracy (%)');ax.set_title('Global visual + prompt noise' if kind=='main' else 'Original image CNN');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=7,loc='lower left')
    if any_result:
        fig.suptitle('OxfordPets 16-shot — 200 epochs, seed 1; full 3,669-image test\nMC8 is an added diagnostic; paper does not specify this ensemble protocol',fontsize=11)
        save(fig,official/'official_results')
    else:plt.close(fig)
    print('Completed mechanism plots refreshed',flush=True)


if __name__=='__main__':main()
