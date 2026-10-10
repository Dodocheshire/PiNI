"""Paper-shaped report and three main experiments, averaged over three rounds."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


OUTPUT = Path('reports/main')
SEEDS = [(42, 'region'), (43, 'region_seed43'), (44, 'region_seed44')]
BLUE, ORANGE, PURPLE = '#0072B2', '#E69F00', '#CC79A7'
TEAL = '#009E73'


def read(path):
    return json.loads(Path(path).read_text())


def style():
    font = Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 11,
                         'svg.fonttype': 'path', 'pdf.fonttype': 42})


def save(fig, name):
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(OUTPUT / f'{name}.{extension}', dpi=180)
    plt.close(fig)


def collect():
    result = dict(dataset='RIVAL10',backbone='Frozen OpenAI CLIP ViT-L/14',
                  classes=10,concepts=18,noise_location='RGB image pixels',
                  classification=[],spatial=[],background=[])
    source_files=[]
    for seed,prefix in SEEDS:
        run=Path('artifacts') / (prefix+'_noise')
        performance=read(run/'results.json');metrics=performance['same_checkpoint_mean']
        result['classification'].append(dict(seed=seed,accuracy=metrics['accuracy'],
            NEC=performance['NEC'],concept_mAP=metrics['mAP']))
        spatial=read(run/'localization_audit.json')['summary']
        result['spatial'].append(dict(seed=seed,**{key:spatial[key] for key in [
            'n_positive_pairs','n_local_pairs','sigma2_bg_over_fg',
            'mu_fg_over_bg','local_part_mu_over_other_object']}))
        folder='background_intervention_region' if seed==42 else f'background_intervention_region_seed{seed}'
        background_path=Path('artifacts')/folder/'results.json'
        models=read(background_path)['models']
        result['background'].append(dict(seed=seed,
            mean_trained=models[prefix+'_mean']['background_swap'],
            noise_trained_mean_inference=models[prefix+'_noise_same_checkpoint_mean']['background_swap']))
        source_files.extend([run/'results.json',run/'localization_audit.json',background_path])
    baselines=read('artifacts/paper_baselines/results.json')
    baseline_keys=['method','seed','original','background','concept_train_n','head_train_n',
                   'retained_concepts','concept_supervision','region_supervision','backbone']
    result['baselines']={method:[{key:r[key] for key in baseline_keys} for r in baselines['models'] if r['method']==method]
                         for method in ['supervised_cbm','lf_cbm']}
    source_files.append(Path('artifacts/paper_baselines/results.json'))
    result['baseline_recipe_verification']=read('artifacts/paper_baselines/recipe_verification.json')
    source_files.append(Path('artifacts/paper_baselines/recipe_verification.json'))
    result['rounds']=3;result['seeds']=[seed for seed,_ in SEEDS]
    result['averages']={group:{metric:dict(mean=float(np.mean([r[metric] for r in result[group]])),
        std=float(np.std([r[metric] for r in result[group]],ddof=1))) for metric in metrics}
        for group,metrics in [('classification',['accuracy','NEC','concept_mAP']),
            ('spatial',['sigma2_bg_over_fg','mu_fg_over_bg','local_part_mu_over_other_object'])]}
    result['background_average']={mode:{metric:float(np.mean([r[mode][metric] for r in result['background']]))
        for metric in ['accuracy','mAP']} for mode in ['mean_trained','noise_trained_mean_inference']}
    a=result['background_average']
    result['background_gain_pp']={metric:(a['noise_trained_mean_inference'][metric]-a['mean_trained'][metric])*100
                                 for metric in ['accuracy','mAP']}
    result['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    result['protocol']=dict(concept_train=18987,validation=2111,test=5286,adaptation_train=6003,
        adaptation_epochs=6,shared_concept_region_initialization='artifacts/spatial_region/best.pt',
        background_images=1000,classification_head='same validation-selected proximal L1 grid',
        head_l1_grid=[.003,.01,.03,.06,.1],NEC_threshold=1e-5,
        baseline_features='frozen ViT-L/14 normalized 768-dimensional CLS',
        baseline_concepts=18,lf_cbm_recipe='bias-free projection with cosine-cubed CLIP alignment; predefined 18 concepts retained')
    return result


def spatial_figure(result):
    path=Path('artifacts/region_noise/figure4_tensors.npz')
    with np.load(path) as data:
        mu=np.sqrt(np.mean(data['mu']**2,axis=1));variance=np.mean(data['sigma']**2,axis=1)
        fig,axes=plt.subplots(3,4,figsize=(11.6,8.2),layout='constrained')
        headers=['原图','均值偏移幅度 μ','噪声方差 σ²','实际加噪图片']
        for row,label in enumerate(['卡车：车轮','飞机：机翼','猫：耳朵']):
            axes[row,0].imshow(data['original'][row].transpose(1,2,0))
            m=axes[row,1].imshow(mu[row],cmap='magma',vmin=0,vmax=float(mu.max()))
            v=axes[row,2].imshow(variance[row],cmap='viridis',vmin=0,vmax=float(variance.max()))
            axes[row,3].imshow(np.clip(data['sample'][row].transpose(1,2,0),0,1))
            axes[row,0].set_ylabel(label)
            for col in range(4):
                axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
                if row==0:axes[row,col].set_title(headers[col])
        fig.colorbar(m,ax=axes[:,1],shrink=.55,label='均值偏移幅度（RGB）',pad=.015)
        fig.colorbar(v,ax=axes[:,2],shrink=.55,label='RGB²',pad=.015)
        fig.suptitle('RIVAL10 · CLIP ViT-L/14 · RGB 像素加噪',fontsize=15)
        result['visualization']=dict(ids=data['ids'][:3].tolist(),
            concept_ids=data['concept_ids'][:3].tolist(),sampling_seed=1207,
            tensor_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            mean_scale=[0,float(mu.max())],variance_scale=[0,float(variance.max())])
    save(fig,'spatial_noise')


def classification_figure(result):
    groups=[result['baselines']['lf_cbm'],result['baselines']['supervised_cbm'],result['classification']]
    labels=['LF-CBM\n投影基线','监督 CBM','空间加噪 CBM\n均值推理']
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
    for ax,key,title,limit in [(axes[0],'accuracy','原图分类准确率（%）↑',108),
        (axes[1],'mAP','原图概念 mAP（%）↑',108),(axes[2],'NEC','每个类别使用的概念数（NEC）↓',6.2)]:
        samples=[]
        for i,rows in enumerate(groups):
            samples.append([r['original'][key] if i<2 else r['concept_mAP' if key=='mAP' else key] for r in rows])
        samples=np.array(samples)*(1 if key=='NEC' else 100)
        means=samples.mean(1);stds=samples.std(1,ddof=1)
        ax.bar(np.arange(3),means,yerr=stds,capsize=4,color=[BLUE,TEAL,ORANGE],width=.6)
        for i,value in enumerate(means):
            ax.text(i,value+stds[i]+limit*.024,f'{value:.2f}',ha='center',fontweight='bold')
            ax.scatter(np.full(3,i),samples[i],s=13,color='#333333',zorder=4)
        ax.set_xticks(np.arange(3),labels);ax.set_ylim(0,limit);ax.set_title(title)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        if key!='NEC':ax.set_yticks([0,25,50,75,100])
    fig.suptitle('实验 b：RIVAL10 原图测试 5,286 张 · 3 轮平均与标准差',fontsize=15)
    save(fig,'classification')


def background_figure(result):
    groups=[result['baselines']['lf_cbm'],result['baselines']['supervised_cbm'],result['background'],result['background']]
    labels=['LF-CBM\n投影基线','监督 CBM','空间 CBM\n均值训练','空间加噪 CBM\n噪声训练']
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.8),layout='constrained')
    for ax,key,title in [(axes[0],'accuracy','换背景分类准确率（%）'),(axes[1],'mAP','换背景概念 mAP（%）')]:
        samples=np.array([[r['background'][key] if i<2 else r['mean_trained' if i==2 else 'noise_trained_mean_inference'][key]
            for r in rows] for i,rows in enumerate(groups)])*100
        means=samples.mean(1);stds=samples.std(1,ddof=1)
        ax.bar(np.arange(4),means,yerr=stds,capsize=4,color=[BLUE,TEAL,PURPLE,ORANGE],width=.6)
        for i,value in enumerate(means):
            ax.text(i,value+stds[i]+1.1,f'{value:.2f}%',ha='center',fontweight='bold')
            ax.scatter(np.full(3,i),samples[i],s=13,color='#333333',zorder=4)
        ax.set_xticks(np.arange(4),labels);ax.set_ylim(0,114);ax.set_yticks([0,25,50,75,100]);ax.set_title(title)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.text(.5,.96,f"噪声训练相对均值训练 {result['background_gain_pp'][key]:+.2f} 个百分点",transform=ax.transAxes,
                ha='center',va='top',color='#B55200',fontweight='bold')
    fig.suptitle('实验 c：同一组换背景图片 1,000 张 · 3 轮平均与标准差',fontsize=15)
    save(fig,'background_gain')


def write_report(result):
    from .paper_text import write_report as write_paper
    write_paper(result)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);style();result=collect()
    spatial_figure(result);classification_figure(result);background_figure(result)
    from .pipeline_figure import main as pipeline
    result['pipeline']=pipeline()
    from .detailed_figures import main as detailed
    result['tensor_shapes']=detailed()
    write_report(result)
    (OUTPUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if Path('artifacts/cross_dataset/ALL_COMPLETE.json').exists():
        from .cross_reporting import main as cross_report
        cross_report()
    elif Path('artifacts/cross_dataset/BASELINE_VERIFICATION.json').exists():
        from .cross_reporting import baseline_progress
        baseline_progress()
    print('Updated EXPERIMENT_REPORT.md, detailed modules, and three-round main experiments',flush=True)


if __name__=='__main__':
    main()
