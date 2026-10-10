"""One concise experiment report, drawn from completed regional runs."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


OUTPUT = Path('reports/main')
SEEDS = [(42, 'region'), (43, 'region_seed43')]
BLUE, ORANGE, PURPLE = '#0072B2', '#E69F00', '#CC79A7'


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
        folder='background_intervention_region' if seed==42 else 'background_intervention_region_seed43'
        background_path=Path('artifacts')/folder/'results.json'
        models=read(background_path)['models']
        result['background'].append(dict(seed=seed,
            mean_trained=models[prefix+'_mean']['background_swap'],
            noise_trained_mean_inference=models[prefix+'_noise_same_checkpoint_mean']['background_swap']))
        source_files.extend([run/'results.json',run/'localization_audit.json',background_path])
    result['background_average']={mode:{metric:float(np.mean([r[mode][metric] for r in result['background']]))
        for metric in ['accuracy','mAP']} for mode in ['mean_trained','noise_trained_mean_inference']}
    a=result['background_average']
    result['background_gain_pp']={metric:(a['noise_trained_mean_inference'][metric]-a['mean_trained'][metric])*100
                                 for metric in ['accuracy','mAP']}
    result['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
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
    rows=result['classification'];labels=[f"训练 {i+1}（seed {r['seed']}）" for i,r in enumerate(rows)]
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.9),layout='constrained')
    for ax,key,title,limit in [(axes[0],'accuracy','分类准确率（%）',105),(axes[1],'NEC','每个类别使用的概念数（NEC）',5.6)]:
        values=[r[key]*100 if key=='accuracy' else r[key] for r in rows]
        ax.bar(labels,values,color=[ORANGE,PURPLE],width=.52)
        for i,value in enumerate(values):ax.text(i,value+limit*.02,f'{value:.2f}' if key=='accuracy' else f'{value:.1f}',ha='center',fontweight='bold')
        ax.set_ylim(0,limit);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        if key=='accuracy':ax.set_yticks([0,25,50,75,100])
    fig.suptitle('原图测试 5,286 张：噪声训练，均值推理',fontsize=14)
    save(fig,'classification')


def background_figure(result):
    averages=result['background_average'];modes=['mean_trained','noise_trained_mean_inference']
    fig,axes=plt.subplots(1,2,figsize=(10,4.1),layout='constrained')
    for ax,key,title in [(axes[0],'accuracy','换背景分类准确率（%）'),(axes[1],'mAP','换背景概念 mAP（%）')]:
        values=[averages[mode][key]*100 for mode in modes]
        ax.bar(['均值训练','噪声训练'],values,color=[BLUE,ORANGE],width=.52)
        for i,value in enumerate(values):ax.text(i,value+1,f'{value:.2f}%',ha='center',fontweight='bold')
        ax.set_ylim(0,109);ax.set_yticks([0,25,50,75,100]);ax.set_title(title)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.text(.5,.96,f"提升 +{result['background_gain_pp'][key]:.2f} 个百分点",transform=ax.transAxes,
                ha='center',va='top',color='#B55200',fontweight='bold')
    fig.suptitle('换背景测试 1,000 张 · 两次训练的指标平均 · 两组均取均值推理',fontsize=13)
    save(fig,'background_gain')


def write_report(result):
    c=result['classification'];s=result['spatial'];b=result['background_average'];gain=result['background_gain_pp']
    content=f'''# 空间加噪 CBM：主要实验结果

- 空间 CBM 实现了真实的“背景高方差、概念部位大均值偏移”效果。
- 噪声训练后均值推理：分类约 **99.1%**，NEC **4.4–4.5**。
- 两个固定适配模型平均：换背景分类 **+1.00 个百分点**、概念 mAP **+0.75 个百分点**。

## 模型怎么工作

![空间加噪 CBM Pipeline](reports/main/pipeline.png)

图中从上到下依次是：学习主体与概念区域、在 RGB 像素上训练噪声、取均值图预测 18 个概念并分类。

使用 **RIVAL10**，包含卡车、飞机、猫等 10 个类别，以及车轮、机翼、耳朵等 18 个概念。Backbone 是冻结的 **CLIP ViT-L/14**。模型结合图像特征和概念文本，找出主体和相应部位，先减弱背景，再在原图的 **RGB 像素**上加噪：背景的随机扰动更强，概念部位的均值偏移更大。每个概念对应一张处理后的图片，再由 ViT 识别；最后用 18 个概念的预测概率完成分类。

训练顺序是先学概念和区域，再训练噪声生成器与概念识别器，最后训练稀疏分类头。区域模型用 18,987 张训练图预训练；加噪阶段用 6,003 张图训练 6 轮，分别运行 seed 42 和 43。训练时使用随机采样，性能测试时使用学到的均值。

## 实验一：噪声是否集中在该放的位置

**目的：**观察背景是否受到更强随机扰动，车轮、机翼、耳朵等概念部位是否得到更大的均值偏移。

**怎么跑：**RIVAL10 验证集 2,111 张，backbone 为 CLIP ViT-L/14；在 RGB 像素上生成概念对应的均值 μ 和方差 σ²。图中从左到右为原图、均值偏移、噪声方差、实际加噪图片。

**指标含义：**μ 的幅度表示像素被稳定改变多少，σ² 表示随机扰动有多强。均值图越亮，偏移越大；方差图越黄，扰动越强，紫色表示较弱。统计覆盖 9,022 个有效正概念区域，其中 4,080 个是车轮、耳朵等局部部位。

![背景高方差、概念部位大均值偏移](reports/main/spatial_noise.png)

| 空间指标 | 训练 1 | 训练 2 |
|---|---:|---:|
| 背景方差 / 主体方差 | {s[0]['sigma2_bg_over_fg']:.2f} 倍 | {s[1]['sigma2_bg_over_fg']:.2f} 倍 |
| 主体均值偏移 / 背景均值偏移 | {s[0]['mu_fg_over_bg']:.2f} 倍 | {s[1]['mu_fg_over_bg']:.2f} 倍 |
| 概念部位均值偏移 / 主体其他部位均值偏移 | {s[0]['local_part_mu_over_other_object']:.2f} 倍 | {s[1]['local_part_mu_over_other_object']:.2f} 倍 |

**结论：**背景受到更强随机扰动，均值偏移集中在主体，尤其是当前要识别的概念部位。

## 实验二：分类准确率和概念使用数量

**目的：**检查这个模型能否正确分类，以及每个类别需要多少概念。

**怎么跑：**RIVAL10 原图测试集 5,286 张，backbone 为 CLIP ViT-L/14。使用噪声训练得到的模型，按预测区域减弱背景，在 RGB 像素上施加学到的均值，再预测概念并分类。

**指标含义：**分类准确率是图片类别预测正确的比例，越高越好。NEC 是每个类别平均实际使用的概念数；越低，分类所依赖的概念越少。

![原图分类准确率与 NEC](reports/main/classification.png)

| 训练 | 分类准确率 | NEC |
|---|---:|---:|
| 训练 1（seed 42） | {c[0]['accuracy']*100:.2f}% | {c[0]['NEC']:.1f} |
| 训练 2（seed 43） | {c[1]['accuracy']*100:.2f}% | {c[1]['NEC']:.1f} |

**结论：**噪声训练后均值推理的分类准确率约 99.1%，每个类别平均使用 4.4–4.5 个概念。

## 实验三：换背景后是否更容易认对

**目的：**检查背景变化时，训练加噪能否帮助识别概念和图片类别。

**怎么跑：**从 RIVAL10 测试集取 1,000 张图，保留主体，把背景换成其他类别图片的背景。Backbone 仍为 CLIP ViT-L/14；两组分别用“只学均值”和“均值加随机噪声”训练，噪声加在 RGB 像素上，测试时两组都取均值。分别计算 seed 42、43 的指标，再取平均。

**指标含义：**分类准确率衡量换背景后能否认对类别。概念 mAP 先对每个概念计算平均精度 AP，再对 18 个概念取平均；它衡量模型能否把真正具有该概念的图片排在前面，越高越好。百分点是两个百分比直接相减。

![换背景后的分类与概念识别收益](reports/main/background_gain.png)

| 训练方式 | 换背景分类准确率 | 换背景概念 mAP |
|---|---:|---:|
| 均值训练 | {b['mean_trained']['accuracy']*100:.2f}% | {b['mean_trained']['mAP']*100:.2f}% |
| 噪声训练 | {b['noise_trained_mean_inference']['accuracy']*100:.2f}% | {b['noise_trained_mean_inference']['mAP']*100:.2f}% |
| 提升 | **+{gain['accuracy']:.2f} 个百分点** | **+{gain['mAP']:.2f} 个百分点** |

**结论：**训练时加入概念条件噪声，让模型在背景改变后更容易认对概念和类别。
'''
    Path('EXPERIMENT_RESULT.md').write_text(content,encoding='utf-8')


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);style();result=collect()
    spatial_figure(result);classification_figure(result);background_figure(result)
    from .pipeline_figure import main as pipeline
    result['pipeline']=pipeline()
    write_report(result)
    (OUTPUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Updated EXPERIMENT_RESULT.md, pipeline, and three main figure groups',flush=True)


if __name__=='__main__':
    main()
