"""Compact result report from completed experiments; pending is never a score."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .train_spatial import load_cache


RUNS=['mean_independent','noise_independent','mask_sequential','mean_sequential','noise_sequential','noise_joint','mean_separate_heads','noise_separate_heads',
      'local_mean','local_noise','full_joint_mean','full_joint_noise','region_mean','region_noise','fixed_mean','fixed_noise','region_seed43_mean','region_seed43_noise']


def pair_ci(a,b):
    diff=a.astype(float)-b.astype(float);rng=np.random.default_rng(1149)
    v=[diff[rng.integers(0,len(diff),len(diff))].mean() for _ in range(2000)]
    return float(diff.mean()*100),np.quantile(v,[.025,.975])*100


def main():
    # Refresh audits from the same completed snapshots as this report.
    # These are CPU-only and their expensive bootstrap portions validate reuse.
    from . import concept_audit,semantic_readout,semantic_capacity_audit,audit_mc_selection,foreground_drift_audit,uncertainty,official_controls,background_controls,background_native_audit,background_precision_audit,two_seed_audit,mechanism_figures
    concept_audit.main();semantic_readout.main();semantic_capacity_audit.main();audit_mc_selection.main();foreground_drift_audit.main();uncertainty.main();official_controls.main();background_controls.main();background_native_audit.main();background_precision_audit.main();two_seed_audit.main();mechanism_figures.main()
    if Path('artifacts/region_noise/figure4_tensors.npz').exists():
        from . import main_figure4
        main_figure4.main()
    report=Path('reports/spatial_noise_v1');report.mkdir(parents=True,exist_ok=True)
    cache=Path('artifacts/rival_vitl14');arrays,splits,labels,records=load_cache(cache)
    clean=json.loads(Path('artifacts/spatial_clean/results.json').read_text())
    audit=json.loads(Path('artifacts/spatial_clean/semantic_audit.json').read_text())
    baseline=audit['matched_6000_continuous'];rows=[]
    rows.append(dict(name='clean matched head',test_accuracy=baseline['test_accuracy'],NEC=baseline['NEC'],concepts=clean['concepts'],color='#0072B2'))
    pending=[]
    for name in RUNS:
        p=Path('artifacts')/name/'results.json'
        if not p.exists():pending.append(name);continue
        row=json.loads(p.read_text());row['name']=name
        row['color']='#E69F00' if 'noise' in name else '#009E73'
        if 'joint' in name:row['color']='#D55E00'
        if 'separate' in name:row['color']='#CC79A7'
        rows.append(row)
    local_audit=Path('artifacts/spatial_local/semantic_audit.json')
    if local_audit.exists():
        lc=json.loads(Path('artifacts/spatial_local/results.json').read_text());la=json.loads(local_audit.read_text())['matched_6000_continuous']
        rows.append(dict(name='local clean matched head',test_accuracy=la['test_accuracy'],NEC=la['NEC'],concepts=lc['concepts'],color='#0072B2'))
    region_audit=Path('artifacts/spatial_region/semantic_audit.json')
    if region_audit.exists():
        rc=json.loads(Path('artifacts/spatial_region/results.json').read_text());ra=json.loads(region_audit.read_text())['matched_6000_continuous']
        rows.append(dict(name='region clean matched head',test_accuracy=ra['test_accuracy'],NEC=ra['NEC'],concepts=rc['concepts'],color='#0072B2'))
    fig,axes=plt.subplots(1,3,figsize=(14,max(4.8,len(rows)*.48+1.6)),layout='constrained')
    vals=[[r['test_accuracy']*100 for r in rows],[r['concepts']['test']['mAP']*100 for r in rows],[r['NEC'] for r in rows]]
    titles=['Classification accuracy (%)','Native concept mAP (%)','NEC (active connections / class)']
    xs=np.arange(len(rows));names=[r['name'].replace('_',' ') for r in rows]
    for ax,v,title in zip(axes,vals,titles):
        ax.scatter(v,xs,color=[r['color'] for r in rows],s=60)
        ax.set_yticks(xs,names if ax is axes[0] else ['']*len(names),fontsize=9);ax.invert_yaxis();ax.set_title(title,fontsize=10)
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        if ax is axes[2]:ax.set_xlim(0,max(8,max(v)*1.25))
        else:ax.set_xlim(max(0,np.floor(min(v))-1),min(100,np.ceil(max(v))+2))
        for y,value in zip(xs,v):ax.annotate(f'{value:.2f}',(value,y),xytext=(6,0),textcoords='offset points',ha='left',va='center',fontsize=9)
    fig.suptitle('RIVAL10 — full 5,286-image test; frozen ViT-L/14; 18 supervised native concepts',fontsize=12)
    for ext in ['png','svg','pdf']:fig.savefig(report/('main_results.'+ext),dpi=180)
    plt.close(fig)
    text=['# PiNI 复现与空间加噪 CBM 实验结果','',
          '本报告由已完成的实验输出生成；正在训练或评估的配置不填写成绩。官方复验、训练方式对照、两次区域适配和背景干预均已完成；完整目标的核验另见 GOAL_VERIFICATION.md。','',
          '## 主要结果','',
          '目的：在真实概念监督、相同初始化和固定训练/验证/测试协议下，检验 RGB 噪声能否提高概念识别与稀疏分类。分类头仅读取 18 个概念概率，无图像特征旁路。NEC 是每类别非零概念连接数，不是模型参数数目。','',
          '| 方法 | 测试分类准确率 | NEC | 原生概念 mAP | 宏精确率 @0.5 | 宏召回率 @0.5 |',
          '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        m=r['concepts']['test'];text.append(f"| {r['name']} | {r['test_accuracy']*100:.2f}% | {r['NEC']:.2f} | {m['mAP']*100:.2f}% | {m['macro_precision']*100:.2f}% | {m['macro_recall']*100:.2f}% |")
    text+=['','![主要结果](reports/spatial_noise_v1/main_results.png)','',
           f"干净模型全 18,987 个训练样本拟合分类头：准确率 {clean['test_accuracy']*100:.2f}%，NEC {clean['NEC']:.2f}；表中使用与加噪实验相同的 6,003 个分类头训练样本。主体定位测试 IoU {clean['concepts']['test']['fg_IoU']*100:.2f}%，正概念区域 IoU {clean['concepts']['test']['concept_mask_IoU']*100:.2f}%。",'',
           f"可解释性检查：将预测概率二值化后，分类准确率 {audit['binary']['test_accuracy']*100:.2f}%，NEC {audit['binary']['NEC']:.2f}；原生真值二元概念分类头为 {clean['oracle']['test_accuracy']*100:.2f}%。预测概念中的概率或系统性误差能携带额外类别信息，因此高分类成绩不能代替原生概念精度。",'',
           '## 真实 Figure 4 风格可视化','',
           '目的：检查主体均值增强与背景随机扰动是否实际发生。六个样本按固定类别/概念选择验证集中首个文件，不按改善幅度挑选。μ 使用 RGB RMS，σ² 使用实际 RGB²；列内共用色标，采样不放大。','',
           '![实际加噪图](reports/spatial_noise_v1/figure4_style.png)','']
    energy=Path('artifacts/noise_independent/energy_audit.json')
    if energy.exists():
        e=json.loads(energy.read_text())['summary'];p=e['part_localization']
        text+=[f"结论：全 {e['n']} 个有效验证样本的背景方差/主体方差为 {e['sigma2_bg_over_fg']:.2f}，主体 μ/背景 μ 为 {e['mu_fg_over_bg']:.2f}。概念部位 μ 与主体其他部位之比仅为 {p['concept_mu_rms']/p['other_object_mu_rms']:.2f}，尚需改善细粒度定位。",'']
    text+=['主体区域由带监督的 ViT/text 模型预测；高背景 σ 和前景 μ 保护包含显式结构先验。μ 表示主体门控之后的概念均值偏移；相对原图的总均值变化还包括背景抑制。图形证明实际机制实现，不证明这些区域是原始 PiNI 分类 CE 自发发现的，也不能单独证明随机噪声带来性能收益。','',
           '## 随机噪声的增量作用','']
    contrasts=[]
    for order,an,bn in [('independent','noise_independent','mean_independent'),('sequential','noise_sequential','mean_sequential'),
                        ('local','local_noise','local_mean'),('full_joint','full_joint_noise','full_joint_mean'),('region','region_noise','region_mean')]:
        a=Path('artifacts')/an;b=Path('artifacts')/bn
        if (a/'results.json').exists() and (b/'results.json').exists():
            ap=np.load(a/'test_class_logits.npy').argmax(1)==labels[splits['test']]
            bp=np.load(b/'test_class_logits.npy').argmax(1)==labels[splits['test']]
            d,ci=pair_ci(ap,bp);contrasts.append(dict(order=order,difference_pp=d,CI95_pp=ci.tolist()))
            text.append(f"{order}：分别训练的 μ+σξ − μ-only 为 {d:+.3f} 个百分点，配对 bootstrap 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]。同样主体抑制和初始化，随机部分的收益必须由这组比较支持。")
            text.append('')
    for r in rows:
        if 'same_checkpoint_mean' in r:
            mean=r['same_checkpoint_mean'];d=(r['test_accuracy']-mean['accuracy'])*100
            text.append(f"{r['name']} 同一检查点 MC1 − 均值推理为 {d:+.3f} 个百分点；这是推理诊断，不代替分别训练的均值对照。")
            text.append('')
    if not contrasts:text+=['匹配的确定性均值对照尚未全部完成，不给随机噪声独立增益结论。','']
    concept=report/'concept_audit.json'
    if concept.exists():
        c=json.loads(concept.read_text());text+=['固定 0.5 阈值的精确率提升可能来自召回率下降。以下阈值只由验证集正样本的第 10 百分位确定，以目标召回率 90% 比较：','',
            '| 方法 | 测试宏精确率 | 测试宏召回率 |','|---|---:|---:|']
        for name,r in c.items():text.append(f"| {name} | {r['macro_precision']*100:.2f}% | {r['macro_recall']*100:.2f}% |")
        text+=['','不同模型的测试召回率仍可能不同，需同时读取这两列；逐概念 AP 和阈值见 `reports/spatial_noise_v1/concept_audit.json`。','']
    bg=Path('artifacts/background_intervention/results.json')
    text+=['## 背景互换实验','']
    if bg.exists():
        b=json.loads(bg.read_text());text+=['目的：将主体像素保留、移植异类图像背景，并先移除供体主体；真值掩膜仅构造外部干预，不作为 learned 模型推理输入。','',
            '| 方法 | 原图分类准确率 | 换背景分类准确率 | 换背景概念 mAP |','|---|---:|---:|---:|']
        for name,r in b['models'].items():text.append(f"| {name} | {r['original']['accuracy']*100:.2f}% | {r['background_swap']['accuracy']*100:.2f}% | {r['background_swap']['mAP']*100:.2f}% |")
        text+=['',f"固定分层测试子集 N={b['n']}；配对差值与区间保存在 `artifacts/background_intervention/results.json`。需同时比较 noise 和 same-checkpoint mean，不能将主体门控效应全部归给随机采样。",'']
    else:text+=['已固定构造和采样规则，尚待 GPU 队列完成。','']
    text+=['## 为什么 PiNI 可能有效','',
           '图像 CNN 将每张图每个通道 μ、σ 的 RMS 固定为 0.1，避免方差直接收缩到零。对平滑分类损失，局部近似 E[L(x+μ+σξ)]≈L(x+μ)+½∑σ²·Hii；固定噪声预算促使方差移向损失曲率较小的区域，均值承担适配。这是可验证的机制假说，曲率小不必然等同背景。', '',
           '主方法在全局视觉特征和文本提示嵌入上联合加噪，与 Figure 4 的图像 CNN 属于不同实验。论文 OxfordPets 图像 CNN 89.51%，主方法 92.97%，zero-shot 89.13%。主方法较大增益不能全部归因图像背景屏蔽。[论文](https://arxiv.org/abs/2412.10817)，[官方代码](https://github.com/hyzhang98/PiNI)。', '',
           '高方差加性噪声降低背景信噪比，但不会使背景的像素期望自动消失；本变体显式增加预测主体门控，再比较同门控的 mean-only 和 Gaussian 模型。官方历史绘图使用了不同的反归一化参数，且 uint8 热图乘 256 溢出；新图使用正确 RGB 还原和物理色标。','',
           '## 官方复现进度','']
    for kind in ['main','image']:
        p=Path('artifacts/pets_'+kind+'_seed1/results.json')
        if p.exists():
            r=json.loads(p.read_text());text+=[f"{kind}：OxfordPets 16-shot、200 epochs、seed 1；MC1 {r['MC1']['accuracy']*100:.2f}%，MC8 {r['MC8']['accuracy']*100:.2f}%，同骨干 zero-shot {r['zero_shot']['accuracy']*100:.2f}%。",'']
        else:text+=[f'{kind}：排队或训练中，尚无复现成绩。','']
    if Path('reports/pini_official/figure4_official.png').exists():
        text+=['![官方 CNN 校正可视化](reports/pini_official/figure4_official.png)','',
               '目的：检查原始 CE/RMS 约束是否自身产生背景高方差和主体均值，未使用区域监督。该图与新增带掩膜监督的 CBM 图分开解释。','']
    text+=['官方神经网络类由原始源码 AST 无修改加载，冻结 CLIP，只训练原始生成器。Pets 专用 LR 和论文 split 文件未公开，本次固定 LR=0.002 并使用官方划分算法 seed 1；不宣称匹配作者未公开配置或三种子均值。','',
           '## 实验覆盖与限制','',
           ('V1 对照：'+', '.join(pending)) if pending else 'V1 训练方式与生成器对照已完成。','',
           'V1 joint 采用单概念真实视图、其他 17 个当前干净概念概率作为 detach 锚的近似；full_joint 分类前向使用全部 18 张实际概念视图。细粒度 V2、MC 稳定性与机制归因以对应完成记录为准；不能只用图像外观判断完成。','',
           '训练可用原生概念和主体掩膜；推理没有真值掩膜。缺失的 39 张主体掩膜不参与对应定位损失/指标。所有性能数字来自固定测试集；区域版本完成适配 seed 42/43，仍共享干净初始化、划分和测试图，不包含端到端多种子或跨数据集确认。','',
           '运行协议见 [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md)。真实缓存、完整预测、checkpoint 和日志保存在本机 ignored `artifacts/`，图表及简要结论入库。']
    from .reporting import write_reports
    write_reports(rows,pending,'\n'.join(text)+'\n')
    (report/'summary.json').write_text(json.dumps(dict(rows=rows,pending=pending,paired=contrasts),indent=2))
    print('Updated EXPERIMENT_RESULT.md; completed',len(rows),'pending',len(pending),flush=True)


if __name__=='__main__':main()
