"""Compact main findings plus the full auditable experiment appendix."""
import json
from pathlib import Path


def read(path):
    p=Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def table(rows):
    result=['| 方法 | 分类准确率 | NEC | 概念 mAP |','|---|---:|---:|---:|']
    for r in rows:result.append(f"| {r['name']} | {r['test_accuracy']*100:.2f}% | {r['NEC']:.1f} | {r['concepts']['test']['mAP']*100:.2f}% |")
    return result


def write_reports(rows,pending,detail):
    uncertainty=read('reports/spatial_noise_v1/paired_uncertainty.json') or {'pairs':{}}
    txt=['# PiNI / 空间加噪 CBM：主要实验结果','',
         'RIVAL10：固定测试集 5,286 张，冻结 ViT-L/14，18 个原生概念，分类仅经过概念概率与稀疏线性头。推理不使用真值掩膜。NEC 是每类别非零概念连接数；不包含噪声生成器参数。当前为单划分、单训练种子研究。','',
         '## 1. 分类与概念识别','',
         '**目的：**比较独立、顺序、抽样联合、完整联合训练，以及逐概念输出头；用相同初始化、训练视图预算和分类头选择规则检验噪声。','']
    txt+=table(rows)+['','![主要结果](reports/spatial_noise_v1/main_results.png)','']
    seq=uncertainty['pairs'].get('noise_sequential_minus_mean_sequential')
    if seq:
        c=seq['classification'];ci=c['CI95_pp'];mci=seq['mAP_CI95_pp']
        txt+=[f"**已证实的结论：**顺序训练 μ+σξ 比分别训练的 μ-only 分类高 {c['difference_pp']:.2f} 个百分点，配对 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]；概念 mAP 差为 {seq['mAP_difference_pp']:+.2f}，CI [{mci[0]:+.2f}, {mci[1]:+.2f}]，尚无整体概念识别提升证据。",'']
        shared=seq.get('shared_GT_semantic_head')
        if shared:
            ci=shared['CI95_pp'];txt+=[f"使用同一个、仅由真值概念训练的分类头，增益仍为 {shared['difference_pp']:+.2f} 个百分点，CI [{ci[0]:+.2f}, {ci[1]:+.2f}]，说明收益不完全依赖分别拟合分类头。",'']
    if Path('reports/spatial_noise_v1/noise_increment.png').exists():txt+=['![随机部分的增量](reports/spatial_noise_v1/noise_increment.png)','']
    local_pairs=[(k,v) for k,v in uncertainty['pairs'].items() if k.startswith(('local_','full_joint_'))]
    for key,r in local_pairs:
        c=r['classification'];ci=c['CI95_pp'];mi=r['mAP_CI95_pp']
        txt+=[f"{key}：分类差 {c['difference_pp']:+.2f} pp，CI [{ci[0]:+.2f}, {ci[1]:+.2f}]；概念 mAP 差 {r['mAP_difference_pp']:+.2f} pp，CI [{mi[0]:+.2f}, {mi[1]:+.2f}]。",'']
    txt+=['## 2. 是否减少背景干扰','',
          '**目的：**保留主体像素、移植异类背景并移除供体主体。真值掩膜仅构造外部干预，模型重新从换背景图像预测所有区域。','']
    bg=read('artifacts/background_intervention/results.json')
    if bg:
        txt+=['![背景互换](reports/spatial_noise_v1/background_results.png)','',
              '**结论：**主体屏蔽提高换背景后的分类与概念识别；当前顺序加噪 94.7%，μ-only 95.2%，未证实随机噪声额外提高抗背景干扰能力。','']
    else:txt+=['实验尚未完成。','']
    txt+=['## 3. 真实噪声是否形成预期空间分布','',
          '**目的：**展示未经放大的实际 μ、σ² 和采样图，检验主体均值与背景方差；同一图像不同概念另外比较。','']
    selected=read('reports/spatial_noise_final/selection.json')
    gallery='reports/spatial_noise_final' if selected else 'reports/spatial_noise_v1'
    txt+=[f'![实际加噪图]({gallery}/figure4_style.png)','']
    energy=read((selected['run'] if selected else 'artifacts/noise_independent')+'/energy_audit.json')
    if energy:
        e=energy['summary'];p=e['part_localization']
        txt+=[f"**结论：**全 {e['n']} 个有效验证样本，背景方差/主体方差 {e['sigma2_bg_over_fg']:.2f}，主体 μ/背景 μ {e['mu_fg_over_bg']:.2f}；概念部位 μ/主体其他部位 {p['concept_mu_rms']/p['other_object_mu_rms']:.2f}。主体/背景模式已实现，细粒度概念增强应以区域统计和识别指标同时判断。",'']
    if Path(gallery+'/concept_conditioning.png').exists():txt+=[f'![同图不同概念]({gallery}/concept_conditioning.png)','']
    txt+=['区域监督、主体保护和背景 σ 下限是新增方法的显式约束。μ 是门控后的偏移，总均值变化还包括背景衰减。视觉效果本身不能证明随机噪声收益。','',
          '## 4. 官方 PiNI 复现','',
          '**目的：**用官方原始模型类分别复验主方法与 Figure 4 的图像 CNN；OxfordPets 16-shot，200 轮，seed 1，完整 3,669 张测试。','',
          '| 方法 / 精度协议 | MC1 | MC8 | 同检查点 μ | 论文 |','|---|---:|---:|---:|---:|']
    official_rows=[]
    for kind,paper in [('main',92.97),('image',89.51)]:
        for suffix,label in [('seed1','AMP / test batch 128'),('fp16_seed1','原始 FP16 / test batch 32')]:
            r=read('artifacts/pets_'+kind+'_'+suffix+'/results.json')
            if not r:continue
            mean=f"{r['same_checkpoint_mean']['accuracy']*100:.2f}%" if 'same_checkpoint_mean' in r else '—'
            txt.append(f"| {kind} · {label} | {r['MC1']['accuracy']*100:.2f}% | {r['MC8']['accuracy']*100:.2f}% | {mean} | {paper:.2f}% |")
            official_rows.append((kind,suffix,r))
    if Path('reports/pini_official/official_results.png').exists():txt+=['','![官方方法与均值对照](reports/pini_official/official_results.png)','']
    controls=read('reports/pini_official/paired_controls.json')
    if controls:
        for kind in ['main','image']:
            c=controls['results'].get(kind+'_seed1',{}).get('comparison',{}).get('MC8_minus_independently_trained_mean')
            if c:
                ci=c['CI95_pp'];txt+=[f"{kind} AMP：随机 MC8 − 分别训练的 μ-only 为 {c['difference_pp']:+.2f} pp，配对 CI [{ci[0]:+.2f}, {ci[1]:+.2f}]。",'']
    txt+=['**结论：**原始图像 CNN 的 AMP MC8 为 89.51%，数值上接近论文；主方法仍有复现差距。MC8 是新增诊断，不能等同论文未明确的采样协议。Pets 专用 LR 和论文 split 文件未公开，本次固定 LR=0.002，不依据测试调参。','']
    oe=read('reports/pini_official/energy_audit.json')
    if oe:
        e=oe['summary'];txt+=[f"原始 CNN 全 {e['n']} 个有效验证样本：背景方差/主体方差 {e['sigma2_bg_over_fg']:.2f}，主体 μ/背景 μ {e['mu_fg_over_bg']:.2f}。它没有稳定产生理想的主体/背景分离；原分类 CE 和固定 RMS 不能保证这种模式。见[校正后的原始 CNN 图](reports/pini_official/figure4_official.png)。",'']
    txt+=['主方法在全局视觉特征和文本提示上联合加噪，Figure 4 属于单独的 RGB CNN 实验。其较大分类增益不能全部解释为图像背景屏蔽。[论文](https://arxiv.org/abs/2412.10817)，[官方代码](https://github.com/hyzhang98/PiNI)。','',
          '## 5. 当前状态与完整记录','']
    txt+=['待完成：'+', '.join(pending)+'。' if pending else '所列表中的训练配置均已完成；MC、定位及背景审计以对应完成记录为准。','',
          '抽样 joint 用一张真实概念视图和其余 17 个 detach 干净锚；full_joint 用全部 18 张真实视图回传分类损失，两者明确区分。完整联合采用相同 36,018 个梯度视图预算，但源图覆盖不同。','',
          '逐概念 AP、校准精确率/召回率、置信区间及语义瓶颈检查见[详细报告](EXPERIMENT_REPORT.md)、[逐概念审计](reports/spatial_noise_v1/concept_audit.json)与[协议](RESEARCH_PROTOCOL.md)。所有负结果一并保留；原始预测、真实 RGB 张量与 checkpoint 在本机 `artifacts/`。']
    Path('EXPERIMENT_RESULT.md').write_text('\n'.join(txt)+'\n')
    appendix=['# PiNI 实验详细报告','',
              '主要图表与简要结论见 [EXPERIMENT_RESULT.md](EXPERIMENT_RESULT.md)。以下保留全部已完成配置及机制检查；待完成项不是成绩。','',detail,
              '## 配对不确定性与共享语义分类头','',
              '分类与全部 18 个原生概念的配对 bootstrap 结果见 [paired_uncertainty.json](reports/spatial_noise_v1/paired_uncertainty.json)。同时给出逐概念和覆盖全部概念的同步区间，避免只挑选改善项。该区间不包含训练种子变化。','',
              '![逐概念 AP](reports/spatial_noise_v1/concept_AP_differences.png)','',
              '同一个仅由真值概念训练的稀疏分类头见 [semantic_readout.json](reports/spatial_noise_v1/semantic_readout.json)；它不利用分别拟合每个模型的置信度误差，仍不保证预测误差完全不携带类别。','',
              '完整联合分类图的真实 RGB 梯度检查保存在每个 full_joint 运行的 `class_gradient_audit.json`。现代 AMP 与原始 FP16/default test batch 是两套明确协议；精度与批次对原始提示噪声的抽样都有影响，不将不同协议之间的成绩差归因于单一因素。']
    Path('EXPERIMENT_REPORT.md').write_text('\n'.join(appendix)+'\n')
