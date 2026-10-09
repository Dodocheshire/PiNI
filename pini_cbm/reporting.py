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


def write_full_reports(rows,pending,detail):
    uncertainty=read('reports/spatial_noise_v1/paired_uncertainty.json') or {'pairs':{}}
    txt=['# PiNI / 空间加噪 CBM：主要实验结果','',
         'RIVAL10：固定测试集 5,286 张，冻结 ViT-L/14，18 个原生概念，分类仅经过概念概率与稀疏线性头。推理不使用真值掩膜。NEC 是每类别非零概念连接数；不包含噪声生成器参数。固定划分和同一干净预训练初始化；适配 seed 43 复验若已完成则单独报告，不等同端到端多种子。','',
         '## 1. 分类与概念识别','',
         '**目的：**比较独立、顺序、抽样联合、完整联合训练，以及逐概念输出头；用相同初始化、训练视图预算和分类头选择规则检验噪声。','']
    txt+=table(rows)+['','![主要结果](reports/spatial_noise_v1/main_results.png)','']
    capacity=read('reports/spatial_noise_v1/semantic_capacity.json')
    if capacity:
        txt+=[f"语义信息检查：同一组真值二元概念在 {capacity['n_ambiguous_configurations']} 种配置中对应多个类别；在当前测试样本上，仅用这 18 个真值二元概念的确定性分类上限为 {capacity['empirical_true_binary_label_overlap_upper_bound']*100:.2f}%（不可避免 {capacity['unavoidable_errors_on_this_sample']} 个冲突）。这是测试标签重叠诊断，不是测试拟合模型，也不约束真实连续概念强度。99% 分类不能单独证明概念更准确，应同时检查 native AP、校准精确率和预测二值化结果。",'']
    seq=uncertainty['pairs'].get('noise_sequential_minus_mean_sequential')
    if seq:
        c=seq['classification'];ci=c['CI95_pp'];mci=seq['mAP_CI95_pp']
        txt+=[f"**本次结果：**顺序训练 μ+σξ 比分别训练的 μ-only 分类高 {c['difference_pp']:.2f} 个百分点，配对 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]；概念 mAP 差为 {seq['mAP_difference_pp']:+.2f}，CI [{mci[0]:+.2f}, {mci[1]:+.2f}]，尚无整体概念识别提升证据。",'']
        shared=seq.get('shared_GT_semantic_head')
        if shared:
            ci=shared['CI95_pp'];txt+=[f"使用同一个、仅由真值概念训练的分类头，增益仍为 {shared['difference_pp']:+.2f} 个百分点，CI [{ci[0]:+.2f}, {ci[1]:+.2f}]，说明收益不完全依赖分别拟合分类头。",'']
        train=seq.get('trained_noise_mean_inference_minus_trained_mean');sample=seq.get('same_checkpoint_sampling')
        if train and sample:
            ci=train['CI95_pp'];si=sample['CI95_pp']
            txt+=[f"训练/推理拆分：随机训练模型改用均值推理，仍比 μ-only 训练高 {train['difference_pp']:+.2f} pp，CI [{ci[0]:+.2f}, {ci[1]:+.2f}]；同检查点单次随机采样比均值高 {sample['difference_pp']:+.2f} pp，CI [{si[0]:+.2f}, {si[1]:+.2f}]。这区分训练中的适配/正则化与推理中的随机扰动。",'']
    if Path('reports/spatial_noise_v1/noise_increment.png').exists():txt+=['![随机部分的增量](reports/spatial_noise_v1/noise_increment.png)','']
    local_pairs=[(k,v) for k,v in uncertainty['pairs'].items() if k.startswith(('local_','full_joint_','region_','noise_separate_','fixed_'))]
    for key,r in local_pairs:
        c=r['classification'];ci=c['CI95_pp'];mi=r['mAP_CI95_pp']
        txt+=[f"{key}：分类差 {c['difference_pp']:+.2f} pp，CI [{ci[0]:+.2f}, {ci[1]:+.2f}]；概念 mAP 差 {r['mAP_difference_pp']:+.2f} pp，CI [{mi[0]:+.2f}, {mi[1]:+.2f}]。",'']
    local_clean=read('artifacts/spatial_local/results.json');clean=read('artifacts/spatial_clean/results.json')
    if local_clean and clean:
        l=local_clean['concepts']['test'];c=clean['concepts']['test']
        txt+=[f"局部区域监督本身：正概念区域 IoU 从 {c['concept_mask_IoU']*100:.2f}% 提高到 {l['concept_mask_IoU']*100:.2f}%，概念 mAP 从 {c['mAP']*100:.2f}% 变为 {l['mAP']*100:.2f}%。这些变化发生在加噪前，应与后续随机生成器收益分开。",'']
    if Path('reports/spatial_noise_v1/readout_path_diagnostic.png').exists():
        txt+=['定位与识别的路径检查：将已经训练好的概念头分解为 CLS 和区域聚合两项，分别测量排序能力。这是冻结权重的贡献诊断，不是重新训练的独立模型。局部掩膜更准并不保证识别真的依赖那个区域。','',
              '![概念读出贡献诊断](reports/spatial_noise_v1/readout_path_diagnostic.png)','']
    mc_rows=[r for r in rows if 'MC8' in r]
    if mc_rows:
        txt+=['**采样稳定性：**按验证概念 mAP 选配置，再用原分类头评估八次采样均值，测试不参与选择。','',
              '| 验证选择的随机模型 | MC1 分类 | MC8 分类 | MC8 概念 mAP | 同检查点 μ 分类 |','|---|---:|---:|---:|---:|']
        for r in mc_rows:
            mean=r.get('same_checkpoint_mean',{}).get('accuracy');mean='—' if mean is None else f'{mean*100:.2f}%'
            txt.append(f"| {r['name']} | {r['test_accuracy']*100:.2f}% | {r['MC8']['accuracy']*100:.2f}% | {r['MC8']['mAP']*100:.2f}% | {mean} |")
        txt+=['','共享生成器与逐概念输出头容量不同；逐概念噪声必须与同容量 mean_separate_heads 比较，同检查点 μ 只用于推理诊断。','']
    txt+=['## 2. 是否减少背景干扰','',
          '**目的：**保留主体像素、移植异类背景并移除供体主体。真值掩膜仅构造外部干预，模型重新从换背景图像预测所有区域。','']
    bg=read('artifacts/background_intervention/results.json')
    if bg:
        txt+=['![背景互换](reports/spatial_noise_v1/background_results.png)','',
              '**结论：**主体屏蔽提高换背景后的分类与概念识别；当前顺序加噪 94.7%，μ-only 95.2%，未证实随机噪声额外提高抗背景干扰能力。','']
    else:txt+=['实验尚未完成。','']
    local_bg=read('artifacts/background_intervention_local/results.json')
    if local_bg:
        txt+=['![局部与完整联合背景互换](reports/spatial_noise_v1/background_local_results.png)','',
              '| 匹配随机对照 | 换背景后分类差（噪声 − μ-only） | 配对 95% CI |','|---|---:|---:|']
        for name in ['local_noise_minus_local_mean_background_swap','full_joint_noise_minus_full_joint_mean_background_swap']:
            c=local_bg.get('direct_noise_controls',{}).get(name)
            if c:
                ci=c['CI95_pp'];txt.append(f"| {name} | {c['difference_pp']:+.2f} pp | [{ci[0]:+.2f}, {ci[1]:+.2f}] |")
        txt+=['','局部与完整联合对照在同一批 1,000 张图、相同供体背景上评估；读数以实际配对区间为准。','']
        for name in ['local_noise_minus_local_clean','full_joint_noise_minus_local_clean']:
            c=local_bg.get('matched_clean_controls',{}).get(name)
            if c:
                ci=c['CI95_pp'];txt+=[f"相对更强的局部干净基线，{name} 换背景分类差 {c['difference_pp']:+.2f} pp，CI [{ci[0]:+.2f}, {ci[1]:+.2f}]。",'']
    region_bg=read('artifacts/background_intervention_region/results.json')
    if region_bg:
        txt+=['![区域识别背景互换](reports/spatial_noise_v1/background_region_results.png)','']
        c=region_bg.get('direct_noise_controls',{}).get('region_noise_minus_region_mean_background_swap')
        if c:
            ci=c['CI95_pp'];txt+=[f"区域识别：换背景后噪声 − μ-only 分类差 {c['difference_pp']:+.2f} pp，CI [{ci[0]:+.2f}, {ci[1]:+.2f}]。",'']
        c=region_bg.get('matched_clean_controls',{}).get('region_noise_minus_region_clean')
        if c:
            ci=c['CI95_pp'];txt+=[f"区域识别：相对自身干净基线，换背景分类差 {c['difference_pp']:+.2f} pp，CI [{ci[0]:+.2f}, {ci[1]:+.2f}]。完整加噪流程与随机项的贡献需分别读取这两组比较。",'']
    fixed_bg=read('artifacts/background_intervention_fixed/results.json')
    if fixed_bg:
        txt+=['![冻结主体背景互换](reports/spatial_noise_v1/background_fixed_results.png)','']
        for key in ['fixed_noise_minus_fixed_mean_background_swap','fixed_noise_minus_fixed_noise_same_checkpoint_mean_background_swap']:
            c=fixed_bg.get('direct_noise_controls',{}).get(key)
            if c:
                ci=c['CI95_pp'];txt+=[f"{key}：分类差 {c['difference_pp']:+.2f} pp，CI [{ci[0]:+.2f}, {ci[1]:+.2f}]。",'']
    txt+=['## 3. 真实噪声是否形成预期空间分布','',
          '**目的：**展示未经放大的实际 μ、σ² 和采样图，检验主体均值与背景方差；同一图像不同概念另外比较。','']
    selected=read('reports/spatial_noise_final/selection.json')
    sequential_gallery=Path('reports/spatial_noise_v1/sequential/figure4_style.png').exists()
    gallery='reports/spatial_noise_final' if selected else 'reports/spatial_noise_v1/sequential' if sequential_gallery else 'reports/spatial_noise_v1'
    chosen_run=selected['run'] if selected else 'artifacts/noise_sequential' if sequential_gallery else 'artifacts/noise_independent'
    txt+=[f"图中模型：`{chosen_run}`；训练方式比较阶段按六个配置的完整验证概念 mAP 选择，阶段图库完成前展示指定顺序配置。区域读出版本单独展示。",'',f'![实际加噪图]({gallery}/figure4_style.png)','']
    energy=read(chosen_run+'/energy_audit.json')
    if energy:
        e=energy['summary'];p=e['part_localization']
        txt+=[f"**结论：**全 {e['n']} 个有效验证样本，背景方差/主体方差 {e['sigma2_bg_over_fg']:.2f}，主体 μ/背景 μ {e['mu_fg_over_bg']:.2f}；概念部位 μ/主体其他部位 {p['concept_mu_rms']/p['other_object_mu_rms']:.2f}。这些能量比只描述扰动分布，不能代替主体屏蔽正确性或概念识别指标。",'']
    if Path(gallery+'/concept_conditioning.png').exists():txt+=[f'![同图不同概念]({gallery}/concept_conditioning.png)','']
    region=read('artifacts/region_noise/results.json')
    if region:
        txt+=[f"区域读出版本：概念概率由预测主体×概念区域的 CLIP patch 聚合计算，直接 CLS 特征读出关闭；μ 使用预测区域内固定 RMS 预算 {region['mean_budget']:.5f} RGB（由 PiNI 的 0.1 CLIP 单位换算，训练前固定）。这同时改变读出结构和 μ 约束，应与它自己的 region_mean 对照比较。",'']
        for filename,label in [('figure4_style.png','区域识别实际噪声'),('concept_conditioning.png','区域识别同图不同概念')]:
            path='reports/spatial_noise_region/'+filename
            if Path(path).exists():txt+=[f'![{label}]({path})','']
    elif 'region_noise' in pending:
        txt+=['区域读出版本仍在训练/排队：让概念识别依赖预测区域，并测试区域内固定 μ 能量。完成前不填写效果。','']
    drift=read('reports/spatial_noise_v1/foreground_drift.json')
    if drift and 'noise_sequential' in drift['models']:
        d=drift['models'];s=d['noise_sequential'];c=d['spatial_clean']
        txt+=[f"主体屏蔽漂移：顺序随机模型的验证主体 IoU 从 {c['foreground_IoU']*100:.2f}% 降为 {s['foreground_IoU']*100:.2f}%，真实背景上平均前景概率 {s['foreground_probability_on_true_background']*100:.2f}%。提高前景概率会降低该处 σ，所以分类目标可能通过扩大前景区减弱噪声。这解释了为何漂亮的方差比不能证明背景被正确屏蔽。",'',
              '![主体屏蔽器漂移](reports/spatial_noise_v1/foreground_drift.png)','']
    fixed=read('artifacts/fixed_noise/results.json')
    if fixed:
        a=fixed['blocker_audit'];txt+=[f"冻结屏蔽器版本：独立的预训练主体预测器保持冻结，预测主体同时约束 RGB 和原生概念区域读出；18 个概念分别输出 μ/σ。σ 的全图 RGB RMS 固定为 {fixed['sigma_budget']:.5f}。主体权重变化 {a['max_blocker_weight_change']:.1f}、固定验证主体概率变化 {a['max_fixed_validation_mask_change']:.1f}，不能通过重标背景来降低扰动。",'']
        for filename,label in [('figure4_style.png','冻结主体实际加噪'),('concept_conditioning.png','冻结主体同图不同概念')]:
            path='reports/spatial_noise_fixed/'+filename
            if Path(path).exists():txt+=[f'![{label}]({path})','']
        energy=read('artifacts/fixed_noise/energy_audit.json')
        localization=read('artifacts/fixed_noise/localization_audit.json')
        if energy:
            e=energy['summary'];txt+=[f"冻结主体全 {e['n']} 张有效验证图：背景/主体方差 {e['sigma2_bg_over_fg']:.2f}，主体/背景 μ {e['mu_fg_over_bg']:.2f}。这些是实际 RGB 能量分布，包含显式空间先验。",'']
        if localization:
            s=localization['summary'];txt+=[f"全部 {s['n_positive_pairs']} 个正概念/图像对，原始概念区域 IoU {s['positive_concept_IoU']*100:.2f}%；实际主体×概念读出区域 IoU {s['positive_readout_region_IoU']*100:.2f}%。其中 {s['n_local_pairs']} 个局部部位的 μ/主体其他部分为 {s['local_part_mu_over_other_object']:.2f}。",'']
    elif 'fixed_noise' in pending:
        txt+=['冻结主体／固定噪声预算的匹配 mean/noise 对照正在排队；该版本针对已观察到的屏蔽器漂移，完成前不填写成绩。','']
    txt+=['区域监督、主体保护和背景 σ 下限是新增方法的显式约束。μ 是门控后的偏移，总均值变化还包括背景衰减。视觉效果本身不能证明随机噪声收益。','']
    influence=[]
    for name in ['noise_sequential','region_noise','fixed_noise']:
        r=read('reports/spatial_noise_v1/'+name+'_noise_region_intervention.json')
        if r:influence.append((name,r))
    if influence:
        txt+=['额外检查背景噪声是否仍影响识别：用同一个随机样本分别只扰动前景、只扰动背景、扰动全图。真值主体仅构造外部诊断；学习的生成器保持原输入。','',
              '![噪声区域影响](reports/spatial_noise_v1/noise_region_influence.png)','']
        for name,r in influence:
            txt+=[f"{name}：固定 {r['n_validation_images']} 张验证图、全部 18 个概念、{r['draws']} 次配对采样；背景/前景单独扰动引起的原生 logit 方差比 {r['background_over_foreground_output_variance']:.2f}。该比值同时受注入能量和模型敏感性影响，不是测试准确率；高背景 σ 本身不保证识别忽略背景。",'']
    txt+=['## 4. 官方 PiNI 复现','',
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
        main=controls['results'].get('main_seed1',{}).get('comparison',{})
        mc=main.get('MC8_minus_MC1');single=main.get('MC1_minus_independently_trained_mean')
        if mc and single:
            ci=mc['CI95_pp'];si=single['CI95_pp']
            txt+=[f"为什么主方法看起来收益较大：本次 AMP 主方法八次采样比单次高 {mc['difference_pp']:+.2f} pp，CI [{ci[0]:+.2f}, {ci[1]:+.2f}]；单次随机预测比独立 μ-only 为 {single['difference_pp']:+.2f} pp，CI [{si[0]:+.2f}, {si[1]:+.2f}]。当前最明确的收益包含采样平均，不能全部归因单次加噪，更不能归因图像背景屏蔽。视觉特征与提示扰动的各自贡献尚未分离。",'']
    txt+=['**结论：**原始图像 CNN 的 AMP MC8 为 89.51%，数值上接近论文；主方法仍有复现差距。MC8 是新增诊断，不能等同论文未明确的采样协议。Pets 专用 LR 和论文 split 文件未公开，本次固定 LR=0.002，不依据测试调参。','']
    oe=read('reports/pini_official/energy_audit.json')
    if oe:
        e=oe['summary'];txt+=[f"原始 CNN 的 AMP 训练权重，全 {e['n']} 个有效验证样本：背景方差/主体方差 {e['sigma2_bg_over_fg']:.2f}，主体 μ/背景 μ {e['mu_fg_over_bg']:.2f}。本次 AMP 复验没有稳定产生理想的主体/背景分离；原分类 CE 和固定 RMS 不保证这种模式。见[校正后的 AMP CNN 图](reports/pini_official/figure4_official.png)。",'']
    canonical=read('reports/pini_official/fp16/energy_audit.json')
    if canonical:
        e=canonical['summary'];txt+=[f"原始默认 FP16 训练及实际 FP16 前向，全 {e['n']} 个有效验证样本：背景方差/主体方差 {e['sigma2_bg_over_fg']:.2f}，主体 μ/背景 μ {e['mu_fg_over_bg']:.2f}。对应[FP16 真实 CNN 图](reports/pini_official/fp16/figure4_official.png)，与 AMP 版本分开评估。",'']
    txt+=['主方法在全局视觉特征和文本提示上联合加噪，Figure 4 属于单独的 RGB CNN 实验。其较大分类增益不能全部解释为图像背景屏蔽。[论文](https://arxiv.org/abs/2412.10817)，[官方代码](https://github.com/hyzhang98/PiNI)。','',
          'LF-Pi 的先前大幅增益也需要分离训练适配与随机项：匹配 CIFAR-10 全局对照中，确定性均值 88.44%、随机 MC8 88.42%，NEC 分别 21.5 与 21.9；这支持“任务投影/均值适配压缩了分类所需连接”，未证明背景随机屏蔽。真实图像、人工概念与背景干预见 [LF-Pi 机制报告](https://github.com/Dodocheshire/LF-Pi/blob/main/NOISE_MECHANISM_REPORT.md)。','',
          '## 5. 当前状态与完整记录','']
    txt+=['待完成：'+', '.join(pending)+'。' if pending else '所列表中的训练配置均已完成；MC、定位及背景审计以对应完成记录为准。','',
          '抽样 joint 用一张真实概念视图和其余 17 个 detach 干净锚；full_joint 用全部 18 张真实视图回传分类损失，两者明确区分。完整联合采用相同 36,018 个梯度视图预算，但源图覆盖不同。','',
          '逐概念 AP、校准精确率/召回率、置信区间及语义瓶颈检查见[详细报告](EXPERIMENT_REPORT.md)、[逐概念审计](reports/spatial_noise_v1/concept_audit.json)与[协议](RESEARCH_PROTOCOL.md)。所有负结果一并保留；原始预测、真实 RGB 张量与 checkpoint 在本机 `artifacts/`。']
    Path('EXPERIMENT_RESULT.md').write_text('\n'.join(txt).rstrip()+'\n')
    appendix=['# PiNI 实验详细报告','',
              '主要图表与简要结论见 [EXPERIMENT_RESULT.md](EXPERIMENT_RESULT.md)。以下保留全部已完成配置及机制检查；待完成项不是成绩。','',detail,
              '## 配对不确定性与共享语义分类头','',
              '分类与全部 18 个原生概念的配对 bootstrap 结果见 [paired_uncertainty.json](reports/spatial_noise_v1/paired_uncertainty.json)。同时给出逐概念和覆盖全部概念的同步区间，避免只挑选改善项。该区间不包含训练种子变化。','',
              '![逐概念 AP](reports/spatial_noise_v1/concept_AP_differences.png)','',
              '同一个仅由真值概念训练的稀疏分类头见 [semantic_readout.json](reports/spatial_noise_v1/semantic_readout.json)；它不利用分别拟合每个模型的置信度误差，仍不保证预测误差完全不携带类别。','',
              '完整联合分类图的真实 RGB 梯度检查保存在每个 full_joint 运行的 `class_gradient_audit.json`。现代 AMP 与原始 FP16/default test batch 是两套明确协议；精度与批次对原始提示噪声的抽样都有影响，不将不同协议之间的成绩差归因于单一因素。']
    Path('EXPERIMENT_REPORT.md').write_text('\n'.join(appendix)+'\n')


def primary_figures(rows):
    """A fixed, short comparison list; no selection by test score."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    chosen=['clean matched head','mean_sequential','noise_sequential','region clean matched head','region_mean','region_noise','region_noise_mean_inference']
    labels=['Original clean','Sequential mean','Sequential Gaussian','Region clean','Region mean','Region Gaussian','Region noise-trained / mean']
    candidates=list(rows)
    region=next((r for r in rows if r['name']=='region_noise'),None)
    if region and 'same_checkpoint_mean' in region:
        mean=region['same_checkpoint_mean']
        candidates.append(dict(name='region_noise_mean_inference',test_accuracy=mean['accuracy'],NEC=region['NEC'],concepts={'test':mean}))
    selected=[(label,next(r for r in candidates if r['name']==name)) for name,label in zip(chosen,labels) if any(r['name']==name for r in candidates)]
    colors=['#CC79A7' if 'mean_inference' in r['name'] else '#E69F00' if 'noise' in r['name'] else '#009E73' if 'mean' in r['name'] else '#0072B2' for _,r in selected]
    fig,axes=plt.subplots(1,3,figsize=(11,4.5),layout='constrained');y=np.arange(len(selected))
    for ax,key,title in zip(axes,['accuracy','mAP','NEC'],['Class accuracy (%)','Native concept mAP (%)','NEC / class']):
        values=[r['test_accuracy']*100 if key=='accuracy' else r['concepts']['test']['mAP']*100 if key=='mAP' else r['NEC'] for _,r in selected]
        ax.scatter(values,y,c=colors,s=70);ax.set_yticks(y,[label for label,_ in selected] if ax is axes[0] else ['']*len(y));ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_title(title,fontsize=10)
        for j,v in enumerate(values):ax.annotate(f'{v:.2f}',(v,j),xytext=(5,0),textcoords='offset points',fontsize=9)
        ax.set_xlim(0,6 if key=='NEC' else 100) if key=='NEC' else ax.set_xlim(np.floor(min(values))-1,min(100,np.ceil(max(values))+1))
    fig.suptitle('RIVAL10: 5,286 test images; 18 native concepts; matched 6,003-image heads\nBlue: clean | Green: trained mean | Orange: Gaussian | Purple: noise-trained, mean inference',fontsize=10)
    out=Path('reports/spatial_noise_v1')
    for ext in ['png','svg','pdf']:fig.savefig(out/('key_results.'+ext),dpi=180)
    plt.close(fig)
    return [r for _,r in selected]


def write_reports(rows,pending,detail):
    # Preserve the extensive attribution record in the appendix, while the
    # entry report remains a small number of main results and conclusions.
    write_full_reports(rows,pending,detail)
    full=Path('EXPERIMENT_RESULT.md').read_text()
    appendix=Path('EXPERIMENT_REPORT.md')
    appendix.write_text(appendix.read_text()+'\n## 补充机制与官方复验\n\n'+full.split('\n',1)[1])
    primary=primary_figures(rows)
    uncertainty=read('reports/spatial_noise_v1/paired_uncertainty.json') or {'pairs':{}}
    seq=uncertainty['pairs'].get('noise_sequential_minus_mean_sequential')
    fixed=read('artifacts/fixed_noise/results.json')
    native_bg=read('reports/spatial_noise_v1/background_native_uncertainty.json') or {'pairs':{}}
    regional_native=native_bg['pairs'].get('region_noise_minus_region_mean')
    repeat_native=native_bg['pairs'].get('region_seed43_noise_minus_region_seed43_mean')
    two_seed=read('reports/spatial_noise_v1/two_seed_summary.json')
    lead='区域加噪 CBM 已实现主体保护与概念部位增强；常规测试分类约 99%、NEC 约 4.4。'
    if two_seed:
        lead+='两个固定适配模型的平均结果支持“训练加噪、推理取均值”改善换背景分类和概念 mAP；固定阈值概念精确率与推理继续采样的额外收益尚未得到支持。'
    elif regional_native and regional_native['mAP_CI95_pp'][0]>0:
        if repeat_native and repeat_native['mAP_CI95_pp'][0]>0:
            lead+='区域读出版本在两个适配种子的换背景测试中均出现概念 mAP 收益。'
        elif repeat_native and repeat_native['mAP_difference_pp']<=0:
            lead+='区域读出版本第一适配种子有换背景概念收益，第二种子未复现，尚不能认为稳定。'
        else:lead+='区域读出版本第一适配种子的换背景测试出现概念 mAP 收益，主要来自噪声训练后的适配。'
    else:lead+='随机项的抗背景干扰收益尚未得到支持。'
    txt=['# PiNI：主要实验结果','',
         '固定 RIVAL10 测试集 5,286 张，冻结 ViT-L/14，18 个原生概念；分类仅经过概念概率与稀疏头。推理不使用真值掩膜。主表为适配 seed 42；第二种子复验单独报告，共享划分和干净初始化。区间只衡量配对测试样本不确定性。','',
         '**最终判断：**'+lead,'',
         '## 1. 主要性能','',
         '**目的：**同时检查分类、连接稀疏度和原生概念识别，避免只看分类分数。','']
    txt+=table(primary)+['','![主要性能](reports/spatial_noise_v1/key_results.png)','']
    if seq:
        c=seq['classification'];ci=c['CI95_pp'];mi=seq['mAP_CI95_pp']
        txt+=[f"**结论：**顺序噪声相对匹配均值模型，分类差 {c['difference_pp']:+.2f} pp，95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]；概念 mAP 差 {seq['mAP_difference_pp']:+.2f} pp，CI [{mi[0]:+.2f}, {mi[1]:+.2f}]。分类收益主要来自训练后的适配；同检查点单次采样的额外收益尚不明确。",'']
    pair=uncertainty['pairs'].get('fixed_noise_minus_fixed_mean')
    if pair:
        c=pair['classification'];ci=c['CI95_pp'];mi=pair['mAP_CI95_pp']
        txt+=[f"冻结主体匹配对照：噪声 − 均值分类差 {c['difference_pp']:+.2f} pp，CI [{ci[0]:+.2f}, {ci[1]:+.2f}]；概念 mAP 差 {pair['mAP_difference_pp']:+.2f} pp，CI [{mi[0]:+.2f}, {mi[1]:+.2f}]。",'']
    txt+=['## 2. 训练方式与随机项','',
          '**目的：**比较独立、顺序、完整联合及逐概念生成器；每组与自身同容量的均值模型配对。','',
          '![噪声增量与区间](reports/spatial_noise_v1/noise_increment.png)','',
          '**结论：**已完成的训练方式比较没有显示稳定的整体概念 mAP 增益。逐概念输出头能训练，但并未解决随机项的语义收益问题。完整联合的分类损失确实经过全部 18 张真实加噪 RGB 视图，仍不能靠联合训练本身保证有效。','',
          '## 3. 背景干扰','',
          '**目的：**保留主体、移植异类背景，在同一批 1,000 张测试图上区分屏蔽/均值适配和随机噪声的贡献。','']
    bg=read('artifacts/background_intervention_local/results.json')
    strong=read('artifacts/background_clean_models/results.json')
    if bg and strong:
        clean=strong['models']['local_clean']['background_swap']['accuracy']*100
        m=bg['models']['full_joint_mean']['background_swap']['accuracy']*100;n=bg['models']['full_joint_noise']['background_swap']['accuracy']*100
        chart='two_seed_background' if two_seed else 'background_noise_increment'
        txt+=[f'![背景干预增量](reports/spatial_noise_v1/{chart}.png)','',
              f'**结论：**更强的局部干净基线 {clean:.1f}%，完整联合均值 {m:.1f}%，随机噪声 {n:.1f}%。抗背景干扰的收益已存在于确定性模型，不能全部归给随机项。','']
    if two_seed:
        txt+=['**两次适配的平均结果：**同一批 1,000 张图与供体，分别计算指标再平均；不集成预测。','',
              '| 训练 / 推理 | 换背景分类 | 换背景概念 mAP |','|---|---:|---:|']
        for label,key in [('均值训练 / 均值','mean_trained'),('噪声训练 / 单次采样','noise_trained_MC1'),('噪声训练 / 均值','noise_trained_mean_inference')]:
            m=two_seed[key];txt.append(f"| {label} | {m['background_accuracy']*100:.2f}% | {m['background_mAP']*100:.2f}% |")
        p=two_seed['pairs']['noise_trained_mean_inference_minus_mean_trained'];c=p['classification'];ci=c['CI95_pp'];mi=p['mAP_CI95_pp']
        sample=two_seed['pairs']['noise_trained_MC1_minus_mean_trained'];si=sample['classification']['CI95_pp'];sm=sample['mAP_CI95_pp']
        txt+=['',f"**训练噪声的证据：**均值推理相对均值训练，分类 {c['difference_pp']:+.2f} pp，95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]；概念 mAP {p['mAP_difference_pp']:+.2f} pp，CI [{mi[0]:+.2f}, {mi[1]:+.2f}]。噪声 MC1 相对均值训练的分类区间 [{si[0]:+.2f}, {si[1]:+.2f}]、mAP 区间 [{sm[0]:+.2f}, {sm[1]:+.2f}] 均包含零。区间条件于这两个已训练模型，只反映图像抽样，不证明训练种子总体或其他数据集的稳定收益。",'']
        appendix.write_text((appendix.read_text()+'\n## 两次固定适配的背景收益\n\n'+'\n'.join(txt[txt.index('**两次适配的平均结果：**同一批 1,000 张图与供体，分别计算指标再平均；不集成预测。'):])).rstrip()+'\n')
    if regional_native:
        mi=regional_native['mAP_CI95_pp'];trained=native_bg['pairs'].get('region_noise_same_checkpoint_mean_minus_region_mean')
        txt+=[f"区域读出版本在换背景后的概念 mAP 为 89.01%，均值对照 87.92%；差 {regional_native['mAP_difference_pp']:+.2f} pp，CI [{mi[0]:+.2f}, {mi[1]:+.2f}]。这是当前支持噪声训练提高背景变化下概念识别的证据，但并未同时提高常规测试分类与概念 mAP。",'']
        if trained:
            ci=trained['mAP_CI95_pp'];txt+=[f"随机训练模型改用均值推理后，背景概念 mAP 增量仍为 {trained['mAP_difference_pp']:+.2f} pp，CI [{ci[0]:+.2f}, {ci[1]:+.2f}]；收益不要求推理继续随机采样。",'']
    precision=read('reports/spatial_noise_v1/background_precision.json')
    if precision and 'region_noise_minus_region_mean' in precision['pairs']:
        p=precision['models'];c=precision['pairs']['region_noise_minus_region_mean']['metrics'];ci=c['threshold05_precision']['CI95_pp'];r=c['threshold05_recall']
        txt+=[f"精确率与召回率分开读取：固定 0.5 阈值下，区域噪声/均值模型精确率 {p['region_noise']['threshold05']['macro_precision']*100:.2f}% / {p['region_mean']['threshold05']['macro_precision']*100:.2f}%，差的 CI [{ci[0]:+.2f}, {ci[1]:+.2f}]；召回率差 {r['difference_pp']:+.2f} pp。当前收益不能表述为显著提高固定阈值精确率。[完整 PR 诊断](reports/spatial_noise_v1/background_precision.png)同时显示固定召回率的排序表现，曲线使用测试真值做指标评估，不提供测试选阈值。",'']
    fixed_bg=read('artifacts/background_intervention_fixed/results.json')
    if fixed_bg:
        c=fixed_bg.get('direct_noise_controls',{}).get('fixed_noise_minus_fixed_mean_background_swap')
        if c:
            ci=c['CI95_pp'];txt+=[f"冻结主体的换背景随机增量 {c['difference_pp']:+.2f} pp，CI [{ci[0]:+.2f}, {ci[1]:+.2f}]。",'']
    if repeat_native:
        mi=repeat_native['mAP_CI95_pp'];bg43=read('artifacts/background_intervention_region_seed43/results.json')
        txt+=[f"**第二适配种子复验：**使用同样的区域模型、超参数、干净初始化、测试图与供体，仅改变适配训练随机种子；换背景概念 mAP 差 {repeat_native['mAP_difference_pp']:+.2f} pp，CI [{mi[0]:+.2f}, {mi[1]:+.2f}]。这不是端到端独立预训练的多种子结果。",'']
        if bg43:
            c=bg43.get('direct_noise_controls',{}).get('region_seed43_noise_minus_region_seed43_mean_background_swap')
            if c:
                ci=c['CI95_pp'];txt+=[f"第二适配种子的换背景分类差 {c['difference_pp']:+.2f} pp，CI [{ci[0]:+.2f}, {ci[1]:+.2f}]。",'']
    txt+=['## 4. Figure 4 风格真实分布','',
          '**目的：**在预先选定验证样本上显示实际 μ、σ² 和采样 RGB；所有图使用物理单位，采样不放大。','']
    drift=read('reports/spatial_noise_v1/foreground_drift.json')
    if drift:
        d=drift['models'];s=d.get('noise_sequential')
        if s:txt+=[f"发现的失效机制：原顺序随机模型将 {s['false_foreground_fraction_of_background']*100:.1f}% 的真实背景判为主体，主体 IoU 为 {s['foreground_IoU']*100:.1f}%。背景方差较高的热图并不能证明屏蔽器正确。",'']
    gallery='reports/spatial_noise_region/figure4_main.png' if Path('reports/spatial_noise_region/figure4_main.png').exists() else 'reports/spatial_noise_region/figure4_style.png'
    if Path(gallery).exists():txt+=[f'![区域模型真实加噪]({gallery})','']
    local=read('artifacts/region_noise/localization_audit.json')
    if local:
        s=local['summary'];txt+=[f"区域模型全部 {s['n_positive_pairs']} 个正概念/图像对：背景/主体方差 {s['sigma2_bg_over_fg']:.2f}，主体/背景 μ {s['mu_fg_over_bg']:.2f}；{s['n_local_pairs']} 个局部部位的 μ/主体其他部分为 {s['local_part_mu_over_other_object']:.2f}。图中的主体/背景与概念部位分离有全验证集统计支持。",'']
    repeat_local=read('artifacts/region_seed43_noise/localization_audit.json')
    if repeat_local:
        s=repeat_local['summary'];txt+=[f"第二适配种子对应全验证集比值为 {s['sigma2_bg_over_fg']:.2f} / {s['mu_fg_over_bg']:.2f} / {s['local_part_mu_over_other_object']:.2f}；实际主体×概念读出区域 IoU 为 {s['positive_readout_region_IoU']*100:.2f}%。空间模式可复现，细粒度定位仍有限。",'']
    txt+=['主图取原预定六个样本中的前三个，便于阅读；[完整六样本](reports/spatial_noise_region/figure4_style.png)与[同图不同概念](reports/spatial_noise_region/concept_conditioning.png)保留小部位漏定位等失败情况。','']
    if fixed:
        path='reports/spatial_noise_fixed/figure4_style.png'
        if Path(path).exists():txt+=[f'冻结主体版本实际图见[对照图库]({path})；该图与性能一起读取。','']
        energy=read('artifacts/fixed_noise/energy_audit.json')
        if energy:
            e=energy['summary'];txt+=[f"**结论：**冻结主体后，权重与固定验证主体概率变化均为零；全 {e['n']} 张有效验证图的背景/主体方差 {e['sigma2_bg_over_fg']:.2f}，主体/背景 μ {e['mu_fg_over_bg']:.2f}。空间分离来自明确先验，实际收益仍需读取前述匹配对照。",'']
    else:
        txt+=['冻结主体匹配 mean/noise 对照正在完成，结束前不填写效果。','']
    txt+=['## 5. 官方 PiNI 与 LF-Pi 的解释','',
          '**目的：**区分官方主方法、图像 CNN 和自身 LF-Pi 中训练适配、随机采样的贡献。','',
          '官方 OxfordPets AMP 主方法：单次 89.26%，MC8 91.50%，独立均值模型 90.02%；图像 CNN 的 MC8 89.51%，与独立均值模型相同。默认 FP16 图像 CNN 在全部 147 张有效验证图上的背景/主体方差比仅 1.07，未复现稳定的理想空间分离。MC8 是新增采样诊断，不能当作论文原始协议的精确复现。','',
          '**结论：**本次 PiNI 主方法收益包含采样平均；LF-Pi 匹配对照中均值 88.44%、随机 MC8 88.42%，大幅提升与 NEC 下降主要支持任务投影/均值适配，尚不能解释为随机噪声清除了背景。','',
          '注意：这 18 个真值二元概念在当前测试样本上的标签重叠上限为 93.40%；它不约束真实连续语义强度，但说明 99% 分类成绩不能单独证明概念准确或纯净。','',
          '完整负结果、逐概念精确率/召回率、配对区间、定位、MC 和复现协议见 [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) 与 [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md)；实现、推理命令和逐项目标核验见 [GOAL_VERIFICATION.md](GOAL_VERIFICATION.md)。','']
    if pending:txt+=['待完成训练：'+', '.join(pending)+'。','']
    elif fixed and not fixed_bg:txt+=['训练已完成；冻结主体的背景、定位及噪声影响审计仍在队列中。','']
    Path('EXPERIMENT_RESULT.md').write_text('\n'.join(txt).rstrip()+'\n')
