# PiNI / 空间加噪 CBM：主要实验结果

RIVAL10：固定测试集 5,286 张，冻结 ViT-L/14，18 个原生概念，分类仅经过概念概率与稀疏线性头。推理不使用真值掩膜。NEC 是每类别非零概念连接数；不包含噪声生成器参数。当前为单划分、单训练种子研究。

## 1. 分类与概念识别

**目的：**比较独立、顺序、抽样联合、完整联合训练，以及逐概念输出头；用相同初始化、训练视图预算和分类头选择规则检验噪声。

| 方法 | 分类准确率 | NEC | 概念 mAP |
|---|---:|---:|---:|
| clean matched head | 99.02% | 4.5 | 91.19% |
| mean_independent | 98.15% | 4.8 | 91.00% |
| noise_independent | 97.24% | 5.2 | 90.58% |
| mask_sequential | 98.79% | 4.4 | 91.21% |
| mean_sequential | 98.69% | 4.4 | 91.22% |
| noise_sequential | 99.30% | 4.2 | 91.07% |
| noise_joint | 99.00% | 4.2 | 91.16% |
| noise_separate_heads | 99.05% | 4.3 | 91.14% |
| local_mean | 99.13% | 4.5 | 90.81% |
| local_noise | 98.52% | 4.4 | 90.30% |
| full_joint_mean | 99.24% | 4.1 | 90.72% |
| local clean matched head | 99.45% | 4.3 | 90.65% |
| region clean matched head | 99.32% | 4.3 | 89.64% |

![主要结果](reports/spatial_noise_v1/main_results.png)

**本次结果：**顺序训练 μ+σξ 比分别训练的 μ-only 分类高 0.61 个百分点，配对 95% CI [0.38, 0.87]；概念 mAP 差为 -0.16，CI [-0.38, +0.08]，尚无整体概念识别提升证据。

使用同一个、仅由真值概念训练的分类头，增益仍为 +0.36 个百分点，CI [+0.06, +0.64]，说明收益不完全依赖分别拟合分类头。

训练/推理拆分：随机训练模型改用均值推理，仍比 μ-only 训练高 +0.44 pp，CI [+0.19, +0.70]；同检查点单次随机采样比均值高 +0.17 pp，CI [-0.02, +0.38]。这区分训练中的适配/正则化与推理中的随机扰动。

![随机部分的增量](reports/spatial_noise_v1/noise_increment.png)

local_noise_minus_local_mean：分类差 -0.61 pp，CI [-0.91, -0.30]；概念 mAP 差 -0.51 pp，CI [-0.77, -0.26]。

局部区域监督本身：正概念区域 IoU 从 35.07% 提高到 51.52%，概念 mAP 从 91.19% 变为 90.65%。这些变化发生在加噪前，应与后续随机生成器收益分开。

定位与识别的路径检查：将已经训练好的概念头分解为 CLS 和区域聚合两项，分别测量排序能力。这是冻结权重的贡献诊断，不是重新训练的独立模型。局部掩膜更准并不保证识别真的依赖那个区域。

![概念读出贡献诊断](reports/spatial_noise_v1/readout_path_diagnostic.png)

**采样稳定性：**按验证概念 mAP 选配置，再用原分类头评估八次采样均值，测试不参与选择。

| 验证选择的随机模型 | MC1 分类 | MC8 分类 | MC8 概念 mAP | 同检查点 μ 分类 |
|---|---:|---:|---:|---:|
| noise_separate_heads | 99.05% | 99.19% | 91.27% | 99.07% |

共享生成器与逐概念输出头容量不同；逐概念噪声必须与同容量 mean_separate_heads 比较，同检查点 μ 只用于推理诊断。

## 2. 是否减少背景干扰

**目的：**保留主体像素、移植异类背景并移除供体主体。真值掩膜仅构造外部干预，模型重新从换背景图像预测所有区域。

![背景互换](reports/spatial_noise_v1/background_results.png)

**结论：**主体屏蔽提高换背景后的分类与概念识别；当前顺序加噪 94.7%，μ-only 95.2%，未证实随机噪声额外提高抗背景干扰能力。

## 3. 真实噪声是否形成预期空间分布

**目的：**展示未经放大的实际 μ、σ² 和采样图，检验主体均值与背景方差；同一图像不同概念另外比较。

图中模型：`artifacts/noise_sequential`；最终模型按完整验证概念 mAP 选择，当前中期图展示指定顺序配置。

![实际加噪图](reports/spatial_noise_v1/sequential/figure4_style.png)

**结论：**全 2103 个有效验证样本，背景方差/主体方差 6.84，主体 μ/背景 μ 1.45；概念部位 μ/主体其他部位 1.09。主体/背景模式已实现，细粒度概念增强应以区域统计和识别指标同时判断。

![同图不同概念](reports/spatial_noise_v1/sequential/concept_conditioning.png)

区域读出版本仍在训练/排队：让概念识别依赖预测区域，并测试区域内固定 μ 能量。完成前不填写效果。

区域监督、主体保护和背景 σ 下限是新增方法的显式约束。μ 是门控后的偏移，总均值变化还包括背景衰减。视觉效果本身不能证明随机噪声收益。

## 4. 官方 PiNI 复现

**目的：**用官方原始模型类分别复验主方法与 Figure 4 的图像 CNN；OxfordPets 16-shot，200 轮，seed 1，完整 3,669 张测试。

| 方法 / 精度协议 | MC1 | MC8 | 同检查点 μ | 论文 |
|---|---:|---:|---:|---:|
| main · AMP / test batch 128 | 89.26% | 91.50% | 90.68% | 92.97% |
| main · 原始 FP16 / test batch 32 | 88.83% | 91.91% | 88.09% | 92.97% |
| image · AMP / test batch 128 | 89.29% | 89.51% | 88.66% | 89.51% |
| image · 原始 FP16 / test batch 32 | 89.40% | 89.40% | 88.85% | 89.51% |

![官方方法与均值对照](reports/pini_official/official_results.png)

main AMP：随机 MC8 − 分别训练的 μ-only 为 +1.47 pp，配对 CI [+0.68, +2.29]。

image AMP：随机 MC8 − 分别训练的 μ-only 为 +0.00 pp，配对 CI [-0.71, +0.71]。

**结论：**原始图像 CNN 的 AMP MC8 为 89.51%，数值上接近论文；主方法仍有复现差距。MC8 是新增诊断，不能等同论文未明确的采样协议。Pets 专用 LR 和论文 split 文件未公开，本次固定 LR=0.002，不依据测试调参。

原始 CNN 的 AMP 训练权重，全 147 个有效验证样本：背景方差/主体方差 1.13，主体 μ/背景 μ 0.94。本次 AMP 复验没有稳定产生理想的主体/背景分离；原分类 CE 和固定 RMS 不保证这种模式。见[校正后的 AMP CNN 图](reports/pini_official/figure4_official.png)。

原始默认 FP16 训练及实际 FP16 前向，全 147 个有效验证样本：背景方差/主体方差 1.07，主体 μ/背景 μ 0.99。对应[FP16 真实 CNN 图](reports/pini_official/fp16/figure4_official.png)，与 AMP 版本分开评估。

主方法在全局视觉特征和文本提示上联合加噪，Figure 4 属于单独的 RGB CNN 实验。其较大分类增益不能全部解释为图像背景屏蔽。[论文](https://arxiv.org/abs/2412.10817)，[官方代码](https://github.com/hyzhang98/PiNI)。

## 5. 当前状态与完整记录

待完成：mean_separate_heads, full_joint_noise, region_mean, region_noise。

抽样 joint 用一张真实概念视图和其余 17 个 detach 干净锚；full_joint 用全部 18 张真实视图回传分类损失，两者明确区分。完整联合采用相同 36,018 个梯度视图预算，但源图覆盖不同。

逐概念 AP、校准精确率/召回率、置信区间及语义瓶颈检查见[详细报告](EXPERIMENT_REPORT.md)、[逐概念审计](reports/spatial_noise_v1/concept_audit.json)与[协议](RESEARCH_PROTOCOL.md)。所有负结果一并保留；原始预测、真实 RGB 张量与 checkpoint 在本机 `artifacts/`。
