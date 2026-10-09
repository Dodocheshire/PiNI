# PiNI 实验详细报告

主要图表与简要结论见 [EXPERIMENT_RESULT.md](EXPERIMENT_RESULT.md)。以下保留全部已完成配置及机制检查；待完成项不是成绩。

# PiNI 复现与空间加噪 CBM 实验结果

本报告由已完成的实验输出生成；正在训练或评估的配置不填写成绩。研究 goal 保持 active，直到官方复现、训练方式对照与背景干预完成。

## 主要结果

目的：在真实概念监督、相同初始化和固定训练/验证/测试协议下，检验 RGB 噪声能否提高概念识别与稀疏分类。分类头仅读取 18 个概念概率，无图像特征旁路。NEC 是每类别非零概念连接数，不是模型参数数目。

| 方法 | 测试分类准确率 | NEC | 原生概念 mAP | 宏精确率 @0.5 | 宏召回率 @0.5 |
|---|---:|---:|---:|---:|---:|
| clean matched head | 99.02% | 4.50 | 91.19% | 87.50% | 89.59% |
| mean_independent | 98.15% | 4.80 | 91.00% | 88.41% | 85.55% |
| noise_independent | 97.24% | 5.20 | 90.58% | 88.36% | 82.61% |
| mask_sequential | 98.79% | 4.40 | 91.21% | 87.04% | 89.76% |
| mean_sequential | 98.69% | 4.40 | 91.22% | 86.99% | 90.29% |
| noise_sequential | 99.30% | 4.20 | 91.07% | 86.85% | 90.76% |
| noise_joint | 99.00% | 4.20 | 91.16% | 87.07% | 89.33% |
| noise_separate_heads | 99.05% | 4.30 | 91.14% | 86.86% | 90.74% |
| local clean matched head | 99.45% | 4.30 | 90.65% | 86.87% | 89.71% |

![主要结果](reports/spatial_noise_v1/main_results.png)

干净模型全 18,987 个训练样本拟合分类头：准确率 98.94%，NEC 4.50；表中使用与加噪实验相同的 6,003 个分类头训练样本。主体定位测试 IoU 73.16%，正概念区域 IoU 35.07%。

可解释性检查：将预测概率二值化后，分类准确率 96.71%，NEC 5.10；原生真值二元概念分类头为 92.45%。预测概念中的概率或系统性误差能携带额外类别信息，因此高分类成绩不能代替原生概念精度。

## 真实 Figure 4 风格可视化

目的：检查主体均值增强与背景随机扰动是否实际发生。六个样本按固定类别/概念选择验证集中首个文件，不按改善幅度挑选。μ 使用 RGB RMS，σ² 使用实际 RGB²；列内共用色标，采样不放大。

![实际加噪图](reports/spatial_noise_v1/figure4_style.png)

结论：全 2103 个有效验证样本的背景方差/主体方差为 6.73，主体 μ/背景 μ 为 4.89。概念部位 μ 与主体其他部位之比仅为 1.13，尚需改善细粒度定位。

主体区域由带监督的 ViT/text 模型预测；高背景 σ 和前景 μ 保护包含显式结构先验。μ 表示主体门控之后的概念均值偏移；相对原图的总均值变化还包括背景抑制。图形证明实际机制实现，不证明这些区域是原始 PiNI 分类 CE 自发发现的，也不能单独证明随机噪声带来性能收益。

## 随机噪声的增量作用

independent：分别训练的 μ+σξ − μ-only 为 -0.908 个百分点，配对 bootstrap 95% CI [-1.324, -0.473]。同样主体抑制和初始化，随机部分的收益必须由这组比较支持。

sequential：分别训练的 μ+σξ − μ-only 为 +0.605 个百分点，配对 bootstrap 95% CI [+0.378, +0.870]。同样主体抑制和初始化，随机部分的收益必须由这组比较支持。

noise_independent 同一检查点 MC1 − 均值推理为 -0.416 个百分点；这是推理诊断，不代替分别训练的均值对照。

noise_sequential 同一检查点 MC1 − 均值推理为 +0.170 个百分点；这是推理诊断，不代替分别训练的均值对照。

noise_joint 同一检查点 MC1 − 均值推理为 +0.284 个百分点；这是推理诊断，不代替分别训练的均值对照。

noise_separate_heads 同一检查点 MC1 − 均值推理为 -0.019 个百分点；这是推理诊断，不代替分别训练的均值对照。

固定 0.5 阈值的精确率提升可能来自召回率下降。以下阈值只由验证集正样本的第 10 百分位确定，以目标召回率 90% 比较：

| 方法 | 测试宏精确率 | 测试宏召回率 |
|---|---:|---:|
| clean | 85.27% | 89.58% |
| local_clean | 85.33% | 88.95% |
| mean_independent | 84.53% | 89.36% |
| noise_independent | 83.91% | 89.50% |
| mask_sequential | 84.90% | 88.97% |
| mean_sequential | 84.89% | 88.97% |
| noise_sequential | 85.24% | 88.51% |
| noise_joint | 85.03% | 89.06% |
| noise_separate_heads | 84.87% | 88.98% |

不同模型的测试召回率仍可能不同，需同时读取这两列；逐概念 AP 和阈值见 `reports/spatial_noise_v1/concept_audit.json`。

## 背景互换实验

目的：将主体像素保留、移植异类图像背景，并先移除供体主体；真值掩膜仅构造外部干预，不作为 learned 模型推理输入。

| 方法 | 原图分类准确率 | 换背景分类准确率 | 换背景概念 mAP |
|---|---:|---:|---:|
| clean | 98.30% | 92.30% | 88.24% |
| mask_sequential | 99.00% | 94.90% | 89.79% |
| mean_sequential | 98.10% | 95.20% | 89.63% |
| noise_sequential | 98.90% | 94.70% | 89.38% |
| noise_sequential_same_checkpoint_mean | 98.70% | 93.90% | 89.19% |

固定分层测试子集 N=1000；配对差值与区间保存在 `artifacts/background_intervention/results.json`。需同时比较 noise 和 same-checkpoint mean，不能将主体门控效应全部归给随机采样。

## 为什么 PiNI 可能有效

图像 CNN 将每张图每个通道 μ、σ 的 RMS 固定为 0.1，避免方差直接收缩到零。对平滑分类损失，局部近似 E[L(x+μ+σξ)]≈L(x+μ)+½∑σ²·Hii；固定噪声预算促使方差移向损失曲率较小的区域，均值承担适配。这是可验证的机制假说，曲率小不必然等同背景。

主方法在全局视觉特征和文本提示嵌入上联合加噪，与 Figure 4 的图像 CNN 属于不同实验。论文 OxfordPets 图像 CNN 89.51%，主方法 92.97%，zero-shot 89.13%。主方法较大增益不能全部归因图像背景屏蔽。[论文](https://arxiv.org/abs/2412.10817)，[官方代码](https://github.com/hyzhang98/PiNI)。

高方差加性噪声降低背景信噪比，但不会使背景的像素期望自动消失；本变体显式增加预测主体门控，再比较同门控的 mean-only 和 Gaussian 模型。官方历史绘图使用了不同的反归一化参数，且 uint8 热图乘 256 溢出；新图使用正确 RGB 还原和物理色标。

## 官方复现进度

main：OxfordPets 16-shot、200 epochs、seed 1；MC1 89.26%，MC8 91.50%，同骨干 zero-shot 89.13%。

image：OxfordPets 16-shot、200 epochs、seed 1；MC1 89.29%，MC8 89.51%，同骨干 zero-shot 89.13%。

![官方 CNN 校正可视化](reports/pini_official/figure4_official.png)

目的：检查原始 CE/RMS 约束是否自身产生背景高方差和主体均值，未使用区域监督。该图与新增带掩膜监督的 CBM 图分开解释。

官方神经网络类由原始源码 AST 无修改加载，冻结 CLIP，只训练原始生成器。Pets 专用 LR 和论文 split 文件未公开，本次固定 LR=0.002 并使用官方划分算法 seed 1；不宣称匹配作者未公开配置或三种子均值。

## 仍待完成

V1 对照：local_mean, local_noise, full_joint_mean, full_joint_noise

V1 joint 采用单概念真实视图、其他 17 个当前干净概念概率作为 detach 锚的近似；full_joint 分类前向使用全部 18 张实际概念视图。细粒度 V2、MC 稳定性与机制归因以对应完成记录为准；不能只用图像外观判断完成。

训练可用原生概念和主体掩膜；推理没有真值掩膜。缺失的 39 张主体掩膜不参与对应定位损失/指标。所有性能数字来自固定测试集；当前属于单划分单种子探索实验，尚无跨数据集或多种子确认。

运行协议见 [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md)。真实缓存、完整预测、checkpoint 和日志保存在本机 ignored `artifacts/`，图表及简要结论入库。

## 配对不确定性与共享语义分类头

分类与全部 18 个原生概念的配对 bootstrap 结果见 [paired_uncertainty.json](reports/spatial_noise_v1/paired_uncertainty.json)。同时给出逐概念和覆盖全部概念的同步区间，避免只挑选改善项。该区间不包含训练种子变化。

![逐概念 AP](reports/spatial_noise_v1/concept_AP_differences.png)

同一个仅由真值概念训练的稀疏分类头见 [semantic_readout.json](reports/spatial_noise_v1/semantic_readout.json)；它不利用分别拟合每个模型的置信度误差，仍不保证预测误差完全不携带类别。

完整联合分类图的真实 RGB 梯度检查保存在每个 full_joint 运行的 `class_gradient_audit.json`。现代 AMP 与原始 FP16/default test batch 是两套明确协议；精度与批次对原始提示噪声的抽样都有影响，不将不同协议之间的成绩差归因于单一因素。
