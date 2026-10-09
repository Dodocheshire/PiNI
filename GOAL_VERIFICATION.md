# PiNI 空间加噪 CBM：目标核验

本轮形成了可运行的概念条件 RGB 加噪 CBM，并完成官方复验、匹配训练对照、两次区域适配、背景干预及真实可视化。达到的结果是：主体保护、概念部位均值偏移与高背景方差能够同时呈现；分类约 99%、每类别约 4.4 个概念连接；训练噪声在本次换背景测试中有分类与概念 mAP 收益。固定阈值精确率、推理继续采样及跨数据集的普遍收益仍未得到支持。

最终核验从保存的完整预测重算全部 18 个适配配置的分类准确率与概念 mAP，并检查真实梯度、两次空间统计、RGB 推理、源张量与检查点 SHA256；[核验结果与来源指纹](reports/spatial_noise_v1/goal_evidence_verification.json)全部通过。

## 逐项证据

| 用户要求 | 实际实现与核验 | 证据 |
|---|---|---|
| 自有 PiNI submodule 与云端仓库 | `Dodocheshire/PiNI` 私有仓库，保留官方完整历史；父仓库登记远端与 commit | [仓库](https://github.com/Dodocheshire/PiNI)、父仓库 `submodules.json` |
| 复验 PiNI、解释收益 | OxfordPets 16-shot、200 轮、完整 3,669 图；分别运行主方法、图像 CNN、独立均值和同检查点 MC/均值 | [官方复验记录](reports/pini_official)、[协议](RESEARCH_PROTOCOL.md) |
| Figure 4 风格真实加噪 | 预定样本、真实训练权重与 RGB 张量；μ 的三通道 RMS、σ² 和未经放大的实际采样，统一物理色标 | [主图](reports/spatial_noise_region/figure4_main.png)、[张量来源与 SHA256](reports/spatial_noise_region/figure4_main.json)、[完整六样本及失败例](reports/spatial_noise_region/figure4_style.png) |
| 背景高方差、主体/概念高均值偏移 | 两次适配的全验证集 9,022 个有效正概念对；背景/主体方差 6.44 / 7.61，主体/背景 μ 11.88 / 10.57；4,080 个局部部位 μ/主体其他部分 10.39 / 10.56 | [全验证集汇总与逐概念统计](reports/spatial_noise_region/two_seed_localization.json)、[主报告](EXPERIMENT_RESULT.md)；全部逐图行保留在本机对应 `artifacts/` |
| 独立、顺序、联合训练 | 冻结识别器独立适配、顺序适配、抽样联合及完整联合；完整联合分类损失经过全部 18 张真实 RGB 概念视图 | [全部配置与负结果](EXPERIMENT_REPORT.md)、[配对区间](reports/spatial_noise_v1/paired_uncertainty.json) |
| 每概念生成器、ViT 与概念文本 | 冻结 ViT-L/14、18 个命名概念文本、预测区域注意力；共享文本条件生成器与 18 个独立小输出头配对比较，共享 ViT 和空间输入分支 | [模型](pini_cbm/models.py)、[训练](pini_cbm/train_noise.py)；逐概念输出头并非 18 个完全独立的 ViT |
| 先训练背景屏蔽器，再训练概念噪声 | 区域预训练后适配；另复制并冻结完整主体预测分支，供 RGB 门控和概念读出使用；213,505 个参数、固定验证主体概率变化均为零 | [冻结契约核验](reports/spatial_noise_v1/fixed_blocker_contract.json)、[冻结版本图库](reports/spatial_noise_fixed/figure4_style.png) |
| 真实 CBM、良好分类、低 NEC | 分类头仅接收 18 个原生概念概率。区域噪声训练后均值推理，seed 42/43 分类 99.09% / 99.13%，NEC 4.5 / 4.4；无图像向分类头的直接旁路 | [主表](EXPERIMENT_RESULT.md)、[RGB 独立推理核验](reports/spatial_noise_v1/RGB_only_inference.json) |
| 加噪确实有用 | 同容量的区域均值/噪声训练，保持同一 1,000 图与异类背景供体。两个固定适配模型的平均指标中，噪声训练后均值推理相对均值训练：背景分类 +1.00 pp，图像 bootstrap CI [+0.35,+1.65]；原生概念 mAP +0.75 pp，CI [+0.29,+1.16] | [配对图](reports/spatial_noise_v1/two_seed_background.png)、[全部三种对比和统计](reports/spatial_noise_v1/two_seed_summary.json) |
| 排查 LF-Pi 的大幅提升和 NEC 下降 | 匹配 CIFAR-10 对照中，均值 88.44%、噪声 MC8 88.42%；更支持任务投影/均值适配，未证明随机噪声去背景 | [LF-Pi 机制报告](https://github.com/Dodocheshire/LF-Pi/blob/main/NOISE_MECHANISM_REPORT.md) |
| SOTA 作为参考 | 已复验 SALF-CBM（CVPR 2025）官方 ImageNet 模型；自身实验使用有概念/区域监督的 RIVAL10，单独报告 | [SALF-Pi 报告](https://github.com/Dodocheshire/SALF-Pi/blob/main/EXPERIMENT_REPORT.md)；不同数据集与监督条件不作排名 |

## 支持范围

正证据来自训练中的真实 Gaussian RGB 扰动及其适配效果，不能表述为部署时必须继续随机采样。两次适配共享干净预训练与划分；置信区间只反映固定模型上的配对图像抽样，不包含训练种子总体变化。均值推理沿用噪声训练的稀疏头，没有按背景测试重拟合；这是机制诊断，后续独立数据应预先固定推理协议。

主体门控、区域标注及 σ 空间约束是明确先验；高背景方差并非原始 PiNI 分类损失自发保证。实际主体×概念读出区域 IoU 约 47%，仍有小部位漏定位。固定 0.5 阈值概念精确率未明确提高，原图测试也没有稳定的整体概念 mAP 增益。真值二元概念的样本标签重叠上限 93.40% 提醒我们，99% 分类可利用概念置信度或系统性误差，不能单独证明纯净语义。

官方 PiNI 已实际复验，但主方法未精确达到论文报告的 92.97%；默认 FP16 主方法 MC8 为 91.91%，MC8 又是新增诊断。原图像 CNN 的全验证集方差/均值空间比接近 1，本次没有复现其理想自发分离。实现所需视觉模式的是单独开发的监督空间 CBM，不能冒充原方法的成功精确复现。

## 本机推理

在 `/home/pengcheng/Pi-CBM/PiNI` 执行；训练权重、原始预测和缓存保留在本机 ignored `artifacts/`，未上传数据或 checkpoint。GitHub 中有源码、结果与图表，单独克隆仓库不包含训练权重。

```bash
/home/pengcheng/miniconda3/envs/pi-cbm/bin/python -m pini_cbm.infer /absolute/path/image.jpg --run artifacts/region_noise --mode noise --draws 1 --out prediction.json
/home/pengcheng/miniconda3/envs/pi-cbm/bin/python -m pini_cbm.infer /absolute/path/image.jpg --run artifacts/region_noise --mode mean --out prediction_mean.json
```

默认保留随机推理协议；`mean` 展示同检查点的确定性推理。入口只接收 RGB 文件，输出 18 个命名概念概率和分类；不加载 RIVAL10 标注、掩膜或 token 缓存。内部仍需下载或命中冻结 CLIP ViT-L/14 权重缓存。NEC 衡量分类连接，完整概念推理仍需原图编码加 18 次视图编码；不是低计算量模型。

固定十张测试图的 RGB 契约检查中，均值概念概率与归档差为零，分类 logits 重算误差小于 1e-6；仅用于核验实际入口，不代替全 5,286 图性能评估。CUDA 输入缩放、FP16 批次与参数元数据的协议见 [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md)。

## 固定训练配方

本轮区域原型共享干净初始化，6,003 张分层训练图、6 轮适配；μ 预算在训练前由 PiNI 单位换算固定为 0.02686347473647816 RGB，没有继续细调。原生概念和区域监督只进入训练损失。

```bash
python -m pini_cbm.cache_rival --root /path/to/RIVAL10 --out artifacts/rival_vitl14
python -m pini_cbm.train_spatial --locality-loss --region-readout --out artifacts/spatial_region
python -m pini_cbm.train_noise --clean artifacts/spatial_region --mode mean --order sequential --concept-local --mean-budget 0.02686347473647816 --suppress --seed 42 --out artifacts/region_mean
python -m pini_cbm.train_noise --clean artifacts/spatial_region --mode noise --order sequential --concept-local --mean-budget 0.02686347473647816 --suppress --seed 42 --out artifacts/region_noise
```

seed 43 仅改变最后两条命令的适配种子与输出目录，保留相同干净初始化。新环境依赖与全部实验差异以固定协议为准；以上命令不保证跨硬件逐位一致。本轮不再追加参数网格；后续价值在独立背景变化数据上的预先固定验证。
