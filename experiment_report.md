# ResNet18 共享扰动 CBM 实验结果

## CUB、RIVAL10、Waterbirds：换背景分类准确率

![三个数据集的换背景分类准确率](reports/main/rn18_three_dataset_results.png)

## CUB-200-2011：ResNet18、112 个概念、seed 42

### Pipeline

![CUB Pipeline](reports/main/cub_rn18_pipeline.png)

### 原图、换背景分类准确率与概念激活偏移

![CUB 实验结果](reports/main/cub_rn18_main_results.png)

### RGB 扰动、均值与噪声方差

![CUB 扰动可视化](reports/main/cub_rn18_noise_visualization.png)

## RIVAL10：ResNet18、18 个概念、seed 42

### Pipeline

![RIVAL10 Pipeline](reports/main/rival10_rn18_pipeline.png)

### 原图、换背景分类准确率与概念激活偏移

![RIVAL10 实验结果](reports/main/rival10_rn18_main_results.png)

### RGB 扰动、均值与噪声方差

![RIVAL10 扰动可视化](reports/main/rival10_rn18_noise_visualization.png)

## Waterbirds（CUB + Places）：ResNet18、112 个概念、seed 42

### Pipeline

![Waterbirds Pipeline](reports/main/waterbirds_rn18_pipeline.png)

### 原图、换背景分类准确率与概念激活偏移

![Waterbirds 实验结果](reports/main/waterbirds_rn18_main_results.png)

### RGB 扰动、均值与噪声方差

![Waterbirds 扰动可视化](reports/main/waterbirds_rn18_noise_visualization.png)

## 当前设计的问题

- **目标函数允许噪声缩小。** 分类、概念损失要求预测准确，幅度惩罚又鼓励小扰动；没有任何目标鼓励“在不影响预测的前提下，尽量扰动无关区域”。
- **随机换背景训练已经提供了较强监督**，进一步加噪声的收益空间变小。
- **框标注和 14×14 特征比较粗**，难以画出贴合主体边缘的噪声分布。
