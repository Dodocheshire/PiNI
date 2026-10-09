# Enhance Vision-Language Alignment with Noise (AAAI 2025)

本仓库保留官方 PiNI 历史与网络实现，并在 `pini_cbm/` 中单独实现真实 RGB 空间加噪概念瓶颈模型。研究记录：[主要结果与图表](EXPERIMENT_RESULT.md)、[完整报告](EXPERIMENT_REPORT.md)、[固定实验协议](RESEARCH_PROTOCOL.md)、[目标核验与推理用法](GOAL_VERIFICATION.md)。官方复验、训练方式对照与两次区域适配已完成。当前证据支持训练噪声改善背景变化下的分类与概念排序，未证明固定阈值概念精确率或随机推理普遍提高。

## Introduction
This is a [Pytorch](https://pytorch.org) implementation of **PiNI**:
>   Sida Huang, Hongyuan Zhang<sup>\*</sup>, and Xuelong Li<sup>\*</sup>, "Enhance Vision-Language Alignment with Noise", in Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2025. ([arXiv](https://arxiv.org/abs/2412.10817))

![Overview of PiNI](framework.png)

## Requirements
### Installation
```bash
# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
conda install pytorch torchvision torchaudio pytorch-cuda=12.1
# install Dassl
cd Dassl.pytorch/
# Install dependencies in Dassl
pip install -r requirements.txt
# Install this library (no need to re-build if the source code is modified)
python setup.py develop
# install other dependencies
cd ..
pip install -r requirements.txt

```

### Dataset
Follow [DATASETS.md](DATASETS.md) to install the datasets.

## Usage
1. Modify the paths of the data and the models.
    ```bash
    # scripts/clip_vpn.sh
    DATA=/your/path/to/$DATA
    MODEL=/your/path/to/pretrained/clip/models
    ```
2. Run the code
    ```bash
    bash scripts/clip_vpn.sh DATASET CFG SHOTS
    ```
    parameters:
    
    DATASET: dataset name in configs/datasets/, such as **imagenet**

    CFG: config file name in configs/trainsers/TRAINER/, such as **imagenet_config**

    SHOTS: the number of shots in **1, 2, 4, 8, 16**

## Ackownledgement
This repo benefits from [CLIP](https://github.com/openai/CLIP), [CoOp](https://github.com/KaiyangZhou/CoOp.git). Thanks for their excellent works.

## Contact
If you have any question about this project, please contact sidahuang2001@gmail.com and hyzhang98@gmail.com.

## Citation

If you find the code useful for your research, please consider citing our work:

```
@inproceedings{PiNI,
  author={Huang, Sida and Zhang, Hongyuan and Li, Xuelong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  title={Enhance Vision-Language Alignment with Noise}, 
  year={2025},
  pages={},
}
```
