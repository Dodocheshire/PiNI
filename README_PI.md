# PiNI 复现与空间噪声 CBM

原仓库 README 和官方源码保留，完整历史 remote 为 `upstream`。新增研究实现位于 `pini_cbm/`，自有私有 remote 为 [Dodocheshire/PiNI](https://github.com/Dodocheshire/PiNI)。

主要结果：[EXPERIMENT_RESULT.md](EXPERIMENT_RESULT.md)。协议：[RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md)。

本机 Python：`/home/pengcheng/miniconda3/envs/pi-cbm/bin/python`（torch 2.11、torchvision、OpenAI CLIP、numpy、scikit-learn、matplotlib、Pillow、einops、yacs）。`PYTHONPATH` 和当前目录须选择本仓库，确保加载官方修改过的 `clip/model.py`，支持空间 tokens。无需为复验安装全部旧版 Dassl/WILDS 环境；`official.py` 直接执行原文件中未改动的网络类定义并记录哈希。

## 本机重跑

从 `PiNI/` 执行。下面用 `python` 表示上述环境 Python，GPU 命令均需要统一锁。

```bash
flock ../D-Pi/artifacts/resources/gpu_work.lock python -m pini_cbm.cache_rival
flock ../D-Pi/artifacts/resources/gpu_work.lock python -m pini_cbm.train_spatial
python -m pini_cbm.audit_clean
flock ../D-Pi/artifacts/resources/gpu_work.lock python -m pini_cbm.train_noise --mode noise --order independent --suppress --out artifacts/noise_independent
flock ../D-Pi/artifacts/resources/gpu_work.lock python -m pini_cbm.run_suite
python -m pini_cbm.visualize
python -m pini_cbm.energy_audit
python -m pini_cbm.summarize
flock ../D-Pi/artifacts/resources/gpu_work.lock python -m pini_cbm.background_eval
```

RIVAL10 默认路径 `/home/pengcheng/VLG-CBM/datasets/RIVAL10`；需准备官方 `meta`、`ordinary` 图像、原生概念 npy 与 mask pkl。不能用 `entire_object_masks` 内的 RGB 遮罩图代替二值 PNG 标注。

OxfordPets 使用 torchvision 下载标准原始数据。已有下载完成后：

```bash
flock ../D-Pi/artifacts/resources/gpu_work.lock python -m pini_cbm.reproduce_pets --kind main
flock ../D-Pi/artifacts/resources/gpu_work.lock python -m pini_cbm.reproduce_pets --kind image
flock ../D-Pi/artifacts/resources/gpu_work.lock python -m pini_cbm.official_gallery
```

`run_official_when_ready` 仅等待本机正在进行的下载并调用以上两个 200 epoch 复验，不重复下载。官方训练 CE 与生成器公式保持不变；seed、LR、精度和 split 差异在报告中披露。

`--locality-loss` 和 `--concept-local` 为第二阶段明确的概念区域训练变体；不改 V1 默认公式。V1 更突出整个主体，V2 用正概念区域 Dice 损失训练注意力、只在预测概念区域施加 μ，并对主体中的概念无关区域施加较小 σ。

所有实际前向只读取 RGB、缓存图像 tokens、概念文本。原生标签与掩膜用于训练和外部评估，不作为 learned 模型推理输入。报告区分确定性主体抑制、均值适配和随机噪声，不将漂亮图表自动当作性能收益。
