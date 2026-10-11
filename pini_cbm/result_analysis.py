"""Summarize observed noise failures without changing the experiment recipe."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .cross_data import ROOT, SEEDS, write_json
from .cross_experiment import load, restore_noise


def failure_analysis_text():
    names = {"cub": "CUB-200-2011", "cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}
    if not all((ROOT / name / "noise_seed42/best.pt").exists() for name in names):
        return ""
    audit = {"datasets": {}, "source_sha256": {}}
    for source in [Path(__file__), Path(__file__).with_name("models.py"),
                   Path(__file__).with_name("cross_experiment.py")]:
        audit["source_sha256"][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    audit["cub_head_selection"] = {}
    for seed in SEEDS:
        for method in ["mean", "noise"]:
            source = ROOT / "cub" / f"{method}_seed{seed}/head_grid.json"
            if source.exists():
                audit["cub_head_selection"][f"{method}_{seed}"] = json.loads(source.read_text())
                audit["source_sha256"][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    lines = [
        "### 性能下降的原因与下一步",
        "",
        "**判断。** 当前设计的主要负担是逐概念重编码和训练信号分散。CUB 的分类下降还受到稀疏分类头门槛的明显影响；"
        "CIFAR 的下降同时伴随概念分数变差。下一版优先简化图像处理路径，并让分类头训练与实际推理使用相同的视图。",
        "",
        "以下比较使用已完成原图及换背景评估的同一批 seed。骨干均为冻结的 CLIP ViT-L/14，"
        "扰动加在 RGB 像素，测试使用均值视图。",
        "",
        "| 数据集 | 完成轮数 | 原图分类：均值 → 噪声 | 换背景分类变化 | 原图概念指标：均值 → 噪声 |",
        "|---|---:|---:|---:|---|",
    ]
    complexity = []
    timings = []
    for name, title in names.items():
        root = ROOT / name
        _, arrays, _, _, _, protocol = load(name)
        checkpoint = root / "noise_seed42/best.pt"
        noise, _ = restore_noise(checkpoint, arrays["text"], device="cpu")
        parameters = sum(p.numel() for p in noise.parameters() if p.requires_grad)
        k = noise.concepts.k
        train_n = len(protocol["head_train_ids"])
        epochs = protocol["adapt_epochs"]
        pairs = []
        for seed in SEEDS:
            paths = [root / f"{method}_seed{seed}/results.json" for method in ["mean", "noise"]]
            if not all(p.exists() for p in paths):
                continue
            rows = [json.loads(p.read_text()) for p in paths]
            if not all("background" in r and "original" in r for r in rows):
                continue
            pairs.append((seed, *rows))
            for path in paths:
                audit["source_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        metric = "mAP" if name == "cub" else "concept_correlation"
        values = {}
        if pairs:
            for group in ["original", "background"]:
                for key in ["accuracy", metric]:
                    values[f"{group}_{key}"] = [float(np.mean([p[index][group][key] for p in pairs])) for index in [1, 2]]
            accuracy = values["original_accuracy"]
            background = values["background_accuracy"]
            concept = values[f"original_{metric}"]
            concept_text = (f"mAP {concept[0]*100:.2f}% → {concept[1]*100:.2f}%" if name == "cub" else
                            f"r {concept[0]:.3f} → {concept[1]:.3f}")
            lines.append(f"| {title} | {len(pairs)} | {accuracy[0]*100:.2f}% → {accuracy[1]*100:.2f}% | "
                         f"{(background[1]-background[0])*100:+.2f} 个百分点 | {concept_text} |")
        expected = train_n * epochs / k
        audit["datasets"][name] = dict(seeds=[p[0] for p in pairs], metrics=values, concepts=k,
            trainable_parameters=parameters, adaptation_images=train_n, adaptation_epochs=epochs,
            expected_processed_views_per_concept=expected, image_encoder_passes_per_test_image=k+1)
        audit["source_sha256"][str(checkpoint)] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        complexity.append(f"| {title} | {k} | {parameters:,} | {k+1} | 约 {expected:.0f} |")
        log_seed = 43 if name == "cifar100" else 42
        log = ROOT / f"{name}_noise_{log_seed}.log"
        if log.exists():
            for line in log.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (row.get("stage") == "views" and str(row.get("output", "")).endswith("/test_scores.npy")
                        and row.get("done") == row.get("total")):
                    audit["datasets"][name]["observed_full_test_seconds"] = row["elapsed_s"]
                    audit["datasets"][name]["observed_full_test_views"] = row["total"]
                    timings.append(f"{title} {row['elapsed_s']/60:.1f} 分钟")
            audit["source_sha256"][str(log)] = hashlib.sha256(log.read_bytes()).hexdigest()
    lines += [
        "",
        "准确率是认对类别的图片比例；mAP 衡量真实概念的识别排序；r 衡量与 CLIP 概念软分数的一致性。",
        "",
        "**为什么慢、为什么训练信号少。** 一张图有 K 个概念，就要生成 K 张处理图，再分别经过 ViT。"
        "而适配训练每张图每轮只随机选一个概念，6 轮只直接训练约 6 个图像—概念组合。概念越多，"
        "每个概念分到的处理图监督越少。原图概念与区域辅助损失另外覆盖全部概念。",
        "",
        "| 数据集 | 概念数 K | 新增可训练参数 | 每张测试图的 ViT 编码次数 | 6 轮每概念处理图监督次数（期望） |",
        "|---|---:|---:|---:|---:|",
        *complexity,
        "",
        "RIVAL10 只有 18 个概念，同样 6 轮、6,003 张适配图，每概念约有 2,001 次处理图监督。"
        "CUB/CIFAR-100 每概念约 280–290 次，约少 7 倍。CLIP 权重冻结；μ、σ 两个输出层合计仅 516 个参数。"
        "减少逐概念图像编码比单纯缩小这两个输出层更能解决速度问题。",
        "",
        ("已完成日志中的完整测试处理视图提取耗时为：" + "；".join(timings) +
         "。这些是各次运行的实际耗时，包含当时的设备负载。" if timings else ""),
        "",
        "**CUB：分类头丢掉了太多概念。** L1 候选先按 NEC≤16 筛选，NEC 是每类非零概念权重数的平均值。"
        "seed 43 的噪声组候选 NEC 为 16.48，被筛掉后跳到 8.20；对应验证分类从 66.33% 降到 53.83%。"
        "均值组同档候选为 15.79，保留下来。seed 44 也出现相同跳变。两组 mAP 为 55.77% 与 55.91%，"
        "概念排序几乎相同，分类却下降很多。这里的 NEC 变小意味着分类依据被强行减少。",
        "",
        "**CIFAR：扰动损伤了有用信息。** 原图概念 r 和分类一起下降，CIFAR-10 的噪声组 NEC 还反而增加。"
        "背景方差公式含固定下限，预测主体概率低的位置必须承受较强扰动；背景还先被向灰色压缩。"
        "根据这些现象，我判断主体/概念区域预测误差和较强扰动会一起伤到识别线索。"
        "换背景 r 上升表示概念分数更接近原图目标，分类能否提升还取决于各概念是否保留了类别差异。",
        "",
        "**训练与推理使用不同的概念分布。** 噪声组适配选模、分类头训练和验证都使用随机视图，"
        "实际测试使用均值视图。分类头的标准化均值、尺度和权重因此是在另一种输入分布上拟合的。"
        "RIVAL10 的均值推理有效，而 CIFAR 概念数更多、监督更稀疏；下一步先让分类头完整使用均值视图。",
        "",
        "**改进顺序。**",
        "",
        "1. **先修分类头比较。** 固定现有概念模型，用均值视图拟合及验证分类头；"
        "在同一 NEC 预算比较均值组与噪声组，并展示准确率随 NEC 的变化，避免 16.48→8.20 的跳变。",
        "2. **下一版每张图只处理一次。** 共享一张背景扰动图，同时预测全部概念；"
        "先用固定的小噪声，仅在训练时扰动，推理回到原图。将每图 K+1 次编码改为训练原图/扰动图两次、推理一次。",
        "3. **集中概念与样本。** CIFAR-10 先用 16–32 个概念，CIFAR-100 用 32–64 个，"
        "按训练集类别区分力筛选并去掉同义概念；使用全部 45,000 张训练图。CUB 保留 112 个原生属性，"
        "用全部属性损失训练共享扰动视图。",
        "4. **固定区域预测，减轻扰动。** 先冻结预训练区域模块，保持主体 RGB，"
        "去掉强制灰背景处理，让背景噪声可以降到零；再根据原图与换背景分类收益决定是否学习 μ、σ。",
        "",
        "逐轮指标、参数计数和来源哈希见 [noise_failure_analysis.json](reports/main/noise_failure_analysis.json)。",
        "",
    ]
    write_json(Path("reports/main/noise_failure_analysis.json"), audit)
    return "\n".join(lines)
