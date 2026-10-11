"""Append measured CUB/CIFAR a-b-c results to the single paper report."""
import argparse
import hashlib
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .cross_data import ROOT, SEEDS, write_json
from .cross_experiment import load, restore_noise
from .reporting import OUTPUT, BLUE, TEAL, PURPLE, ORANGE, style, save

NAMES = ["cub","cifar10","cifar100"]
TITLES = {"cub":"CUB-200-2011","cifar10":"CIFAR-10","cifar100":"CIFAR-100"}
LABELS = {"lf_cbm":"LF-CBM","supervised_cbm":"监督 CBM","mean":"均值训练","noise":"噪声训练"}
COLORS = {"lf_cbm":BLUE,"supervised_cbm":TEAL,"mean":PURPLE,"noise":ORANGE}
EMOJIS = {"lf_cbm":"🔵","supervised_cbm":"🟢","mean":"🟣","noise":"🟠"}


def collect():
    result = dict(datasets={},seeds=SEEDS,backbone="frozen CLIP ViT-L/14",
                  noise_location="RGB pixels",inference="deterministic mean views")
    sources = []
    for name in NAMES:
        root = ROOT/name
        protocol = json.loads((root/"cache/protocol.json").read_text())
        concepts = json.loads((root/"cache/concepts.json").read_text())
        data = dict(protocol=protocol,concepts=concepts,models={},spatial=[])
        for method in (["lf_cbm","supervised_cbm","mean","noise"] if name=="cub" else ["lf_cbm","mean","noise"]):
            data["models"][method] = []
            for seed in SEEDS:
                path = root/(method+"_seed"+str(seed))/"results.json"
                record = json.loads(path.read_text())
                assert record["seed"]==seed and record["dataset"]==name and "background" in record
                record = {key:value for key,value in record.items() if key!="history"}
                record["intervention"] = {key:value for key,value in record["intervention"].items()
                    if key not in ["ids","donors","donor_background_crops","donor_pool_ids"]}
                data["models"][method].append(record)
                sources.append(path)
        for seed in SEEDS:
            path = root/("noise_seed"+str(seed))/"spatial_statistics.json"
            data["spatial"].append({key:value for key,value in json.loads(path.read_text()).items() if key!="rows"})
            sources.append(path)
        original_gain = np.asarray([r["original"]["accuracy"] for r in data["models"]["noise"]])-np.asarray(
            [r["original"]["accuracy"] for r in data["models"]["mean"]])
        background_gain = np.asarray([r["background"]["accuracy"] for r in data["models"]["noise"]])-np.asarray(
            [r["background"]["accuracy"] for r in data["models"]["mean"]])
        metric = "mAP" if name=="cub" else "concept_correlation"
        concept_gain = np.asarray([r["background"][metric] for r in data["models"]["noise"]])-np.asarray(
            [r["background"][metric] for r in data["models"]["mean"]])
        data["gain"] = dict(original_accuracy_pp=float(original_gain.mean()*100),
            background_accuracy_pp=float(background_gain.mean()*100),
            background_concept_mAP_pp=float(concept_gain.mean()*100) if name=="cub" else None,
            background_concept_correlation=float(concept_gain.mean()) if name!="cub" else None,
            paired_background_accuracy_pp=(background_gain*100).tolist(),
            paired_background_concept_difference=concept_gain.tolist())
        result["datasets"][name] = data
        sources += [root/"cache/protocol.json",root/"cache/CONCEPTS.json",root/"cache/REGIONS.json",
                    root/"background/protocol.json",root/"region/COMPLETE.json"]
        if name!="cub":
            sources += [root/"cache/FOREGROUND_REVIEW.json",root/"cache/foreground_audit.json",
                        root/"foreground_review/CONSTRUCTION_VERIFICATION.json"]
    sources += [ROOT/"FOREGROUND_CACHE_VERIFICATION.json",ROOT/"BASELINE_BACKGROUND_VERIFICATION.json"]
    result["source_sha256"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    result["implementation_sha256"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path("pini_cbm")/(name+".py") for name in
                  ["cross_data","cross_experiment","cross_run","cross_aux_run","cross_finalize",
                   "cross_reporting","cross_verification","cross_foreground_review","models"]]}
    dependencies = Path("requirements-cbm.txt")
    result["implementation_sha256"][str(dependencies)] = hashlib.sha256(dependencies.read_bytes()).hexdigest()
    return result


def mean_std(values,scale=1,precision=2):
    values = np.asarray(values,dtype=float)*scale
    if len(values)==1:
        return f"{values[0]:.{precision}f}"
    return f"{values.mean():.{precision}f} ± {values.std(ddof=1):.{precision}f}"


def bars(ax,methods,records,condition,key,scale=1,precision=2):
    samples = np.asarray([[r[condition][key] for r in records[m]] for m in methods])*scale
    means = samples.mean(1)
    errors = samples.std(1,ddof=1) if samples.shape[1]>1 else np.zeros(len(methods))
    positions = np.arange(len(methods))
    ax.bar(positions,means,yerr=errors,capsize=4,color=[COLORS[m] for m in methods],width=.62)
    for i,mean in enumerate(means):
        ax.scatter(np.full(samples.shape[1],i),samples[i],s=12,color="#303030",zorder=4)
        ax.annotate(f"{mean:.{precision}f}",(i,mean+errors[i]),xytext=(0,5),textcoords="offset points",
                    ha="center",fontsize=10,fontweight="bold")
    ax.set_xticks(positions,[LABELS[m] for m in methods])
    ax.grid(axis="y",alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[["right","top"]].set_visible(False)
    if key=="concept_correlation":
        ax.set_ylim(min(0,float(samples.min())-.08),1.12)
        ax.set_yticks([0,.25,.5,.75,1])
    elif key=="NEC":
        ax.set_ylim(0,max(20,float((means+errors).max())*1.24))
    else:
        ax.set_ylim(0,112)
        ax.set_yticks([0,25,50,75,100])


def classification_figure(result,filename="cross_classification"):
    names = [n for n in NAMES if n in result["datasets"]]
    fig,axes = plt.subplots(len(names),3,figsize=(13,max(4,11*len(names)/3)),layout="constrained",squeeze=False)
    for row,name in enumerate(names):
        data = result["datasets"][name]
        methods = ["lf_cbm","supervised_cbm","mean","noise"] if name=="cub" else ["lf_cbm","mean","noise"]
        methods = [m for m in methods if m in data["models"]]
        for col,(key,scale,title,precision) in enumerate([
            ("accuracy",100,"分类准确率（%）↑",2),
            ("mAP" if name=="cub" else "concept_correlation",100 if name=="cub" else 1,
             "概念 mAP（%）↑" if name=="cub" else "CLIP 概念一致性 r ↑",2 if name=="cub" else 3),
            ("NEC",1,"每类使用概念数 NEC ↓",2)]):
            bars(axes[row,col],methods,data["models"],"original",key,scale,precision)
            axes[row,col].set_title(TITLES[name]+" · "+title,fontsize=12)
    rounds = "3 个种子 · 平均值与标准差" if filename in ["cross_classification","cross_baselines"] else "已完成的独立轮次"
    fig.suptitle("实验 b 扩展：原图测试集 · "+rounds,fontsize=16)
    save(fig,filename)


def background_figure(result,filename="cross_background"):
    names = [n for n in NAMES if n in result["datasets"]]
    fig,axes = plt.subplots(len(names),2,figsize=(13,max(4.5,11*len(names)/3)),layout="constrained",squeeze=False)
    for row,name in enumerate(names):
        data = result["datasets"][name]
        methods = ["lf_cbm","supervised_cbm","mean","noise"] if name=="cub" else ["lf_cbm","mean","noise"]
        methods = [m for m in methods if m in data["models"]]
        for col,(key,scale,title,precision) in enumerate([
            ("accuracy",100,"换背景分类准确率（%）↑",2),
            ("mAP" if name=="cub" else "concept_correlation",100 if name=="cub" else 1,
             "换背景概念 mAP（%）↑" if name=="cub" else "原图概念分数一致性 r ↑",2 if name=="cub" else 3)]):
            ax = axes[row,col]
            bars(ax,methods,data["models"],"background",key,scale,precision)
            ax.set_title(TITLES[name]+" · "+title,fontsize=12)
            if "mean" in methods and "noise" in methods:
                gain = float(np.mean([n["background"][key]-m["background"][key]
                                     for m,n in zip(data["models"]["mean"],data["models"]["noise"])]))*scale
                unit = "个百分点" if col==0 or name=="cub" else "r"
                ax.text(.5,.97,f"噪声训练 - 均值训练：{gain:+.2f} {unit}" if unit!="r" else
                        f"噪声训练 - 均值训练：{gain:+.3f} r",
                        transform=ax.transAxes,ha="center",va="top",color="#A34D00",fontsize=11,fontweight="bold")
    rounds = "3 轮" if filename=="cross_background" else "已完成的独立轮次"
    fig.suptitle("实验 c 扩展：各数据集相同 1,000 张换背景图 · "+rounds,fontsize=16)
    save(fig,filename)


def choose_example(name,arrays,protocol,concepts,labels):
    preferred = {"cub":["has_wing_color::black","has_bill_color::black","has_tail_shape::notched_tail"],
                 "cifar10":["four wheels","wings","fur"],
                 "cifar100":["a petal","leaves","fur covering the body"]}
    compatible = {
        "cifar10":{"four wheels":{"automobile","truck"},"wings":{"airplane","bird"},
                   "fur":{"cat","deer","dog","horse"}},
        "cifar100":{"a petal":{"orchid","poppy","rose","sunflower","tulip"},
                    "leaves":{"maple_tree","oak_tree","palm_tree","pine_tree","willow_tree"},
                    "fur covering the body":{"bear","fox","hamster","leopard","lion","mouse","rabbit","shrew","squirrel","tiger","wolf"}}}
    def matches_class(i,k):
        return name=="cub" or protocol["classes"][int(labels[i])] in compatible[name].get(concepts[k],set())
    ordered = [concepts.index(x) for x in preferred[name] if x in concepts]
    ordered += [i for i in range(len(concepts)) if i not in ordered]
    for k in ordered:
        for i in protocol["region_val_ids"]:
            if not matches_class(i,k):
                continue
            if arrays["attrs"][i,k]<.6 or not arrays["mask_valid"][i,k]:
                continue
            fg = np.asarray(arrays["fg224"][i])
            if not .12<fg.mean()<.85:
                continue
            body = np.asarray(arrays["masks"][i,-1])
            part = np.asarray(arrays["masks"][i,k])
            if .08<part.sum()/max(body.sum(),1e-6)<.85:
                return int(i),int(k)
    for i in protocol["region_val_ids"]:
        ks = np.flatnonzero((arrays["attrs"][i]>=.6)&np.asarray(arrays["mask_valid"][i,:-1]))
        ks = [k for k in ks if matches_class(i,k)]
        if len(ks):
            return int(i),int(ks[0])
    raise AssertionError("No valid spatial example for "+name)


@torch.no_grad()
def spatial_figure(result,filename="cross_spatial"):
    # Only the small generator uses CPU; the running ViT experiment keeps GPU.
    from .models import SpatialConcepts,ConditionalRGBNoise
    rows = []
    for name in [n for n in NAMES if n in result["datasets"]]:
        _,arrays,_,labels,_,protocol = load(name)
        concepts = result["datasets"][name]["concepts"]
        i,k = choose_example(name,arrays,protocol,concepts,labels)
        saved = torch.load(ROOT/name/"noise_seed42/best.pt",map_location="cpu",weights_only=False)
        model = SpatialConcepts(torch.tensor(np.array(arrays["text"])),region_readout=True)
        noise = ConditionalRGBNoise(model,concept_local=True,mean_budget=protocol["mean_budget"])
        noise.load_state_dict(saved["model"])
        noise.eval()
        tok = torch.tensor(np.array(arrays["tokens"][[i]]),dtype=torch.float32)
        rgb = torch.tensor(np.array(arrays["rgb"][[i]]),dtype=torch.float32)/255
        generator = torch.Generator().manual_seed(1207)
        image,mu,sigma,_ = noise.perturb(rgb,tok,torch.tensor([k]),mode="noise",
            epsilon=torch.randn(rgb.shape,generator=generator),suppress=True)
        rows.append(dict(name=name,image_id=i,concept_id=k,concept=concepts[k],
            class_name=protocol["classes"][int(labels[i])],teacher_target=float(arrays["attrs"][i,k]),
            rgb=rgb[0].permute(1,2,0).numpy(),mu=mu[0].square().mean(0).sqrt().numpy(),
            variance=sigma[0].square().mean(0).numpy(),sample=image[0].permute(1,2,0).clamp(0,1).numpy()))
    mu_max = max(float(r["mu"].max()) for r in rows)
    variance_max = max(float(r["variance"].max()) for r in rows)
    fig,axes = plt.subplots(len(rows),4,figsize=(13,max(5.2,10*len(rows)/3)),layout="constrained",squeeze=False)
    fig.set_layout_engine("constrained",rect=(0,0,1,.90))
    for row,r in enumerate(rows):
        axes[row,0].imshow(r["rgb"])
        mu_image = axes[row,1].imshow(r["mu"],cmap="magma",vmin=0,vmax=mu_max)
        var_image = axes[row,2].imshow(r["variance"],cmap="viridis",vmin=0,vmax=variance_max)
        axes[row,3].imshow(r["sample"])
        for ax in axes[row]:
            ax.set_xticks([]);ax.set_yticks([])
        concept = r["concept"].replace("has_","").replace("::",": ").replace("_"," ")
        label = TITLES[r["name"]]+(" / "+r["class_name"] if r["name"]!="cub" else "")
        axes[row,0].set_ylabel(label+"\n"+"\n".join(textwrap.wrap(concept,24)),fontsize=10)
        data = result["datasets"][r["name"]]["spatial"]
        values = [mean_std([d["ratios"][key] for d in data]) for key in
                  ["background_foreground_variance","foreground_background_mean","part_other_mean"]]
        axes[row,3].set_xlabel("背景/主体方差："+values[0]+" 倍\n主体/背景均值："+values[1]+" 倍\n部位/其他均值："+values[2]+" 倍",fontsize=9)
    for ax,title in zip(axes[0],["原图","均值偏移 |μ|","噪声方差 σ²","实际 RGB 加噪图"]):
        ax.set_title(title,fontsize=12)
    fig.colorbar(mu_image,ax=axes[:,1],location="bottom",shrink=.75,label="三通道均值 RMS")
    fig.colorbar(var_image,ax=axes[:,2],location="bottom",shrink=.75,label="三通道平均方差")
    fig.suptitle("实验 a 扩展：背景与概念部位的像素扰动",fontsize=16,y=.985)
    save(fig,filename)
    metadata = [{k:v for k,v in r.items() if k not in ["rgb","mu","variance","sample"]} for r in rows]
    write_json(OUTPUT/(filename+"_examples.json"),dict(seed=42,sampling_seed=1207,examples=metadata,
        selection="first valid validation example in predefined concept preference order; CIFAR native class must match the depicted part (wheels: automobile/truck; wings: airplane/bird; petals: flowers; leaves: trees; fur: mammals); object area 12–85%, local part area 8–85%",
        mean_color_max=mu_max,variance_color_max=variance_max))


def setup_text(result):
    lines = [
        "### 5. CUB、CIFAR-10、CIFAR-100 的统一设置",
        "",
        "以下扩展仍使用冻结的 **CLIP ViT-L/14** 和 $224\\times224$ RGB 图片，加噪位置仍是 **RGB 像素**。"
        "每张图针对 $K$ 个概念生成 $K$ 张视图；噪声训练组和均值训练组测试时都使用均值视图。种子为 **42、43、44**。",
        "图 1–3 和表 1 以 RIVAL10 的 18 个概念、10 个类别展开；以下三个数据集使用相同模块，"
        "概念张量中的 18 替换为 $K$，分类头输出维度采用各数据集的类别数。",
        "",
        "| 数据集 | 类别数 | 概念数 K | 训练 / 验证 / 测试 | 概念和区域怎样得到 |",
        "|---|---:|---:|---|---|",
    ]
    for name in NAMES:
        data = result["datasets"][name]
        p = data["protocol"]
        counts = p["counts"]
        source = ("112 个原生逐图属性；主体分割图；15 个部位点生成局部区域" if name=="cub" else
                  "LF-CBM 概念文本；CLIP 概念软分数；SAM2 主体与 CLIP patch 概念区域")
        lines.append(f"| {TITLES[name]} | {len(p['classes'])} | {len(data['concepts'])} | "
                     f"{counts['train']:,} / {counts['val']:,} / {counts['test']:,} | {source} |")
    lines += [
        "",
        "**表 6 注释。** 数据从本机已下载的副本拷贝。验证集从官方训练集按类别分层划分，比例 10%，划分种子 310；"
        "官方测试集用于最终评估。CUB 使用 [Koh 2020 公布的 112 个属性索引]"
        "(https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/generate_new_data.py)，保留每张图自己的 0/1 属性标签。"
        "部位区域是以可见部位点为中心、半径为图像边长 9% 的圆，与主体分割图相交。",
        "",
        "CIFAR 的 64 / 128 个概念从 LF-CBM 候选文本中，按训练集 CLIP 分数的“类别间方差 / 总方差”选出，"
        "类别名称本身不作为概念。概念目标是训练集标准化 CLIP 分数经过 sigmoid 得到的软分数。"
        "主体区域由冻结的 Grounding DINO 与 SAM2 生成：CLIP 先预测图像类别，Grounding DINO 据此生成检测框；"
        "SAM2 同时使用检测框和五个中心附近的提示点。以真实 RGB 主体图和背景图的 CLIP 分数差选择掩码，"
        "再合并重叠的同一主体区域。"
        "主体区域监督覆盖固定的 6,000 张训练图；概念区域由主体内 CLIP 文本匹配分数最高的 40% patch 生成。",
        "",
        "区域预训练使用完整训练划分，共 40 轮。像素适配使用 CUB 的全部 5,394 张训练图、每个 CIFAR 的固定 6,000 张训练图，"
        "各训练 6 轮；固定 500 张验证图用于选择适配权重。噪声组在验证时采样随机扰动，均值组使用均值视图，"
        "各自按验证概念指标选择适配权重。分类头使用同一批适配训练图，完整验证集选取 L1 强度。"
        "L1 网格沿用 0.003、0.01、0.03、0.06、0.1，并乘以 $10/N$，$N$ 为类别数；"
        "从平均 NEC 不超过 16 的候选中选择验证准确率最高的分类头。",
        "",
        "换背景使用各数据集固定的 1,000 张测试图。CUB 的背景来自验证集其他类别图片，供体主体先填成灰色。"
        "CIFAR 从供体的 SAM2 区域和检测框外截取纯背景片区，面积至少占原图 10%，放大到 224×224 后替换目标背景。"
        "每个数据集的图片、供体和背景片区在全部方法及三轮中固定。",
        "",
        "LF-CBM 使用 CLS 特征与 CLIP 概念文本分数训练 cosine-cubed 投影。CUB 的监督 CBM 按 Koh 的顺序训练思路，"
        "先以 BCE 学习属性，再拟合稀疏分类头。CIFAR 使用 LF-CBM 对照。",
        "",
    ]
    return "\n".join(lines)


def a_text(result):
    lines = [
        "#### CUB 与 CIFAR 扩展",
        "",
        "**目的与设置。** 每个数据集使用固定的 500 张验证图，检查 RGB 噪声是否更多落在背景、均值偏移是否集中在概念部位。"
        "CUB 按主体分割与部位点区域统计；CIFAR 按 SAM2 主体与 CLIP 概念区域统计。",
        "",
        "![实验 a 扩展：三个数据集的实际 RGB 加噪及空间强度](reports/main/cross_spatial.png)",
        "",
        "**图 7 注释。** 三行分别为 CUB、CIFAR-10、CIFAR-100 的验证图，列含义与图 4 相同。"
        "示例使用 seed 42 的噪声模型、固定采样种子 1207；均值列和方差列分别共享色标。"
        "示例按概念区域有效与主体占比选择，具体图片、概念编号见 [cross_spatial_examples.json](reports/main/cross_spatial_examples.json)。",
        "",
        "| 数据集 | 背景方差 / 主体方差 | 主体均值 / 背景均值 | 部位均值 / 主体其他均值 | 有效概念 / 局部区域数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in NAMES:
        stats = result["datasets"][name]["spatial"]
        values = [mean_std([r["ratios"][key] for r in stats])+" 倍" for key in
                  ["background_foreground_variance","foreground_background_mean","part_other_mean"]]
        lines.append(f"| {TITLES[name]} | "+" | ".join(values)+f" | {stats[0]['positive_pairs']:,} / {stats[0]['local_pairs']:,} |")
    lines += ["",
        "**表 7 注释。** 每个比值先分别平均各个有效区域中的强度，再相除；表中对三个模型的比值计算平均值和样本标准差。"
        "局部概念区域覆盖主体不足 95%。方差代表随机扰动强度，均值代表稳定的像素改变幅度。",
        ""]
    statements = []
    for name in NAMES:
        stats = result["datasets"][name]["spatial"]
        bg = np.mean([r["ratios"]["background_foreground_variance"] for r in stats])
        local = np.mean([r["ratios"]["part_other_mean"] for r in stats])
        statements.append(f"{TITLES[name]} 的背景方差为主体的 {bg:.2f} 倍，概念部位均值为主体其他部位的 {local:.2f} 倍")
    lines += ["**结论。** "+"；".join(statements)+"。",""]
    return "\n".join(lines)


def b_text(result):
    lines = [
        "#### CUB 与 CIFAR 扩展",
        "",
        "**目的与设置。** 使用三个数据集各自的完整官方测试集，比较原图分类、概念识别和 NEC。"
        "冻结的 ViT-L/14 与概念库在同一数据集各方法间一致；空间加噪模型按均值视图推理。",
        "",
        "![实验 b 扩展：三个数据集的原图分类、概念指标及 NEC](reports/main/cross_classification.png)",
        "",
        "**图 8 注释。** 蓝色 LF-CBM，绿色监督 CBM，紫色均值训练，橙色噪声训练。柱高为三轮平均，误差线为样本标准差，黑点为每轮指标。"
        "CUB 中间列是原生属性 mAP；CIFAR 中间列是与 CLIP 概念软分数的平均 Pearson 相关系数 r。",
        "",
        "| 数据集 | 方法 | 原图准确率（%）↑ | 概念指标 ↑ | NEC ↓ |",
        "|---|---|---:|---:|---:|",
    ]
    for name in NAMES:
        data = result["datasets"][name]
        for method in (["lf_cbm","supervised_cbm","mean","noise"] if name=="cub" else ["lf_cbm","mean","noise"]):
            rows = data["models"][method]
            accuracy = mean_std([r["original"]["accuracy"] for r in rows],100)
            concept = ("mAP "+mean_std([r["original"]["mAP"] for r in rows],100)+"%" if name=="cub" else
                       "r "+mean_std([r["original"]["concept_correlation"] for r in rows],1,3))
            nec = mean_std([r["original"]["NEC"] for r in rows])
            lines.append(f"| {TITLES[name]} | {EMOJIS[method]} {LABELS[method]} | {accuracy} | {concept} | {nec} |")
    lines += ["",
        "**表 8 注释。** 准确率是认对类别的图片比例。CUB mAP 衡量 112 个真实属性的识别排序；"
        "CIFAR r 先对每个概念计算模型分数与 CLIP 软分数的相关性，再平均，越接近 1 表示两者越一致。"
        "NEC 是每个类别的非零概念权重数，再对类别平均，非零阈值为 $10^{-5}$。",
        ""]
    statements = []
    for name in NAMES:
        models = result["datasets"][name]["models"]
        own = np.mean([r["original"]["accuracy"] for r in models["noise"]])*100
        nec = np.mean([r["original"]["NEC"] for r in models["noise"]])
        baseline = "supervised_cbm" if name=="cub" else "lf_cbm"
        reference = np.mean([r["original"]["accuracy"] for r in models[baseline]])*100
        original_gain = result["datasets"][name]["gain"]["original_accuracy_pp"]
        statements.append(f"{TITLES[name]}：空间加噪准确率 {own:.2f}%，NEC {nec:.2f}，相对{LABELS[baseline]}为 {own-reference:+.2f} 个百分点，相对均值训练为 {original_gain:+.2f} 个百分点")
    lines += ["**结论。** "+"；".join(statements)+"。",""]
    return "\n".join(lines)


def c_text(result):
    lines = [
        "#### CUB 与 CIFAR 扩展",
        "",
        "**目的与设置。** 每个数据集固定 1,000 张测试图，保留主体，换成验证集其他类别图片的背景。"
        "各方法及三轮使用完全相同的图片与背景；均值训练组和噪声训练组都按均值视图推理。",
        "",
        "![实验 c 扩展：三个数据集的换背景结果与噪声训练收益](reports/main/cross_background.png)",
        "",
        "**图 9 注释。** 紫色为只学习均值的空间 CBM，其余颜色与前图一致。CUB 用原生主体分割图，CIFAR 用 SAM2 主体区域构造换背景图。"
        "目标主体像素逐像素保留；CUB 供体主体先变成灰色，CIFAR 使用供体物体外的纯背景片区。CUB 使用目标原图属性标签，"
        "CIFAR 使用目标原图的 CLIP 概念软分数衡量概念保持程度。",
        "",
        "| 数据集 | 方法 | 换背景准确率（%）↑ | 换背景概念指标 ↑ |",
        "|---|---|---:|---:|",
    ]
    for name in NAMES:
        data = result["datasets"][name]
        for method in (["lf_cbm","supervised_cbm","mean","noise"] if name=="cub" else ["lf_cbm","mean","noise"]):
            rows = data["models"][method]
            accuracy = mean_std([r["background"]["accuracy"] for r in rows],100)
            concept = ("mAP "+mean_std([r["background"]["mAP"] for r in rows],100)+"%" if name=="cub" else
                       "r "+mean_std([r["background"]["concept_correlation"] for r in rows],1,3))
            lines.append(f"| {TITLES[name]} | {EMOJIS[method]} {LABELS[method]} | {accuracy} | {concept} |")
        gain = data["gain"]
        concept_gain = (f"{gain['background_concept_mAP_pp']:+.2f} 个百分点" if name=="cub" else
                        f"{gain['background_concept_correlation']:+.3f} r")
        lines.append(f"| {TITLES[name]} | **噪声训练 − 均值训练** | **{gain['background_accuracy_pp']:+.2f} 个百分点** | **{concept_gain}** |")
    lines += ["",
        "**表 9 注释。** 指标和三轮统计方式与表 8 相同。差值在相同种子的均值组与噪声组间计算，再对三轮平均；"
        "两组起点、适配训练图、轮数、均值预算和背景处理一致。",
        ""]
    statements = []
    for name in NAMES:
        gain = result["datasets"][name]["gain"]["background_accuracy_pp"]
        statements.append(f"{TITLES[name]} 换背景分类"+("提高" if gain>=0 else "降低")+f" {abs(gain):.2f} 个百分点")
    lines += ["**结论。** 噪声训练相对只学习均值，"+"；".join(statements)+"。",
        "",
        "全部扩展实验的逐轮指标与来源见 [cross_results.json](reports/main/cross_results.json)，"
        "核对记录见 [cross_verification.json](reports/main/cross_verification.json)。三个扩展图均提供同名 SVG、PDF。",
        "在本目录运行 <code>flock ../D-Pi/artifacts/resources/gpu_work.lock python -m pini_cbm.cross_run</code> "
        "执行三数据集实验队列，<code>python -m pini_cbm.summarize</code> 生成报告，"
        "<code>python -m pini_cbm.cross_verification</code> 核对结果；"
        "实验依赖见 [requirements-cbm.txt](requirements-cbm.txt)。",
        ""]
    return "\n".join(lines)


def extend_report(result=None):
    result = collect() if result is None else result
    # Rebuild the RIVAL text before insertion, so repeated runs are idempotent.
    from .reporting import collect as rival_collect
    from .paper_text import write_report
    write_report(rival_collect())
    path = Path("EXPERIMENT_REPORT.md")
    text = path.read_text().replace("所有实验使用 [RIVAL10]","RIVAL10 实验使用 [RIVAL10]",1)
    text = text.replace("### 4. 如何训练与推理","### 4. 如何训练与推理（RIVAL10 设置）",1)
    text = text.replace("## 主要实验和结果",setup_text(result)+"\n## 主要实验和结果",1)
    text = text.replace("### b. 原图分类与概念识别",a_text(result)+"\n### b. 原图分类与概念识别",1)
    text = text.replace("### c. 换背景后是否更容易认对",b_text(result)+"\n### c. 换背景后是否更容易认对",1)
    text += "\n"+c_text(result)
    summary = "扩展验证使用 CUB、CIFAR-10、CIFAR-100；噪声训练相对均值训练的换背景分类变化依次为 "
    summary += "、".join(f"**{result['datasets'][n]['gain']['background_accuracy_pp']:+.2f} 个百分点**" for n in NAMES)+"。"
    text = text.replace("\n## Background（为什么要做）","\n"+summary+"\n\n## Background（为什么要做）",1)
    path.write_text(text)
    write_json(OUTPUT/"cross_results.json",result)
    for filename in ["cross_baselines.png","cross_baselines.svg","cross_baselines.pdf",
                     "cross_baselines.json","cross_baseline_verification.json","cross_progress.json",
                     "cross_progress_spatial_examples.json",
                     *[f"cross_progress_{group}.{ext}" for group in ["spatial","classification","background"] for ext in ["png","svg","pdf"]]]:
        (OUTPUT/filename).unlink(missing_ok=True)


def baseline_progress():
    """Publish the completed baselines while the full spatial queue continues."""
    result = dict(datasets={},seeds=SEEDS,stage="baselines_complete_spatial_experiments_running")
    sources = []
    for name in NAMES:
        root = ROOT/name
        data = dict(protocol=json.loads((root/"cache/protocol.json").read_text()),
                    concepts=json.loads((root/"cache/concepts.json").read_text()),models={})
        for method in (["lf_cbm","supervised_cbm"] if name=="cub" else ["lf_cbm"]):
            data["models"][method] = []
            for seed in SEEDS:
                source = root/(method+"_seed"+str(seed))/"original_results.json"
                row = json.loads(source.read_text())
                data["models"][method].append({k:v for k,v in row.items() if k!="history"})
                sources.append(source)
        result["datasets"][name] = data
    result["source_sha256"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    style()
    classification_figure(result,"cross_baselines")
    from .reporting import collect as rival_collect
    from .paper_text import write_report
    write_report(rival_collect())
    path = Path("EXPERIMENT_REPORT.md")
    text = path.read_text().replace("所有实验使用 [RIVAL10]","RIVAL10 实验使用 [RIVAL10]",1)
    text = text.replace("### 4. 如何训练与推理","### 4. 如何训练与推理（RIVAL10 设置）",1)
    text = text.replace("## 主要实验和结果",setup_text(result)+"\n## 主要实验和结果",1)
    lines = [
        "#### CUB 与 CIFAR 扩展：已完成的三轮基线",
        "",
        "**目的与设置。** 在 CUB 的 5,794 张、CIFAR-10 与 CIFAR-100 各 10,000 张官方测试图上，"
        "使用冻结的 ViT-L/14，先测量 LF-CBM 与 CUB 监督 CBM 的原图表现。种子为 42、43、44。",
        "",
        "![实验 b 扩展：已完成的三轮基线](reports/main/cross_baselines.png)",
        "",
        "**图注。** 蓝色为 LF-CBM，绿色为监督 CBM；柱高为三轮平均，误差线为样本标准差，黑点为每轮指标。"
        "左列是分类准确率，右列是平均使用概念数 NEC；中列在 CUB 上是原生属性 mAP，在 CIFAR 上是与 CLIP 概念软分数的相关系数 r。",
        "",
        "| 数据集 | 方法 | 原图准确率（%）↑ | 概念指标 ↑ | NEC ↓ |",
        "|---|---|---:|---:|---:|",
    ]
    for name in NAMES:
        for method,rows in result["datasets"][name]["models"].items():
            accuracy = mean_std([r["original"]["accuracy"] for r in rows],100)
            nec = mean_std([r["original"]["NEC"] for r in rows])
            concept = ("mAP "+mean_std([r["original"]["mAP"] for r in rows],100)+"%" if name=="cub" else
                       "r "+mean_std([r["original"]["concept_correlation"] for r in rows],1,3))
            lines.append(f"| {TITLES[name]} | {EMOJIS[method]} {LABELS[method]} | {accuracy} | {concept} | {nec} |")
    cub_accuracy = np.mean([r["original"]["accuracy"] for r in result["datasets"]["cub"]["models"]["supervised_cbm"]])*100
    cifar10_accuracy = np.mean([r["original"]["accuracy"] for r in result["datasets"]["cifar10"]["models"]["lf_cbm"]])*100
    cifar100_accuracy = np.mean([r["original"]["accuracy"] for r in result["datasets"]["cifar100"]["models"]["lf_cbm"]])*100
    lines += ["",
        "**指标含义。** 准确率是认对类别的图片比例；mAP 是真实概念的平均识别排序质量；"
        "r 衡量模型概念分数与 CLIP 软分数的一致程度，越接近 1 越一致；NEC 是每类非零概念权重数的平均值。",
        "",
        f"**当前结果。** CUB 的监督 CBM 原图准确率为 {cub_accuracy:.2f}%；CIFAR-10 与 CIFAR-100 的 LF-CBM 分别为 {cifar10_accuracy:.2f}% 和 {cifar100_accuracy:.2f}%。"
        "空间模型的均值组、噪声组以及换背景实验继续运行，完成后加入同一表图。",
        "",
        "逐轮基线指标见 [cross_baselines.json](reports/main/cross_baselines.json)，"
        "从完整测试预测与分类权重复算的记录见 [cross_baseline_verification.json](reports/main/cross_baseline_verification.json)。",
        "",
    ]
    text = text.replace("### c. 换背景后是否更容易认对","\n".join(lines)+"\n### c. 换背景后是否更容易认对",1)
    text = completed_round_progress(text)
    path.write_text(text)
    write_json(OUTPUT/"cross_baselines.json",result)
    verification = json.loads((ROOT/"BASELINE_VERIFICATION.json").read_text())
    assert verification["passed"] and verification["models"]==12
    write_json(OUTPUT/"cross_baseline_verification.json",verification)
    print("Published completed three-round CUB/CIFAR baselines; spatial experiment queue continues",flush=True)


def completed_round_progress(text):
    """Add only complete matched rounds, with metrics replayed from predictions."""
    from .cross_experiment import concept_metrics
    from .cross_verification import verify_head_selection
    from .train_spatial import head_predict
    verified = dict(passed=True,datasets={},source_sha256={},head_selection=[],original_models=[],spatial_datasets={})
    sections = {key:[] for key in "abc"}
    for name in NAMES:
        methods = ["lf_cbm","supervised_cbm","mean","noise"] if name=="cub" else ["lf_cbm","mean","noise"]
        ready = []
        for seed in SEEDS:
            paths = [ROOT/name/f"{m}_seed{seed}/results.json" for m in methods]
            stats_path = ROOT/name/f"noise_seed{seed}/spatial_statistics.json"
            if stats_path.exists() and all(p.exists() and "background" in json.loads(p.read_text()) for p in paths):
                ready.append(seed)
        original_ready = [(seed,method) for seed in SEEDS if seed not in ready for method in ["mean","noise"]
                          if (ROOT/name/f"{method}_seed{seed}/original_results.json").exists()]
        if not ready and not original_ready:
            continue
        _,arrays,splits,labels,_,protocol = load(name)
        metric = "mAP" if name=="cub" else "concept_correlation"
        for seed,method in original_ready:
            root = ROOT/name/f"{method}_seed{seed}"
            source = root/"original_results.json"
            row = json.loads(source.read_text())
            head = torch.load(root/"head.pt",map_location="cpu",weights_only=False)
            scores = np.load(root/"test_scores.npy")
            logits = np.load(root/"test_class_logits.npy")
            assert scores.shape==(len(splits["test"]),len(arrays["text"])) and np.isfinite(scores).all()
            assert np.allclose(logits,head_predict(head,scores),atol=1e-5)
            measured = concept_metrics(name,np.array(arrays["attrs"][splits["test"]]),scores)
            measured["accuracy"] = float((logits.argmax(1)==labels[splits["test"]]).mean())
            measured["NEC"] = float((head["weight"].abs()>1e-5).sum(1).float().mean())
            assert all(np.isclose(v,row["original"][k],atol=2e-6,rtol=0) for k,v in measured.items())
            selection = verify_head_selection(name,method,seed,row,head,arrays,splits,labels,protocol)
            verified["head_selection"].append(dict(dataset=name,method=method,seed=seed,**selection))
            verified["original_models"].append(dict(dataset=name,method=method,seed=seed,test_n=len(splits["test"]),original=measured))
            verified["source_sha256"][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        if not ready:
            continue
        rows = {m:[] for m in methods}
        stats = []
        for seed in ready:
            for method in methods:
                root = ROOT/name/f"{method}_seed{seed}"
                source = root/"results.json"
                row = json.loads(source.read_text())
                head = torch.load(root/("head.pt" if method in ["mean","noise"] else "model.pt"),
                                  map_location="cpu",weights_only=False)
                if method not in ["mean","noise"]:
                    head = head["head"]
                selection = verify_head_selection(name,method,seed,row,head,arrays,splits,labels,protocol)
                verified["head_selection"].append(dict(dataset=name,method=method,seed=seed,**selection))
                for condition,ids in [("test",splits["test"]),("background",protocol["background_ids"])]:
                    scores = np.load(root/f"{condition}_scores.npy")
                    logits = np.load(root/f"{condition}_class_logits.npy")
                    assert scores.shape==(len(ids),len(arrays["text"])) and np.isfinite(scores).all()
                    assert np.allclose(logits,head_predict(head,scores),atol=1e-5)
                    measured = concept_metrics(name,np.array(arrays["attrs"][ids]),scores)
                    measured["accuracy"] = float((logits.argmax(1)==labels[ids]).mean())
                    expected = row["original" if condition=="test" else "background"]
                    assert all(np.isclose(v,expected[k],atol=2e-6,rtol=0) for k,v in measured.items())
                nec = float((head["weight"].abs()>1e-5).sum(1).float().mean())
                assert np.isclose(nec,row["original"]["NEC"],atol=2e-6,rtol=0)
                rows[method].append(dict(seed=seed,original=row["original"],background=row["background"]))
                verified["source_sha256"][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
            source = ROOT/name/f"noise_seed{seed}/spatial_statistics.json"
            spatial = json.loads(source.read_text())
            pairs = spatial["rows"]
            local = [r for r in pairs if "part_mean" in r]
            for group,num,den,key in [(pairs,"background_variance","foreground_variance","background_foreground_variance"),
                                      (pairs,"foreground_mean","background_mean","foreground_background_mean"),
                                      (local,"part_mean","other_mean","part_other_mean")]:
                value = np.mean([r[num] for r in group])/max(np.mean([r[den] for r in group]),1e-12)
                assert np.isclose(value,spatial["ratios"][key],atol=2e-6,rtol=0)
            stats.append({k:v for k,v in spatial.items() if k!="rows"})
            verified["source_sha256"][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        verified["datasets"][name] = dict(seeds=ready,models=rows,spatial=stats,
                                          concepts=json.loads((ROOT/name/"cache/concepts.json").read_text()))
        title = f"#### {TITLES[name]}：已完成 {len(ready)} 轮（seed "+"、".join(map(str,ready))+"）"
        def value(xs,scale=1,precision=2):
            return mean_std(xs,scale,precision) if len(xs)>1 else f"{float(xs[0])*scale:.{precision}f}"
        ratios = [value([s["ratios"][key] for s in stats]) for key in
                  ["background_foreground_variance","foreground_background_mean","part_other_mean"]]
        sections["a"] += [title,"",f"**结果。** 固定 500 张验证图中，背景方差是主体的 **{ratios[0]} 倍**，"
                          f"主体均值偏移是背景的 **{ratios[1]} 倍**，概念部位均值偏移是主体其他部位的 **{ratios[2]} 倍**。",""]
        for key,condition in [("b","original"),("c","background")]:
            sections[key] += [title,"","| 方法 | 分类准确率（%）↑ | 概念指标 ↑ | NEC ↓ |","|---|---:|---:|---:|"]
            for method in methods:
                model = rows[method]
                acc = value([r[condition]["accuracy"] for r in model],100)
                concept = value([r[condition][metric] for r in model],100 if name=="cub" else 1,2 if name=="cub" else 3)
                nec = value([r["original"]["NEC"] for r in model])
                sections[key].append(f"| {EMOJIS[method]} {LABELS[method]} | {acc} | "+
                                     (f"mAP {concept}%" if name=="cub" else f"r {concept}")+f" | {nec} |")
            acc_gain = np.mean([n[condition]["accuracy"]-m[condition]["accuracy"] for m,n in zip(rows["mean"],rows["noise"])])*100
            concept_gain = np.mean([n[condition][metric]-m[condition][metric] for m,n in zip(rows["mean"],rows["noise"])])
            concept_text = f"{concept_gain*100:+.2f} 个百分点" if name=="cub" else f"{concept_gain:+.3f} r"
            sections[key] += ["",f"**结论。** 噪声训练相对均值训练的分类变化为 **{acc_gain:+.2f} 个百分点**，概念指标变化为 **{concept_text}**。",""]
    spatial_lines = ["| 数据集 | 已完成 seed | 背景方差 / 主体方差 | 主体均值 / 背景均值 | 部位均值 / 主体其他均值 |",
                     "|---|---|---:|---:|---:|"]
    for name in NAMES:
        measured = []
        for seed in SEEDS:
            source = ROOT/name/f"noise_seed{seed}/spatial_statistics.json"
            if not source.exists():
                continue
            item = json.loads(source.read_text())
            pairs = item["rows"]
            local = [r for r in pairs if "part_mean" in r]
            assert len(pairs)==item["positive_pairs"] and len(local)==item["local_pairs"] and item["n_images"]==500
            for group,num,den,key in [(pairs,"background_variance","foreground_variance","background_foreground_variance"),
                                      (pairs,"foreground_mean","background_mean","foreground_background_mean"),
                                      (local,"part_mean","other_mean","part_other_mean")]:
                ratio = np.mean([r[num] for r in group])/max(np.mean([r[den] for r in group]),1e-12)
                assert np.isclose(ratio,item["ratios"][key],atol=2e-6,rtol=0)
            measured.append({k:v for k,v in item.items() if k!="rows"})
            verified["source_sha256"][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        if measured:
            verified["spatial_datasets"][name] = dict(concepts=json.loads((ROOT/name/"cache/concepts.json").read_text()),spatial=measured)
            values = [mean_std([r["ratios"][key] for r in measured])+" 倍" for key in
                      ["background_foreground_variance","foreground_background_mean","part_other_mean"]]
            spatial_lines.append(f"| {TITLES[name]} | "+"、".join(str(r["seed"]) for r in measured)+" | "+" | ".join(values)+" |")
    if verified["spatial_datasets"]:
        conclusions = []
        for name,data in verified["spatial_datasets"].items():
            background = np.mean([r["ratios"]["background_foreground_variance"] for r in data["spatial"]])
            part = np.mean([r["ratios"]["part_other_mean"] for r in data["spatial"]])
            conclusions.append(f"{TITLES[name]} 的背景方差为主体的 {background:.2f} 倍，概念部位均值偏移为主体其他部位的 {part:.2f} 倍")
        sections["a"] = ["#### CUB 与 CIFAR 扩展：已完成的空间统计","",
            "**目的与设置。** 每个噪声模型使用固定 500 张验证图，测量 RGB 噪声在背景、主体及概念部位的强度；"
            "CUB 使用原生主体分割与部位点区域，CIFAR 使用 SAM2 主体与 CLIP 概念区域。","",*spatial_lines,"",
            "**结论。** "+"；".join(conclusions)+"。",""]
    if verified["original_models"]:
        sections["b"] += ["#### 已完成的空间模型原图结果","",
            "**目的与设置。** 使用冻结的 CLIP ViT-L/14，在 RGB 像素加入均值偏移并重新编码，测量每轮在完整官方测试集上的分类、概念指标及 NEC。","",
            "| 数据集 | seed | 方法 | 测试图数 | 分类准确率（%）↑ | 概念指标 ↑ | NEC ↓ |","|---|---:|---|---:|---:|---:|---:|"]
        conclusions = []
        for row in verified["original_models"]:
            name,method,seed = row["dataset"],row["method"],row["seed"]
            metrics = row["original"]
            concept = f"mAP {metrics['mAP']*100:.2f}%" if name=="cub" else f"r {metrics['concept_correlation']:.3f}"
            sections["b"].append(f"| {TITLES[name]} | {seed} | {EMOJIS[method]} {LABELS[method]} | {row['test_n']:,} | {metrics['accuracy']*100:.2f} | {concept} | {metrics['NEC']:.2f} |")
            baseline = json.loads((ROOT/name/f"lf_cbm_seed{seed}/original_results.json").read_text())["original"]["accuracy"]
            conclusions.append(f"{TITLES[name]} seed {seed} {LABELS[method]}的分类相对 LF-CBM 为 {(metrics['accuracy']-baseline)*100:+.2f} 个百分点")
        sections["b"] += ["","**结论。** "+"；".join(conclusions)+"。",""]
    background_data = dict(verified["datasets"])
    background_baselines = {}
    for name in ["cifar10","cifar100"]:
        paths = [ROOT/name/f"lf_cbm_seed{seed}/results.json" for seed in SEEDS]
        if not all(p.exists() and "background" in json.loads(p.read_text()) for p in paths):
            continue
        _,arrays,_,labels,_,protocol = load(name)
        ids = np.asarray(protocol["background_ids"])
        fingerprint = json.loads((ROOT/name/"background/COMPLETE.json").read_text())["sha256"]
        rows = []
        for seed,path in zip(SEEDS,paths):
            row = json.loads(path.read_text())
            assert row["intervention"]["sha256"]==fingerprint
            checkpoint = torch.load(path.parent/"model.pt",map_location="cpu",weights_only=False)
            scores = np.load(path.parent/"background_scores.npy")
            logits = np.load(path.parent/"background_class_logits.npy")
            assert scores.shape==(1000,len(arrays["text"])) and np.isfinite(scores).all()
            assert np.allclose(logits,head_predict(checkpoint["head"],scores),atol=1e-5)
            measured = concept_metrics(name,np.array(arrays["attrs"][ids]),scores)
            measured["accuracy"] = float((logits.argmax(1)==labels[ids]).mean())
            assert all(np.isclose(v,row["background"][key],atol=2e-6,rtol=0) for key,v in measured.items())
            rows.append(dict(seed=seed,original=row["original"],background=measured))
            verified["source_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        background_baselines[name] = rows
        if name not in background_data:
            background_data[name] = dict(models={"lf_cbm":rows})
    verified["background_baselines"] = background_baselines
    if background_baselines:
        sections["c"] += ["#### CIFAR：已完成的三轮换背景基线","",
            "**目的与设置。** 在 CIFAR-10、CIFAR-100 各固定 1,000 张换背景测试图上，"
            "测量冻结 CLIP ViT-L/14 的 LF-CBM 表现。主体像素保留，背景来自其他类别验证图中的纯背景片区。种子为 42、43、44。","",
            "| 数据集 | 方法 | 换背景分类准确率（%）↑ | CLIP 概念一致性 r ↑ |","|---|---|---:|---:|"]
        conclusions = []
        for name,rows in background_baselines.items():
            acc = mean_std([r["background"]["accuracy"] for r in rows],100)
            corr = mean_std([r["background"]["concept_correlation"] for r in rows],1,3)
            sections["c"].append(f"| {TITLES[name]} | 🔵 LF-CBM | {acc} | {corr} |")
            drop = np.mean([r["original"]["accuracy"]-r["background"]["accuracy"] for r in rows])*100
            conclusions.append(f"{TITLES[name]} 换背景后的分类准确率比原图低 {drop:.2f} 个百分点")
        sections["c"] += ["","**指标含义。** 准确率是认对类别的图片比例；r 是与原图 CLIP 概念软分数的平均相关系数，越接近 1 表示概念分数越一致。","",
                          "**结论。** "+"；".join(conclusions)+"。",""]
    if verified["datasets"]:
        spatial_figure(dict(datasets=verified["spatial_datasets"]),"cross_progress_spatial")
        classification_figure(verified,"cross_progress_classification")
        background_figure(dict(datasets=background_data),"cross_progress_background")
        for key,group in zip("abc",["spatial","classification","background"]):
            caption = ("四列依次为原图、均值偏移幅度、噪声方差和实际 RGB 加噪图；示例使用 seed 42 模型、固定采样种子 1207。"
                       if key=="a" else "蓝色 LF-CBM，绿色监督 CBM，紫色均值训练，橙色噪声训练；黑点表示每轮结果。")
            sections[key][:0] = [f"![实验 {key} 扩展：已完成的空间模型对照](reports/main/cross_progress_{group}.png)",
                                "","**图注。** 统计结果使用下表已完成的轮次。"+caption,""]
        text = text.replace("### b. 原图分类与概念识别","\n".join(sections["a"])+"\n### b. 原图分类与概念识别",1)
        text = text.replace("### c. 换背景后是否更容易认对","\n".join(sections["b"])+"\n### c. 换背景后是否更容易认对",1)
        text += "\n"+"\n".join(sections["c"])
    elif verified["original_models"]:
        text = text.replace("### c. 换背景后是否更容易认对","\n".join(sections["b"])+"\n### c. 换背景后是否更容易认对",1)
    if verified["datasets"] or verified["original_models"]:
        write_json(OUTPUT/"cross_progress.json",verified)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-only",action="store_true")
    parser.add_argument("--baseline-progress",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    style()
    if args.baseline_progress:
        baseline_progress()
        return
    result = collect()
    if not args.text_only:
        spatial_figure(result)
        classification_figure(result)
        background_figure(result)
    extend_report(result)
    print("Extended the single EXPERIMENT_REPORT.md with CUB/CIFAR a-b-c results",flush=True)


if __name__=="__main__":
    main()
