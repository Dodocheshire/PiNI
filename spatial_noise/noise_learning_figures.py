"""Matched test comparisons and unamplified RGB views for the chosen noise run."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np
import torch

from .cub_noise import inputs
from .data import write_json
from .grounding import GroundedNoiseGenerator
from .rn18_datasets import Experiment

OUT = Path('reports/main')
COLORS = ['#A76BC3', '#E58B2A', '#56A76C', '#357ABD']


def read(path):
    return json.loads(path.read_text())


def generator(folder, device):
    state = torch.load(folder/'best.pt', map_location=device, weights_only=True)
    model = GroundedNoiseGenerator(regions=12, mean_limit=.03, std_limit=.1,
                                  spatial_noise_prior=False).to(device)
    model.load_state_dict(state['generator'])
    return model.eval(), state


def probabilities(folder, view):
    return torch.from_numpy(np.load(folder/(view+'_concepts.npy'))).sigmoid().numpy()


def bootstrap(difference, rng, replicates=2000):
    samples = []
    for _ in range(0, replicates, 100):
        indices = rng.integers(len(difference), size=(100, len(difference)))
        samples.extend(difference[indices].mean(1))
    return {'mean': float(difference.mean()),
            'ci95': np.quantile(samples, [.025, .975]).tolist(),
            'replicates': replicates, 'unit': 'paired test image'}


@torch.no_grad()
def spatial(model, arrays, grounded, ids, device):
    totals = np.zeros(6)
    count = 0
    eligible = np.asarray(arrays['background_eligible'][ids])
    for offset in range(0, len(ids), 64):
        selected = ids[offset:offset+64]
        region = {k: torch.as_tensor(np.array(v[selected]), device=device).float()
                  for k, v in grounded['changed'].items()}
        mu, sigma = model.distribution(region)
        fg = torch.as_tensor(np.array(arrays['foreground'][selected]), device=device).float()[:, None]
        columns = []
        for mask in [fg, 1-fg]:
            denominator = mask.sum((1, 2, 3)).clamp_min(1)
            columns.extend([(sigma.square()*mask).sum((1, 2, 3))/denominator,
                            ((sigma>.02)*mask).sum((1, 2, 3))/denominator,
                            (mu.abs().mean(1, keepdim=True)*mask).sum((1, 2, 3))/denominator])
        values = torch.stack(columns, 1).cpu().numpy()[eligible[offset:offset+64]]
        totals += values.sum(0)
        count += len(values)
    return dict(zip(['variance_fg', 'coverage_fg', 'mean_fg',
                     'variance_bg', 'coverage_bg', 'mean_bg'], (totals/count).tolist()))


def comparison(e, mean_folder, noise_folder, device):
    arrays, splits, _, grounded = inputs(experiment=e)
    ids = splits['test']
    eligible = np.asarray(arrays['background_eligible'][ids])
    names = ['Simple CBM', 'Background training', 'Paired mean', 'Paired noise']
    result = {'models': {}, 'paired_test_images': int(eligible.sum()),
              'seed': 42, 'spatial_prior': False, 'coverage_threshold_sigma': .02}
    result['models'][names[0]] = read(e.root/'simple_refit/seed42/paired_results.json')
    result['models'][names[1]] = read(e.root/'prior_off/seed42/refit/clean/results.json')['test']
    for name, folder in zip(names[2:], [mean_folder, noise_folder]):
        values = read(folder/'results.json')['test']
        model, state = generator(folder, device)
        values['checkpoint_nec'] = float((state['cbm']['classifier.weight'].abs()>1e-5).sum(1).float().mean())
        values['probability_std'] = {view: float(probabilities(folder, view).std(0).mean())
                                     for view in ['original', 'changed']}
        if name == 'Paired noise':
            values['spatial_audit'] = spatial(model, arrays, grounded, ids, device)
        result['models'][name] = values
    labels = np.asarray(arrays['labels'][ids])
    result['comparisons'] = {}
    for baseline, folder in [('Paired mean', mean_folder)]:
        metrics = {}
        rng = np.random.default_rng(42)
        for view in ['original', 'changed']:
            a = np.load(noise_folder/(view+'_logits.npy')).argmax(1)==labels
            b = np.load(folder/(view+'_logits.npy')).argmax(1)==labels
            use = eligible if view=='changed' else np.ones(len(ids), dtype=bool)
            metrics[view+'_accuracy_pp'] = bootstrap((a.astype(float)-b.astype(float))[use]*100, rng)
        new_drift = np.abs(probabilities(noise_folder, 'original')-probabilities(noise_folder, 'changed')).mean(1)
        old_drift = np.abs(probabilities(folder, 'original')-probabilities(folder, 'changed')).mean(1)
        metrics['concept_drift_pp'] = bootstrap((new_drift-old_drift)[eligible]*100, rng)
        metrics['concept_drift_relative_reduction'] = float(1-new_drift[eligible].mean()/old_drift[eligible].mean())
        result['comparisons']['Paired noise vs '+baseline] = metrics
    destination = e.root/'noise_exploration'
    destination.mkdir(exist_ok=True)
    write_json(destination/'current_results.json', result)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    rows = [result['models'][name] for name in names]
    x = np.arange(4)
    for j, view in enumerate(['original', 'changed']):
        values = [r[view]['accuracy']*100 for r in rows]
        bars = axes[0, 0].bar(x+(j-.5)*.34, values, width=.34, color=COLORS,
                               alpha=.45 if j==0 else 1, hatch='//' if j==0 else '')
        for bar, value in zip(bars, values):
            axes[0, 0].text(bar.get_x()+bar.get_width()/2, value+.65, f'{value:.2f}', ha='center', fontsize=9)
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].set_title('Hatched: original | Solid: changed background')
    axes[0, 0].set_ylabel('Classification accuracy (%)')
    drift = [r['concept_drift_mae']*100 for r in rows]
    bars = axes[0, 1].bar(x, drift, color=COLORS, width=.55)
    for bar, value in zip(bars, drift):
        axes[0, 1].text(bar.get_x()+bar.get_width()/2, value+.06, f'{value:.2f}', ha='center')
    axes[0, 1].set_ylim(0, max(drift)*1.25)
    axes[0, 1].set_ylabel('Concept probability drift (percentage points)')
    axes[0, 1].set_title('Paired background change | lower is better')
    for ax in axes[0]:
        ax.set_xticks(x, [n.replace(' ', '\n') for n in names])
    audit = result['models']['Paired noise']['spatial_audit']
    for ax, field, scale, title, ylabel in [
        (axes[1, 0], 'variance', 255., 'Learned noise scale | current paired-noise model', 'RMS noise scale (RGB levels)'),
        (axes[1, 1], 'coverage', 100., 'Pixels with sigma > 0.02', 'Noise coverage (%)'),
    ]:
        values = [np.sqrt(audit[field+'_'+part])*scale if field=='variance'
                  else audit[field+'_'+part]*scale for part in ['fg', 'bg']]
        bars = ax.bar([0, 1], values, color=['#8793A0', '#357ABD'], width=.55)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value+.3, f'{value:.2f}', ha='center')
        ax.set_xticks([0, 1], ['Subject', 'Background'])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.margins(y=.18)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('RIVAL10 | ResNet18 | 18 concepts | train+val refit | seed 42 | prior OFF', fontsize=16)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/'rival10_noise_learning_results.png', dpi=180)
    plt.close(fig)
    return result


@torch.no_grad()
def visualization(e, folder, device):
    arrays, splits, _, grounded = inputs(experiment=e)
    indices = np.asarray(read(e.root/'visualization.json')['cache_indices'])
    region = {k: torch.as_tensor(np.array(v[indices]), device=device).float()
              for k, v in grounded['changed'].items()}
    rgb = torch.as_tensor(np.array(arrays['rgb_changed'][indices]), device=device).float()/255
    model, _ = generator(folder, device)
    torch.manual_seed(4242)
    image, mu, sigma = model(rgb, region, 'noise')
    noisy = image.clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
    means = mu.abs().mean(1).cpu().numpy()
    variances = sigma.square()[:, 0].cpu().numpy()
    mu_max, var_max = float(means.max()), float(variances.max())
    fig, axes = plt.subplots(5, 5, figsize=(15, 13), layout='constrained')
    titles = ['Original image', 'Changed background', 'Mean + noise RGB', 'Mean magnitude |mu|', 'Noise variance sigma^2']
    for row, index in enumerate(indices):
        images = [np.asarray(arrays['rgb_original'][index]).transpose(1, 2, 0), rgb[row].cpu().numpy().transpose(1, 2, 0),
                  noisy[row], means[row], variances[row]]
        for col, (ax, pixels) in enumerate(zip(axes[row], images)):
            if col < 3:
                ax.imshow(pixels)
            else:
                im = ax.imshow(pixels, cmap='magma' if col==3 else 'viridis', vmin=0,
                               vmax=mu_max if col==3 else var_max)
                ax.contour(np.asarray(arrays['foreground'][index]), levels=[.5], colors=['white'], linewidths=.7)
                if row==4:
                    fig.colorbar(im, ax=axes[:, col], shrink=.55, pad=.02)
            ax.set_xticks([])
            ax.set_yticks([])
            if row==0:
                ax.set_title(titles[col], fontsize=10)
        axes[row, 0].set_ylabel(f'Test ID {index}', fontsize=10)
    fig.suptitle('RIVAL10 | current paired-noise model | actual RGB perturbations, no amplification\n'
                 'Five fixed test images | spatial prior OFF | white outline: subject', fontsize=14)
    fig.savefig(OUT/'rival10_noise_learning_visualization.png', dpi=170)
    plt.close(fig)
    write_json(e.root/'noise_exploration/visualization.json', {
        'indices': indices.tolist(), 'selection': 'same fixed seed-42 samples as previous report',
        'epsilon_seed': 4242, 'variance_color_max': var_max, 'mean_color_max': mu_max,
        'perturbation_amplified': False,
    })


@torch.no_grad()
def pipeline(e, label, noise_folder=None):
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    arrays, _, _, grounded = inputs(experiment=e)
    index = read(e.root/'visualization.json')['cache_indices'][0]
    model, _ = generator(noise_folder or e.root/'pair_explore/refit/noise', 'cpu')
    originals, perturbed, means, variances = [], [], [], []
    for view in ['original', 'changed']:
        rgb = torch.from_numpy(np.array(arrays['rgb_'+view][index:index+1])).float()/255
        regions = {k: torch.from_numpy(np.array(v[index:index+1])).float()
                   for k, v in grounded[view].items()}
        torch.manual_seed(4242)
        noisy, mu, sigma = model(rgb, regions, 'noise')
        originals.append(rgb[0].permute(1, 2, 0).numpy())
        perturbed.append(noisy[0].clamp(0, 1).permute(1, 2, 0).numpy())
        means.append(mu[0].abs().mean(0).numpy())
        variances.append(sigma[0, 0].square().numpy())
    boxes = np.load(e.root/'regions/changed/boxes.npy', mmap_mode='r')[index]
    valid = np.load(e.root/'regions/changed/valid.npy', mmap_mode='r')[index]
    feature = np.asarray(grounded['changed']['features'][index])
    with plt.rc_context({'font.family': 'WenQuanYi Zen Hei', 'mathtext.fontset': 'stix'}):
        fig, ax = plt.subplots(figsize=(22, 14))
        ax.set(xlim=(0, 22), ylim=(0, 14))
        ax.axis('off')
        blue, pink, green = '#DFEDF9', '#F8DFE7', '#E3F1DE'
        def text(x,y,value,size=13,**kw):
            ax.text(x,y,value,fontsize=size,ha='center',va='center',**kw)
        def box(x,y,w,h,value,color,size=13):
            ax.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor='#334155',linewidth=1.3,zorder=3))
            text(x+w/2,y+h/2,value,size,zorder=4)
        def arrow(a,b,color='#334155',dashed=False,bend=0):
            ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=16,
                linewidth=1.6,color=color,linestyle='--' if dashed else '-',
                connectionstyle=f'arc3,rad={bend}',zorder=2))
        def photo(pixels,x,y,w,title,cmap=None):
            ax.imshow(pixels,extent=(x,x+w,y,y+w),cmap=cmap,zorder=3,aspect='auto')
            ax.add_patch(Rectangle((x,y),w,w,fill=False,edgecolor='#334155',linewidth=1.1,zorder=4))
            text(x+w/2,y-.27,title,12)
        def vector(x,y,values,title,w=1.15,h=.7):
            for j,value in enumerate(values):
                ax.add_patch(Rectangle((x+j*w/len(values),y),w/len(values)*.8,h*value,
                                      facecolor='#53996A',zorder=4))
            ax.plot([x-.07,x-.07,x+w+.07,x+w+.07],[y+h,y,y,y+h],color='#334155',lw=1)
            text(x+w/2,y-.23,title,11)
        for x,y,w,h,title in [(.25,7.65,21.5,5.9,'（a）图像主路与区域信息旁路'),
                              (.25,.25,11.65,7.05,'（b）成对训练：同一主体，两个背景'),
                              (12.2,.25,9.55,7.05,'（c）推理：只生成一张扰动图')]:
            ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='#334155',linewidth=1.4))
            ax.text(x+.3,y+h-.28,title,fontsize=19,weight='bold',va='top')
        photo(originals[1],.85,10.45,2.35,'输入图片')
        # The unmodified RGB bypasses the detector and joins the learned perturbation.
        arrow((3.23,12.5),(14.15,12.5),bend=-.035)
        text(8.0,12.98,'保留原图像素，叠加学习到的均值与随机噪声',12)
        arrow((14.15,12.5),(14.15,12.61))
        # Region prompts are a note card, separate from the image input.
        ax.add_patch(Rectangle((.85,8.12),2.65,1.4,facecolor='#FFF6DB',edgecolor='#B99B52',zorder=3))
        ax.plot([1.08,1.08],[8.3,9.35],color='#D78E89',lw=1,zorder=4)
        text(2.17,9.24,'区域文字提示',14,weight='bold',zorder=4)
        text(2.17,8.85,'动物 · 头部 · 翅膀',12,zorder=4)
        text(2.17,8.46,'喙 · 眼睛 · 身体 …',12,zorder=4)
        box(4.05,8.45,2.65,1.1,'Grounding DINO\n冻结的区域标注模型',blue)
        arrow((2.03,10.43),(4.07,9.28),bend=.12)
        arrow((3.5,8.81),(4.05,8.81))
        # Reuse real detector boxes and image features, rather than invented annotations.
        for j in range(3):
            ax.imshow(feature[j*10],extent=(7.22+j*.12,8.3+j*.12,8.45+j*.12,9.53+j*.12),
                      cmap='Blues',aspect='auto',zorder=3+j)
        text(7.88,8.14,'图像特征',11)
        photo(originals[1],7.25,10.48,1.65,'区域框')
        for b,ok,col in zip(boxes,valid,plt.cm.tab10.colors*2):
            if ok:
                x1,y1,x2,y2=b
                ax.add_patch(Rectangle((7.25+x1*1.65,10.48+(1-y2)*1.65),
                    (x2-x1)*1.65,(y2-y1)*1.65,fill=False,edgecolor=col,linewidth=1.4,zorder=5))
        arrow((6.7,9.0),(7.2,9.0))
        arrow((5.38,9.55),(7.25,11.2),bend=-.15)
        box(9.35,8.6,2.65,1.05,'共享噪声生成器\n学习空间分布',pink)
        arrow((8.6,9.0),(9.35,9.0))
        arrow((8.9,10.9),(10.35,9.65),bend=.15)
        photo(means[1],9.65,10.58,1.48,'均值修正幅度',cmap='magma')
        photo(variances[1],11.35,10.58,1.48,'噪声方差',cmap='viridis')
        arrow((10.42,9.65),(10.42,10.55))
        arrow((11.35,9.65),(12.08,10.55))
        photo(perturbed[1],13.2,10.45,2.15,'均值＋噪声图')
        arrow((12.85,11.3),(13.18,11.3))
        text(13.45,9.64,r'$\tilde{x}=x+\mu+\sigma\odot\epsilon$',18)
        box(16.05,10.75,2.1,1.1,'ResNet18\n冻结图像编码器',blue)
        arrow((15.36,11.52),(16.05,11.52))
        vector(19.05,11.02,[.8,.15,.67,.93,.12,.43],'概念激活',w=1.65,h=.9)
        box(16.05,9.55,2.1,.65,'概念头',green)
        arrow((17.1,10.75),(17.1,10.22))
        arrow((18.15,9.88),(19.75,10.97),bend=.15)
        box(18.75,8.58,2.15,.95,'线性分类器\n输出类别',green)
        arrow((19.85,10.73),(19.85,9.55))
        text(13.3,8.0,'粉色：可训练扰动模块    蓝色：冻结模型    绿色：概念与分类',12)
        # Training is a fork/join diagram with two real RGB views and a shared module.
        for y,src,dst,title in [(4.45,originals[0],perturbed[0],'原图'),
                                (1.7,originals[1],perturbed[1],'换背景图')]:
            photo(src,.85,y,1.75,title)
            photo(dst,6.05,y,1.75,'扰动视图')
            arrow((2.62,y+.87),(3.35,3.9),bend=.1 if y>3 else -.1)
            arrow((5.3,3.9),(6.03,y+.87),bend=-.1 if y>3 else .1)
            arrow((7.8,y+.87),(8.36,3.9),bend=.1 if y>3 else -.1)
        box(3.35,3.35,1.95,1.1,'同一噪声生成器\n共享随机采样',pink,12)
        text(4.33,5.35,'两张图使用同一份 ε',12)
        box(8.35,3.35,1.9,1.1,'同一 ResNet18\n与概念头',blue,12)
        vector(10.43,5.13,[.8,.15,.67,.93,.12,.43],'原图概念',w=1.03)
        vector(10.43,2.32,[.78,.17,.65,.91,.13,.44],'换背景概念',w=1.03)
        arrow((10.25,4.1),(10.93,5.05),bend=.12)
        arrow((10.25,3.65),(10.93,3.05),bend=-.12)
        arrow((10.94,4.76),(10.94,3.3),'#BF3556',True)
        text(9.97,1.4,'① 约束概念一致',13,color='#BF3556',weight='bold')
        text(6.0,.76,'分类与概念监督 ＋ ① 概念一致性 ＋ ② 更广噪声覆盖奖励',13)
        arrow((4.3,1.16),(4.3,3.3),'#BF3556',True)
        text(4.34,2.12,'② 覆盖奖励',12,color='#BF3556',bbox={'facecolor':'white','edgecolor':'none','pad':2})
        # Inference bends through a single perturbed view, without a second image branch.
        photo(originals[1],12.85,4.55,1.8,'输入图片')
        box(15.5,4.85,2.35,1.0,'区域信息旁路\n＋共享噪声生成器',pink,12)
        arrow((14.67,5.43),(15.5,5.43))
        photo(perturbed[1],18.7,4.55,1.8,'唯一扰动视图')
        arrow((17.85,5.43),(18.68,5.43))
        box(18.6,2.35,2.45,1.0,'ResNet18\n＋概念头',blue)
        arrow((19.6,4.18),(19.6,3.37))
        vector(15.25,2.45,[.8,.15,.67,.93,.12,.43],'概念向量',w=1.75,h=.75)
        arrow((18.6,2.82),(17.15,2.82))
        box(12.85,2.35,1.85,1.0,'线性分类器\n鸟类',green)
        arrow((15.14,2.82),(14.73,2.82))
        fig.suptitle('共享空间噪声 CBM：概念一致性与噪声覆盖',fontsize=27,weight='bold',y=.989)
        fig.subplots_adjust(left=.01,right=.99,bottom=.015,top=.95)
        fig.savefig(OUT/'pipeline.png',dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mean-run', required=True, help='Path relative to rival10_aligned_rn18')
    parser.add_argument('--noise-run', required=True, help='Path relative to rival10_aligned_rn18')
    parser.add_argument('--label', default='Paired original / changed views')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    e = Experiment('rival10')
    comparison(e, e.root/args.mean_run, e.root/args.noise_run, args.device)
    visualization(e, e.root/args.noise_run, args.device)
    pipeline(e, args.label, e.root/args.noise_run)


if __name__ == '__main__':
    main()
