"""Verify published numbers against saved predictions and the actual report."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score

from .train_spatial import load_cache, head_predict


def main():
    root=Path('reports/main');report=Path('EXPERIMENT_REPORT.md')
    result=json.loads((root/'results.json').read_text())
    arrays,splits,labels,_=load_cache('artifacts/rival_vitl14')
    checks=[]

    def close(actual,expected,description,tolerance=1e-7):
        assert abs(float(actual)-float(expected))<tolerance,(description,actual,expected)
        checks.append(description)

    def native(scores,ids):
        assert scores.shape==(len(ids),18) and np.isfinite(scores).all()
        return average_precision_score(np.array(arrays['attrs'][ids]),scores,average='macro')

    reference=np.load('artifacts/background_intervention_region/intervention.npz')
    reference_ids=reference['ids'];reference_donors=reference['donors']
    assert len(reference_ids)==1000
    assert result['seeds']==[42,43,44] and result['rounds']==3
    assert len(splits['test'])==5286 and len(splits['val'])==2111 and len(splits['train'])==18987
    for seed,prefix in [(42,'region'),(43,'region_seed43'),(44,'region_seed44')]:
        run=Path('artifacts')/(prefix+'_noise')
        row=next(r for r in result['classification'] if r['seed']==seed)
        scores=np.load(run/'test_mean_probs.npy')
        logits=np.load(run/'test_mean_class_logits.npy')
        head=torch.load(run/'head.pt',map_location='cpu',weights_only=False)
        assert tuple(head['weight'].shape)==(10,18)
        assert np.allclose(logits,head_predict(head,scores),atol=1e-6)
        close((logits.argmax(1)==labels[splits['test']]).mean(),row['accuracy'],f'{seed}: original accuracy from 5286 predictions')
        close(native(scores,splits['test']),row['concept_mAP'],f'{seed}: original native mAP from 18-score predictions')
        close((head['weight'].abs()>1e-5).sum(1).float().mean(),row['NEC'],f'{seed}: NEC from actual 10x18 classifier')
        spatial=json.loads((run/'localization_audit.json').read_text())
        pairs=spatial['per_pair'];local=[r for r in pairs if 'part_mu_rms' in r]
        assert len(pairs)==9022 and len(local)==4080
        summary=next(r for r in result['spatial'] if r['seed']==seed)
        for a,b,key,pop in [('bg_variance','fg_variance','sigma2_bg_over_fg',pairs),
            ('fg_mu_rms','bg_mu_rms','mu_fg_over_bg',pairs),
            ('part_mu_rms','other_object_mu_rms','local_part_mu_over_other_object',local)]:
            close(np.mean([p[a] for p in pop])/np.mean([p[b] for p in pop]),summary[key],f'{seed}: {key} from individual regions')
        folder='background_intervention_region' if seed==42 else f'background_intervention_region_seed{seed}'
        bg=Path('artifacts')/folder
        saved=np.load(bg/'intervention.npz')
        assert np.array_equal(saved['ids'],reference_ids) and np.array_equal(saved['donors'],reference_donors)
        # All rounds use the exact same manipulated pixels, not just the same IDs.
        assert np.array_equal(saved['shifted'],reference['shifted'])
        background=next(r for r in result['background'] if r['seed']==seed)
        for mode,key,name in [('mean','mean_trained',prefix+'_mean'),
            ('noise','noise_trained_mean_inference',prefix+'_noise_same_checkpoint_mean')]:
            scores=np.load(bg/(name+'_background_swap_probs.npy'))
            correct=np.load(bg/(name+'_background_swap_correct.npy'))
            own_head=torch.load(Path('artifacts')/(prefix+'_'+mode)/'head.pt',map_location='cpu',weights_only=False)
            assert np.array_equal(correct,head_predict(own_head,scores).argmax(1)==labels[reference_ids])
            close(correct.mean(),background[key]['accuracy'],f'{seed}/{mode}: background accuracy from 1000 predictions')
            close(native(scores,reference_ids),background[key]['mAP'],f'{seed}/{mode}: background mAP from native predictions')
    for method in ['supervised_cbm','lf_cbm']:
        rows=result['baselines'][method];assert sorted(r['seed'] for r in rows)==[42,43,44]
        for row in rows:
            path=Path('artifacts/paper_baselines')/f"{method}_seed{row['seed']}"
            ck=torch.load(path/'model.pt',map_location='cpu',weights_only=False)
            assert tuple(ck['head']['weight'].shape)==(10,18)
            for condition,ids in [('test',splits['test']),('background',reference_ids)]:
                metrics=row['original' if condition=='test' else 'background']
                scores=np.load(path/(condition+'_scores.npy'))
                logits=np.load(path/(condition+'_class_logits.npy'))
                assert np.allclose(logits,head_predict(ck['head'],scores),atol=1e-6)
                close((logits.argmax(1)==labels[ids]).mean(),metrics['accuracy'],f"{method}/{row['seed']}/{condition}: accuracy from saved logits")
                close(native(scores,ids),metrics['mAP'],f"{method}/{row['seed']}/{condition}: mAP from saved native scores")
    for metric in ['accuracy','mAP']:
        mean=np.mean([r['mean_trained'][metric] for r in result['background']])
        noise=np.mean([r['noise_trained_mean_inference'][metric] for r in result['background']])
        close((noise-mean)*100,result['background_gain_pp'][metric],f'paired three-round {metric} gain')
    for filename,expected in result['source_sha256'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==expected,filename
    text=report.read_text()
    assert re.findall(r'^## (.+)$',text,re.M)==['Abstract','Background（为什么要做）','原理与方法','主要实验和结果']
    assert re.findall(r'^### ([abc])\.',text,re.M)==['a','b','c']
    images=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text)
    expected_images=9 if (root/'cross_results.json').exists() else (
        6+int('reports/main/cross_baselines.png' in images)+
        3*int('reports/main/cross_progress_classification.png' in images))
    assert len(images)==expected_images
    for name in images:
        path=report.parent/name
        with Image.open(path) as image:image.verify()
        assert path.with_suffix('.svg').exists() and path.with_suffix('.pdf').exists()
    assert sorted(str(p) for p in Path('.').rglob('*.md') if 'artifacts' not in p.parts and '.git' not in p.parts)==['EXPERIMENT_REPORT.md']
    shape=result['tensor_shapes']
    assert shape['Z']==[2,257,768] and shape['H']==[2,256,128] and shape['concept_logits']==[2,18]
    assert shape['mu_RGB']==[2,3,224,224] and shape['sigma_RGB']==[2,3,224,224]
    evidence=dict(passed=True,checks=checks,rounds=3,baseline_methods=2,report=str(report),
        figures=images,report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        verification='Raw per-image native scores, class logits, classifier weights, per-region statistics, and identical background pixels checked.',
        background_gain_pp=result['background_gain_pp'])
    (root/'verification.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    print(f"Verified {len(checks)} RIVAL10 numerical claims, three rounds, two baselines, {len(images)} figures and the single paper report.",flush=True)


if __name__=='__main__':main()
