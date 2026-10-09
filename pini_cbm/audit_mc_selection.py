"""Verify MC selection against saved validation predictions, not test scores."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from .train_spatial import load_cache


def main():
    arrays,splits,labels,_=load_cache('artifacts/rival_vitl14')
    yv=np.array(arrays['attrs'][splits['val']]);yt=np.array(arrays['attrs'][splits['test']]);audit=[]
    groups=[['noise_independent','noise_sequential','noise_joint','noise_separate_heads'],['local_noise','full_joint_noise']]
    for group in groups:
        if not all((Path('artifacts')/name/'results.json').exists() for name in group):continue
        scores={name:float(average_precision_score(yv,np.load(Path('artifacts')/name/'val_probs.npy'),average='macro')) for name in group}
        selected=max(scores,key=lambda name:(scores[name],name));run=Path('artifacts')/selected
        result=json.loads((run/'results.json').read_text())
        for name,value in scores.items():
            recorded=json.loads((Path('artifacts')/name/'results.json').read_text())['concepts']['val']['mAP']
            assert np.isclose(value,recorded,atol=1e-12,rtol=0)
        if 'MC8' not in result:continue
        probs=np.load(run/'test_MC8_probs.npy');logits=np.load(run/'test_MC8_class_logits.npy')
        assert probs.shape==yt.shape
        accuracy=float((logits.argmax(1)==labels[splits['test']]).mean());ap=float(average_precision_score(yt,probs,average='macro'))
        assert np.isclose(accuracy,result['MC8']['accuracy'],atol=1e-12,rtol=0)
        assert np.isclose(ap,result['MC8']['mAP'],atol=1e-12,rtol=0)
        description='highest complete validation native mAP among '+','.join(group)+'; original fitted head, no MC test tuning'
        if result['MC8'].get('selection')!=description:
            result['MC8']['selection']=description
            temp=run/'results.selection.tmp';temp.write_text(json.dumps(result,indent=2));temp.replace(run/'results.json')
        audit.append(dict(selection_group=group,validation_mAP=scores,selected=selected,
                          selection_uses_test=False,MC8_metrics_verified_from_saved_predictions=True,
                          head_sha256=hashlib.sha256((run/'head.pt').read_bytes()).hexdigest()))
    Path('reports/spatial_noise_v1/MC_selection.json').write_text(json.dumps(audit,indent=2));print(audit,flush=True)


if __name__=='__main__':main()
