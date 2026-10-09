"""Validation-set thresholds at matched target recall; no test calibration."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from .data import ATTRS
from .train_spatial import load_cache


def main():
    arrays,splits,labels,records=load_cache('artifacts/rival_vitl14')
    yv=np.array(arrays['attrs'][splits['val']]);yt=np.array(arrays['attrs'][splits['test']]);results={}
    names=['clean','local_clean','mean_independent','noise_independent','mask_sequential','mean_sequential','noise_sequential','noise_joint','mean_separate_heads','noise_separate_heads',
           'local_mean','local_noise','full_joint_mean','full_joint_noise','region_clean','region_mean','region_noise']
    for name in names:
        run=Path('artifacts/spatial_clean' if name=='clean' else 'artifacts/spatial_local' if name=='local_clean' else 'artifacts/spatial_region' if name=='region_clean' else 'artifacts/'+name)
        if not (run/'results.json').exists():continue
        if name in ['clean','local_clean','region_clean']:
            val=1/(1+np.exp(-np.clip(np.load(run/'val_logits.npy'),-30,30)))
            test=1/(1+np.exp(-np.clip(np.load(run/'test_logits.npy'),-30,30)))
        else:val=np.load(run/'val_probs.npy');test=np.load(run/'test_probs.npy')
        thresholds=np.array([np.quantile(val[yv[:,k]==1,k],.1) for k in range(18)])
        pred=test>=thresholds;tp=(pred*yt).sum(0);fp=(pred*(1-yt)).sum(0);fn=((1-pred)*yt).sum(0)
        precision=tp/np.maximum(tp+fp,1);recall=tp/np.maximum(tp+fn,1)
        ap=average_precision_score(yt,test,average=None)
        results[name]=dict(macro_precision=float(precision.mean()),macro_recall=float(recall.mean()),target_validation_recall=.9,
                           thresholds=thresholds.tolist(),per_concept=[dict(name=attr,AP=float(ap[k]),precision=float(precision[k]),recall=float(recall[k])) for k,attr in enumerate(ATTRS)])
    Path('reports/spatial_noise_v1/concept_audit.json').write_text(json.dumps(results,indent=2));print({k:(v['macro_precision'],v['macro_recall']) for k,v in results.items()},flush=True)


if __name__=='__main__':main()
