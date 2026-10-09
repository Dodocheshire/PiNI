"""Paired sampling and separately trained mean controls for official PiNI."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .uncertainty import ci_correct


def main():
    result={}
    for kind in ['main','image']:
        for suffix in ['seed1','fp16_seed1']:
            run=Path('artifacts')/('pets_'+kind+'_'+suffix)
            if not (run/'results.json').exists():continue
            split=json.loads((run/'split.json').read_text());y=np.array([r['label'] for r in split['test']]);pred={}
            for name in ['MC1','MC8','mean','zero_shot']:
                p=run/('test_'+name+'_probs.npy')
                if p.exists():pred[name]=np.load(p).argmax(1)==y
            row=dict(n_test=len(y),comparison={},precision_protocol=json.loads((run/'results.json').read_text())['precision'])
            for a,b in [('MC8','MC1'),('MC1','mean'),('MC8','mean'),('MC1','zero_shot'),('MC8','zero_shot')]:
                if a in pred and b in pred:row['comparison'][a+'_minus_'+b]=ci_correct(pred[a],pred[b])
            if suffix=='seed1':
                mean=Path('artifacts')/('pets_'+kind+'_mean_seed1')
                if (mean/'results.json').exists():
                    assert json.loads((mean/'split.json').read_text())==split
                    mp=np.load(mean/'test_MC1_probs.npy').argmax(1)==y
                    for key in ['MC1','MC8','mean']:
                        if key in pred:row['comparison'][key+'_minus_independently_trained_mean']=ci_correct(pred[key],mp)
                    row['independently_trained_mean_accuracy']=float(mp.mean())
            result[kind+'_'+suffix]=row;print(kind+'_'+suffix,row,flush=True)
    out=Path('reports/pini_official');out.mkdir(parents=True,exist_ok=True)
    (out/'paired_controls.json').write_text(json.dumps(dict(results=result,interpretation='Paired image bootstrap within each precision/batch protocol; no cross-protocol causal contrast, no training-seed variability. MC8 is added inference diagnostic.'),indent=2))


if __name__=='__main__':main()
