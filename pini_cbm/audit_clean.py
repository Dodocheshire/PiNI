"""Semantic discretization and matched-head controls, CPU-only."""
import json
from pathlib import Path
import numpy as np
import torch
from .train_spatial import load_cache,fit_head,head_predict
from .train_noise import fixed_subset


def main():
    torch.set_num_threads(2)
    cache=Path('artifacts/rival_vitl14');out=Path('artifacts/spatial_clean')
    arrays,splits,labels,records=load_cache(cache)
    logits={s:np.load(out/(s+'_logits.npy')) for s in splits}
    probs={s:1/(1+np.exp(-np.clip(v,-30,30))) for s,v in logits.items()}
    result={}
    for kind in ['binary','matched_6000_continuous']:
        if kind=='binary':
            values={s:(v>=.5).astype('float32') for s,v in probs.items()};train=splits['train'];trainvalues=values['train']
        else:
            values=probs;train=fixed_subset(splits['train'],labels,6000)
            positions=np.searchsorted(splits['train'],train);trainvalues=values['train'][positions]
        head,curve,i=fit_head(trainvalues,labels[train],values['val'],labels[splits['val']],device='cpu')
        pred=head_predict(head,values['test']);np.save(out/(kind+'_test_class_logits.npy'),pred)
        torch.save(head,out/(kind+'_head.pt'))
        result[kind]=dict(test_accuracy=float((pred.argmax(1)==labels[splits['test']]).mean()),NEC=curve[i]['NEC'],head_grid=curve,head_selected=i,head_train_size=len(train))
        print(kind,result[kind],flush=True)
    valid=np.load(cache/'mask_valid.npy');result['missing_foreground_masks']=int((~valid[:,18]).sum())
    result['interpretation']='Continuous concept probabilities can retain class-specific information beyond native binary attributes; report both.'
    (out/'semantic_audit.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
