"""Shared class head fitted only on native GT concepts, not model confidences."""
import json
from pathlib import Path
import numpy as np
import torch
from .train_noise import fixed_subset
from .train_spatial import load_cache,fit_head,head_predict


def main():
    torch.set_num_threads(2);cache=Path('artifacts/rival_vitl14');out=Path('artifacts/native_semantic_head');out.mkdir(parents=True,exist_ok=True)
    arrays,splits,labels,records=load_cache(cache);ids=fixed_subset(splits['train'],labels,6000)
    if (out/'head.pt').exists() and (out/'fit.json').exists():
        head=torch.load(out/'head.pt',map_location='cpu',weights_only=False);fit=json.loads((out/'fit.json').read_text())
    else:
        head,curve,index=fit_head(np.array(arrays['attrs'][ids]),labels[ids],np.array(arrays['attrs'][splits['val']]),labels[splits['val']],device='cpu')
        torch.save(head,out/'head.pt');fit=dict(head_train_ids=ids.tolist(),head_train_size=len(ids),head_grid=curve,head_selected=index,
            NEC=curve[index]['NEC'],selection='native GT concept validation accuracy under NEC<=8',gt_inputs_at_inference=False)
        (out/'fit.json').write_text(json.dumps(fit,indent=2))
    yt=np.array(arrays['attrs'][splits['test']]);target=labels[splits['test']]
    oracle=head_predict(head,yt);result=dict(NEC=fit['NEC'],head_train_size=len(ids),oracle_accuracy=float((oracle.argmax(1)==target).mean()),models={})
    names=['clean','local_clean','mean_independent','noise_independent','mask_sequential','mean_sequential','noise_sequential','noise_joint','noise_separate_heads',
           'local_mean','local_noise','full_joint_mean','full_joint_noise']
    for name in names:
        run=Path('artifacts/spatial_clean' if name=='clean' else 'artifacts/spatial_local' if name=='local_clean' else 'artifacts/'+name)
        if not (run/'results.json').exists():continue
        if name in ['clean','local_clean']:probs=1/(1+np.exp(-np.clip(np.load(run/'test_logits.npy'),-30,30)))
        else:probs=np.load(run/'test_probs.npy')
        logits=head_predict(head,probs);binary_logits=head_predict(head,(probs>=.5).astype('float32'))
        correct=logits.argmax(1)==target;np.save(out/(name+'_correct.npy'),correct)
        np.save(out/(name+'_binary_correct.npy'),binary_logits.argmax(1)==target)
        result['models'][name]=dict(accuracy=float(correct.mean()),binary_accuracy=float((binary_logits.argmax(1)==target).mean()))
    result['interpretation']='Same sparse head fitted only on native training GT concepts for every learned model; no per-model probability/confidence fitting. Predictions may still carry class information through confidence or systematic concept errors, even under this common head.'
    (out/'results.json').write_text(json.dumps(result,indent=2));Path('reports/spatial_noise_v1/semantic_readout.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':main()
