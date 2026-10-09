"""Finish region-coupled recognition, then its matched RGB-noise experiment."""
import json
import subprocess
import sys
import time
from pathlib import Path


def wait(names):
    started=time.time()
    while not all((Path('artifacts')/name/'results.json').exists() for name in names):
        if time.time()-started>21600:raise SystemExit('Prerequisites require inspection')
        time.sleep(30)


def run(name,args,gpu=False):
    command=[sys.executable,'-u','-m']+args
    if gpu:command=['flock',str(Path('../D-Pi/artifacts/resources/gpu_work.lock').resolve())]+command
    print(json.dumps(dict(start=name,command=command)),flush=True)
    with (Path('artifacts')/(name+'.log')).open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise SystemExit(result.returncode)


def main():
    wait(['spatial_region'])
    # This CPU experiment began just before the serialization flag was added.
    # Complete its metadata once the trainer has finished all checkpoint writes.
    import numpy as np
    import torch
    from torch.nn import functional as F
    from .models import SpatialConcepts
    from .data import STD
    from .train_spatial import load_cache
    torch.set_num_threads(2);root=Path('artifacts/spatial_region')
    ck=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    assert ck.get('region_readout') is True
    if '_region_readout' not in ck['model']:
        ck['model']['_region_readout']=torch.tensor(True)
        temp=root/'best.architecture.tmp';torch.save(ck,temp);temp.replace(root/'best.pt')
    model=SpatialConcepts(torch.tensor(np.load('artifacts/rival_vitl14/text.npy')))
    model.load_state_dict(ck['model']);assert model.region_readout
    arrays,splits,_,_=load_cache('artifacts/rival_vitl14');ids=splits['train'][:8]
    tokens=torch.tensor(np.array(arrays['tokens'][ids]),dtype=torch.float32)
    targets=torch.tensor(np.array(arrays['attrs'][ids]),dtype=torch.float32)
    logits,_,_=model(tokens);loss=F.binary_cross_entropy_with_logits(logits,targets)
    grads=torch.autograd.grad(loss,[model.logit_weights,model.mask_bias.weight,model.fg[-1].weight])
    d=model.texts.shape[-1]
    audit=dict(region_readout=True,actual_training_images=len(ids),direct_CLS_weight_gradient_max=float(grads[0][:,:d].abs().max()),
               region_weight_gradient_norm=float(grads[0][:,d:].norm()),mask_gradient_from_concept_BCE=float(grads[1].norm()),
               foreground_gradient_from_concept_BCE=float(grads[2].norm()),gt_forward_inputs=False,
               limitation='CLIP patch tokens and predicted masks are contextualized; removing direct CLS does not guarantee all global information is absent.')
    assert audit['direct_CLS_weight_gradient_max']==0 and audit['region_weight_gradient_norm']>0
    assert audit['mask_gradient_from_concept_BCE']>0 and audit['foreground_gradient_from_concept_BCE']>0
    (root/'recognition_gradient_audit.json').write_text(json.dumps(audit,indent=2));print(audit,flush=True)
    run('audit_region_clean',['pini_cbm.audit_clean','--clean',str(root)])
    # Preserve the already queued major training-strategy comparison's priority.
    wait(['full_joint_mean','full_joint_noise'])
    # Set once from the paper's 0.1 normalized-image RMS, not test scores.
    mean_budget=.1*float(np.sqrt(np.mean(np.square(STD))))
    for mode in ['mean','noise']:
        name='region_'+mode
        if (Path('artifacts')/name/'results.json').exists():continue
        run(name,['pini_cbm.train_noise','--clean',str(root),'--mode',mode,'--order','sequential','--concept-local','--mean-budget',str(mean_budget),'--suppress','--out','artifacts/'+name],gpu=True)
    run('background_region',['pini_cbm.background_eval','--runs','artifacts/region_mean','artifacts/region_noise','--out','artifacts/background_intervention_region'],gpu=True)
    run('background_controls_region',['pini_cbm.background_controls'])
    run('energy_region',['pini_cbm.energy_audit','--run','artifacts/region_noise'])
    run('localization_region',['pini_cbm.localization_audit','--run','artifacts/region_noise'])
    run('gallery_region',['pini_cbm.visualize','--run','artifacts/region_noise','--out','reports/spatial_noise_region'])
    run('concept_gallery_region',['pini_cbm.concept_gallery','--run','artifacts/region_noise','--out','reports/spatial_noise_region'])
    run('summary_region',['pini_cbm.summarize'])
    print('Region-coupled experiment completed; inspect native precision and controls',flush=True)


if __name__=='__main__':main()
