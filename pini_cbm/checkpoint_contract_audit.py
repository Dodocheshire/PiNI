"""Actual cached-input compatibility and separate frozen-blocker contracts."""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from .models import ConditionalRGBNoise, SpatialConcepts
from .train_spatial import load_cache


def main():
    torch.set_num_threads(2)
    cache=Path('artifacts/rival_vitl14')
    arrays,splits,_,_=load_cache(cache)
    ids=splits['val'][:8]
    tokens=torch.tensor(np.array(arrays['tokens'][ids]),dtype=torch.float32)
    texts=torch.tensor(np.load(cache/'text.npy'))
    concept_ids=torch.arange(8)
    revision='a2b70a052c9263d435107275a803fdfcc71e228d'
    old_source=subprocess.check_output(['git','show',revision+':pini_cbm/models.py'],text=True)
    namespace={};exec(compile(old_source,'historical_models.py','exec'),namespace)
    ck=torch.load('artifacts/noise_sequential/best.pt',map_location='cpu',weights_only=False)
    cfg=ck['args']
    old=namespace['ConditionalRGBNoise'](namespace['SpatialConcepts'](texts),cfg['independent_heads'],cfg.get('concept_local',False))
    new=ConditionalRGBNoise(SpatialConcepts(texts),cfg['independent_heads'],cfg.get('concept_local',False))
    old.load_state_dict(ck['model']);new.load_state_dict(ck['model'])
    with torch.no_grad():
        fields=[float((a-b).abs().max()) for a,b in zip(old.distribution(tokens,concept_ids),new.distribution(tokens,concept_ids))]
        logits=float((old.concepts(tokens)[0]-new.recognize(tokens,tokens)[0]).abs().max())
    assert fields==[0.,0.,0.] and logits==0.
    source=torch.load('artifacts/spatial_region/best.pt',map_location='cpu',weights_only=False)
    concepts=SpatialConcepts(texts);concepts.load_state_dict(source['model'])
    budget=.02686347473647816
    fixed=ConditionalRGBNoise(concepts,independent=True,concept_local=True,mean_budget=budget,sigma_budget=budget,fixed_foreground=True)
    source_weight_delta=max(float((value-source['model'][key]).abs().max()) for key,value in fixed.blocker.state_dict().items())
    with torch.no_grad():
        before=fixed.blocker(tokens).clone()
        source_delta=float((before-concepts.spatial(tokens)[-1].sigmoid().reshape(-1,16,16)).abs().max())
        # In-memory counterfactual only; no trained checkpoint is changed.
        concepts.patch.weight.add_(.1)
        independent_delta=float((before-fixed.blocker(tokens)).abs().max())
        reference=fixed.distribution(tokens,concept_ids)
        restored=ConditionalRGBNoise(SpatialConcepts(texts),independent=True,concept_local=True)
        restored.load_state_dict(fixed.state_dict())
        restore_delta=[float((a-b).abs().max()) for a,b in zip(reference,restored.distribution(tokens,concept_ids))]
        rms=reference[1].square().mean((1,2,3)).sqrt()
    assert source_weight_delta==0. and source_delta<1e-6 and independent_delta==0.,(source_weight_delta,source_delta,independent_delta)
    # Existing scalar budget buffers use FP32. Python-to-buffer rounding is
    # allowed at numerical precision; spatial fields must otherwise agree.
    assert max(restore_delta)<1e-7,restore_delta
    assert restored.fixed_foreground and restored.concepts.region_readout
    assert all(not p.requires_grad for p in restored.blocker.parameters())
    assert torch.allclose(rms,torch.full_like(rms,budget),atol=1e-7,rtol=0.)
    result=dict(historical_revision=revision,validation_ids=ids.tolist(),
        historical_field_max_deltas=dict(zip(['mu','sigma','gate'],fields)),historical_native_logit_max_delta=logits,
        copied_foreground_weight_max_delta=source_weight_delta,copied_foreground_probability_max_delta=source_delta,
        source_prediction_tolerance=1e-6,foreground_change_after_recognizer_weight_mutation=independent_delta,
        fixed_checkpoint_roundtrip_field_max_deltas=restore_delta,fixed_roundtrip_tolerance=1e-7,fixed_flag_restored=restored.fixed_foreground,
        region_readout_flag_restored=restored.concepts.region_readout,sigma_RGB_RMS=rms.tolist(),sigma_budget=budget,
        model_source_sha256=hashlib.sha256(Path('pini_cbm/models.py').read_bytes()).hexdigest(),
        interpretation='CPU contract audit on actual validation tokens. Temporary in-memory recognizer mutation only; no trained performance claim.')
    out=Path('reports/spatial_noise_v1/fixed_blocker_contract.json');out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
