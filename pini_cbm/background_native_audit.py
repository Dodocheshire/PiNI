"""Paired native-concept AP under the fixed external background intervention."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .train_spatial import load_cache
from .uncertainty import bootstrap_concepts


def main():
    arrays,_,_,_=load_cache('artifacts/rival_vitl14')
    destination=Path('reports/spatial_noise_v1/background_native_uncertainty.json')
    prior=json.loads(destination.read_text()) if destination.exists() else {'pairs':{}}
    result=dict(repetitions=1000,pairs={},
        interpretation='Paired 1000-image external background intervention, same IDs and donors. All 18 native concepts; test-sample intervals do not measure seed variability. Comparisons are exploratory.')
    for folder,a,b,baseline in [
        ('background_intervention','noise_sequential','mean_sequential',None),
        ('background_intervention_local','local_noise','local_mean','local_clean'),
        ('background_intervention_local','full_joint_noise','full_joint_mean','local_clean'),
        ('background_intervention_region','region_noise','region_mean','region_clean'),
        ('background_intervention_fixed','fixed_noise','fixed_mean','region_clean'),
        ('background_intervention_region_seed43','region_seed43_noise','region_seed43_mean','region_clean')]:
        run=Path('artifacts')/folder
        if not (run/'results.json').exists():continue
        protocol=json.loads((run/'protocol.json').read_text())
        y=np.array(arrays['attrs'][protocol['ids']],dtype='float32')
        x=np.load(run/(a+'_background_swap_probs.npy'))
        comparators=[(a,x,b,run/(b+'_background_swap_probs.npy')),
                     (a,x,a+'_same_checkpoint_mean',run/(a+'_same_checkpoint_mean_background_swap_probs.npy')),
                     (a+'_same_checkpoint_mean',np.load(run/(a+'_same_checkpoint_mean_background_swap_probs.npy')),b,run/(b+'_background_swap_probs.npy'))]
        if baseline:
            clean=Path('artifacts/background_clean_models')
            cfg=json.loads((clean/'results.json').read_text())
            assert protocol['ids']==cfg['reference_ids'] and protocol['donors']==cfg['reference_donors']
            comparators.append((a,x,baseline,clean/(baseline+'_background_swap_probs.npy')))
        for source,values,name,path in comparators:
            z=np.load(path);key=source+'_minus_'+name
            fingerprint=hashlib.sha256(y.tobytes()+values.tobytes()+z.tobytes()+Path(__file__).read_bytes()).hexdigest()
            old=prior['pairs'].get(key,{})
            if old.get('input_sha256')==fingerprint:row=old
            else:
                row=bootstrap_concepts(y,values,z,1000,4819)
                row.update(input_sha256=fingerprint,n=len(y),condition='background_swap',
                           intervention_folder=str(run),ids_sha256=hashlib.sha256(np.array(protocol['ids']).tobytes()).hexdigest())
            result['pairs'][key]=row
            print(key,row['mAP_difference_pp'],row['mAP_CI95_pp'],flush=True)
    destination.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
