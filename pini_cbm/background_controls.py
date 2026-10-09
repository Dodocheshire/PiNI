"""Direct paired Gaussian versus mean contrasts from existing interventions."""
import json
from pathlib import Path
import numpy as np
from .uncertainty import ci_correct


def main():
    extra_path=Path('artifacts/background_clean_models/results.json')
    extra=json.loads(extra_path.read_text()) if extra_path.exists() else None
    for folder in ['background_intervention','background_intervention_local','background_intervention_region','background_intervention_fixed','background_intervention_region_seed43']:
        out=Path('artifacts')/folder;p=out/'results.json'
        if not p.exists():continue
        data=json.loads(p.read_text());contrasts={}
        for a,b in [('noise_sequential','mean_sequential'),('local_noise','local_mean'),('full_joint_noise','full_joint_mean'),('region_noise','region_mean'),('fixed_noise','fixed_mean'),('region_seed43_noise','region_seed43_mean')]:
            if a not in data['models']:continue
            for comparator in [b,a+'_same_checkpoint_mean']:
                if comparator not in data['models']:continue
                for condition in ['original','background_swap']:
                    x=np.load(out/(a+'_'+condition+'_correct.npy'));y=np.load(out/(comparator+'_'+condition+'_correct.npy'))
                    contrasts[a+'_minus_'+comparator+'_'+condition]=ci_correct(x,y)
            if b in data['models'] and a+'_same_checkpoint_mean' in data['models']:
                for condition in ['original','background_swap']:
                    x=np.load(out/(a+'_same_checkpoint_mean_'+condition+'_correct.npy'));y=np.load(out/(b+'_'+condition+'_correct.npy'))
                    contrasts[a+'_same_checkpoint_mean_minus_'+b+'_'+condition]=ci_correct(x,y)
        data['direct_noise_controls']=contrasts;p.write_text(json.dumps(data,indent=2))
        base='local_clean' if folder=='background_intervention_local' else 'region_clean' if folder in ['background_intervention_region','background_intervention_fixed','background_intervention_region_seed43'] else None
        if extra and base:
            protocol=json.loads((out/'protocol.json').read_text())
            assert protocol['ids']==extra['reference_ids'] and protocol['donors']==extra['reference_donors']
            matched={}
            for name in data['models']:
                if name=='clean':continue
                correct=np.load(out/(name+'_background_swap_correct.npy'))
                comparator=np.load(extra_path.parent/(base+'_background_swap_correct.npy'))
                matched[name+'_minus_'+base]=ci_correct(correct,comparator)
            data['matched_clean_controls']=matched;p.write_text(json.dumps(data,indent=2))
        print(folder,contrasts,flush=True)


if __name__=='__main__':main()
