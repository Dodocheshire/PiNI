"""Direct paired Gaussian versus mean contrasts from existing interventions."""
import json
from pathlib import Path
import numpy as np
from .uncertainty import ci_correct


def main():
    for folder in ['background_intervention','background_intervention_local']:
        out=Path('artifacts')/folder;p=out/'results.json'
        if not p.exists():continue
        data=json.loads(p.read_text());contrasts={}
        for a,b in [('noise_sequential','mean_sequential'),('local_noise','local_mean'),('full_joint_noise','full_joint_mean')]:
            if a not in data['models']:continue
            for comparator in [b,a+'_same_checkpoint_mean']:
                if comparator not in data['models']:continue
                for condition in ['original','background_swap']:
                    x=np.load(out/(a+'_'+condition+'_correct.npy'));y=np.load(out/(comparator+'_'+condition+'_correct.npy'))
                    contrasts[a+'_minus_'+comparator+'_'+condition]=ci_correct(x,y)
        data['direct_noise_controls']=contrasts;p.write_text(json.dumps(data,indent=2))
        print(folder,contrasts,flush=True)


if __name__=='__main__':main()
