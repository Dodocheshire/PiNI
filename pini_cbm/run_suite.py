"""Run preregistered matched controls under the shared GPU flock."""
import json
import subprocess
import sys
from pathlib import Path


def main():
    jobs=[
        ('mean_independent','mean','independent',False),
        ('mask_sequential','mask','sequential',False),
        ('mean_sequential','mean','sequential',False),
        ('noise_sequential','noise','sequential',False),
        ('noise_joint','noise','joint',False),
        ('noise_separate_heads','noise','sequential',True),
    ]
    outcomes=[]
    for name,mode,order,independent_heads in jobs:
        out=Path('artifacts')/name
        if (out/'results.json').exists():
            outcomes.append(dict(job=name,status='already complete'));continue
        args=[sys.executable,'-u','-m','pini_cbm.train_noise','--mode',mode,'--order',order,'--suppress','--out',str(out)]
        if independent_heads:args.append('--independent-heads')
        print(json.dumps(dict(start=name,command=args)),flush=True)
        with (Path('artifacts')/(name+'.log')).open('w') as log:
            proc=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
        outcomes.append(dict(job=name,returncode=proc.returncode))
        Path('artifacts/suite_status.json').write_text(json.dumps(outcomes,indent=2))
        print(json.dumps(outcomes[-1]),flush=True)
        if proc.returncode:raise SystemExit(proc.returncode)


if __name__=='__main__':main()
