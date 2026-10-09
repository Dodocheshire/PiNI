"""Wait for V1 and official reproductions, then run fixed concept-local pair."""
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    required=['mean_independent','mask_sequential','mean_sequential','noise_sequential','noise_joint','noise_separate_heads',
              'pets_main_seed1','pets_image_seed1']
    started=time.time()
    while not all((Path('artifacts')/name/'results.json').exists() for name in required):
        pending=[name for name in required if not (Path('artifacts')/name/'results.json').exists()]
        print(json.dumps(dict(stage='wait_first_stage',pending=pending,elapsed_s=round(time.time()-started))),flush=True)
        if time.time()-started>21600:raise SystemExit('First-stage jobs require inspection; stopping wait after 6 hours')
        time.sleep(30)
    lock=str(Path('../D-Pi/artifacts/resources/gpu_work.lock').resolve())
    jobs=[
        ('spatial_local',['-m','pini_cbm.train_spatial','--locality-loss','--out','artifacts/spatial_local']),
        ('local_mean',['-m','pini_cbm.train_noise','--clean','artifacts/spatial_local','--mode','mean','--order','sequential','--concept-local','--suppress','--out','artifacts/local_mean']),
        ('local_noise',['-m','pini_cbm.train_noise','--clean','artifacts/spatial_local','--mode','noise','--order','sequential','--concept-local','--suppress','--out','artifacts/local_noise']),
    ]
    for name,args in jobs:
        if (Path('artifacts')/name/'results.json').exists():continue
        command=['flock',lock,sys.executable,'-u']+args
        print(json.dumps(dict(start=name,command=command)),flush=True)
        with Path('artifacts/'+name+'.log').open('w') as f:proc=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
        if proc.returncode:raise SystemExit(proc.returncode)
    proc=subprocess.run([sys.executable,'-m','pini_cbm.audit_clean','--clean','artifacts/spatial_local'])
    if proc.returncode:raise SystemExit(proc.returncode)
    print('Second-stage matched pair completed',flush=True)


if __name__=='__main__':main()
