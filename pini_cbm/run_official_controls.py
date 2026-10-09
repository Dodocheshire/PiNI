"""Matched μ-only controls after completing the two primary reproductions."""
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    start=time.time()
    while not (Path('artifacts/pets_image_seed1')/'results.json').exists():
        if time.time()-start>21600:raise SystemExit('Primary official reproduction needs inspection')
        time.sleep(30)
    lock=str(Path('../D-Pi/artifacts/resources/gpu_work.lock').resolve())
    for kind in ['main','image']:
        out=Path('artifacts/pets_'+kind+'_mean_seed1')
        if (out/'results.json').exists():continue
        command=['flock',lock,sys.executable,'-u','-m','pini_cbm.reproduce_pets','--kind',kind,'--noise-mode','mean','--out',str(out)]
        print(json.dumps(dict(start=kind+'_mean',command=command)),flush=True)
        with Path('artifacts/pets_'+kind+'_mean.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise SystemExit(result.returncode)
    print('Official matched mean controls completed',flush=True)


if __name__=='__main__':main()
