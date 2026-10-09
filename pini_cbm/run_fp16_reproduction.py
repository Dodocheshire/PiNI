"""Check original default fp16 and Dassl default test batch 32 explicitly."""
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    started=time.time()
    while not Path('artifacts/pets_image_seed1/results.json').exists():
        if time.time()-started>14400:raise SystemExit('Primary image CNN requires inspection')
        time.sleep(15)
    lock=str(Path('../D-Pi/artifacts/resources/gpu_work.lock').resolve())
    for kind in ['main','image']:
        out=Path('artifacts/pets_'+kind+'_fp16_seed1')
        if (out/'results.json').exists():continue
        args=['flock',lock,sys.executable,'-u','-m','pini_cbm.reproduce_pets','--kind',kind,'--precision','fp16','--eval-batch','32','--out',str(out)]
        print(json.dumps(dict(start=kind+'_fp16',command=args)),flush=True)
        with Path('artifacts/pets_'+kind+'_fp16.log').open('w') as log:result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise SystemExit(result.returncode)
    print('Original fp16/default test batch reproductions completed',flush=True)


if __name__=='__main__':main()
