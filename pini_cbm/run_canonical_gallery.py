"""Audit the literal default FP16 CNN without conflating it with AMP visuals."""
import subprocess
import sys
import time
from pathlib import Path


def main():
    root=Path('artifacts/pets_image_fp16_seed1');started=time.time()
    while not (root/'results.json').exists():
        if time.time()-started>14400:raise SystemExit('Canonical image CNN requires inspection')
        time.sleep(20)
    lock=str(Path('../D-Pi/artifacts/resources/gpu_work.lock').resolve());out='reports/pini_official/fp16'
    for name,module in [('gallery','official_gallery'),('energy','official_energy')]:
        command=['flock',lock,sys.executable,'-u','-m','pini_cbm.'+module,'--run',str(root),'--out',out,'--device','cuda']
        with Path('artifacts/canonical_'+name+'.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise SystemExit(result.returncode)
    result=subprocess.run([sys.executable,'-u','-m','pini_cbm.summarize'])
    if result.returncode:raise SystemExit(result.returncode)
    print('Literal FP16 CNN visualization and energy complete',flush=True)


if __name__=='__main__':main()
