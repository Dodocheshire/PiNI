"""Wait for the existing OxfordPets download, then run both official variants."""
import json
import subprocess
import sys
import time
from pathlib import Path


def ready(root):
    for split in ['trainval','test']:
        annotation=root/'annotations'/(split+'.txt')
        if not annotation.exists():return False
        rows=[x.split()[0] for x in annotation.read_text().splitlines() if x and not x.startswith('#')]
        if not rows or not all((root/'images'/(s+'.jpg')).exists() for s in rows):return False
    return True


def main():
    root=Path('artifacts/data/oxford_pets/oxford-iiit-pet')
    lock=Path('../D-Pi/artifacts/resources/gpu_work.lock').resolve()
    started=time.time()
    while not ready(root):
        print(json.dumps(dict(stage='waiting_existing_download',elapsed_s=round(time.time()-started))),flush=True)
        if time.time()-started>7200:raise SystemExit('Dataset download not ready after 2 hours')
        time.sleep(30)
    for kind in ['main','image']:
        out=Path('artifacts/pets_'+kind+'_seed1')
        if (out/'results.json').exists():continue
        args=['flock',str(lock),sys.executable,'-u','-m','pini_cbm.reproduce_pets','--kind',kind,'--out',str(out)]
        print(json.dumps(dict(stage='official_reproduction',kind=kind,command=args)),flush=True)
        with Path('artifacts/pets_'+kind+'.log').open('w') as f:
            result=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise SystemExit(result.returncode)
        print(json.dumps(dict(stage='completed',kind=kind)),flush=True)


if __name__=='__main__':main()
