"""Close the capacity-matched deterministic control for concept-specific heads."""
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    # Finish the already queued regional experiment before adding one control.
    started=time.time()
    while not (Path('artifacts/region_noise/results.json')).exists():
        if time.time()-started>21600:raise SystemExit('Inspect prerequisite jobs')
        time.sleep(30)
    if not Path('artifacts/mean_separate_heads/results.json').exists():
        command=['flock',str(Path('../D-Pi/artifacts/resources/gpu_work.lock').resolve()),
                 sys.executable,'-u','-m','pini_cbm.train_noise','--mode','mean',
                 '--order','sequential','--independent-heads','--suppress','--out','artifacts/mean_separate_heads']
        print(json.dumps(dict(command=command)),flush=True)
        with Path('artifacts/mean_separate_heads.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,'-u','-m','pini_cbm.summarize'],check=True)


if __name__=='__main__':main()
