"""Run the necessary foreground/background noise influence audit after training."""
import subprocess
import sys
import time
from pathlib import Path


def main():
    started=time.time()
    while not Path('artifacts/region_noise/results.json').exists():
        if time.time()-started>21600:raise SystemExit('Inspect region training prerequisites')
        time.sleep(30)
    command=['flock',str(Path('../D-Pi/artifacts/resources/gpu_work.lock').resolve()),
             sys.executable,'-u','-m','pini_cbm.noise_region_intervention',
             '--runs','artifacts/noise_sequential','artifacts/region_noise']
    with Path('artifacts/noise_region_intervention.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,'-u','-m','pini_cbm.summarize'],check=True)


if __name__=='__main__':main()
