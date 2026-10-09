"""Complete local sequential pair first, then matched full joint RGB pair."""
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    started=time.time();required=['spatial_local','local_mean','local_noise']
    while not all((Path('artifacts')/name/'results.json').exists() for name in required):
        if time.time()-started>21600:raise SystemExit('Local experiments need inspection')
        time.sleep(30)
    lock=str(Path('../D-Pi/artifacts/resources/gpu_work.lock').resolve())
    for mode in ['mean','noise']:
        name='full_joint_'+mode;out=Path('artifacts')/name
        if (out/'results.json').exists():continue
        command=['flock',lock,sys.executable,'-u','-m','pini_cbm.train_full_joint','--mode',mode,'--out',str(out)]
        print(json.dumps(dict(start=name,command=command)),flush=True)
        with (Path('artifacts')/(name+'.log')).open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise SystemExit(result.returncode)
    print('Full joint matched pair complete',flush=True)


if __name__=='__main__':main()
