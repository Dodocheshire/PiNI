"""Evaluate MC stability on validation-selected models, with per-job GPU lock."""
import json
import subprocess
import sys
import time
from pathlib import Path


def wait(names):
    started=time.time()
    while not all((Path('artifacts')/n/'results.json').exists() for n in names):
        if time.time()-started>21600:raise SystemExit('Prerequisite models require inspection')
        time.sleep(30)


def main():
    groups=[['noise_independent','noise_sequential','noise_joint','noise_separate_heads'],['local_noise','full_joint_noise']]
    for names in groups:
        wait(names)
        rows=[(json.loads((Path('artifacts')/n/'results.json').read_text())['concepts']['val']['mAP'],n) for n in names]
        selected=max(rows)[1];run=Path('artifacts')/selected;result=json.loads((run/'results.json').read_text())
        print(json.dumps(dict(selection_group=names,validation_mAP=rows,selected=selected)),flush=True)
        if 'MC8' in result:continue
        command=['flock',str(Path('../D-Pi/artifacts/resources/gpu_work.lock').resolve()),sys.executable,'-u','-m','pini_cbm.mc_eval','--run',str(run),
                 '--selection-description','highest native validation mAP among '+','.join(names)]
        with (Path('artifacts')/(selected+'_MC8.log')).open('w') as log:proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:raise SystemExit(proc.returncode)
    print('Both validation-selected MC8 evaluations complete',flush=True)


if __name__=='__main__':main()
