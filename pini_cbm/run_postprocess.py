"""Finish queued external intervention and audited official visualization."""
import json
import subprocess
import sys
import time
from pathlib import Path


def run(name,args,gpu=False):
    command=[sys.executable,'-u']+args
    if gpu:command=['flock',str(Path('../D-Pi/artifacts/resources/gpu_work.lock').resolve())]+command
    print(json.dumps(dict(stage=name,command=command)),flush=True)
    with Path('artifacts/'+name+'.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise SystemExit(result.returncode)


def main():
    required=['noise_joint','noise_separate_heads','pets_image_seed1']
    started=time.time()
    while not all((Path('artifacts')/n/'results.json').exists() for n in required):
        if time.time()-started>21600:raise SystemExit('Training jobs require inspection')
        time.sleep(30)
    run('background_intervention',['-m','pini_cbm.background_eval'],gpu=True)
    run('official_gallery',['-m','pini_cbm.official_gallery'],gpu=True)
    run('concept_audit',['-m','pini_cbm.concept_audit'])
    run('summary',['-m','pini_cbm.summarize'])
    print('First-stage reports and intervention completed',flush=True)


if __name__=='__main__':main()
