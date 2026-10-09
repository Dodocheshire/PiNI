"""Two-stage fixed blocker and per-concept Gaussians, matched mean/noise pair."""
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from .data import STD


def run(name,module,args,gpu=False):
    command=[sys.executable,'-u','-m',module]+args
    if gpu:command=['flock',str(Path('../D-Pi/artifacts/resources/gpu_work.lock').resolve())]+command
    print(name,command,flush=True)
    with (Path('artifacts')/(name+'.log')).open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    started=time.time()
    while not all((Path('artifacts')/name/'results.json').exists() for name in ['region_noise','mean_separate_heads']):
        if time.time()-started>21600:raise SystemExit('Inspect prerequisite processes')
        time.sleep(30)
    budget=.1*float(np.sqrt(np.mean(np.square(STD))))
    for mode in ['mean','noise']:
        name='fixed_'+mode
        if (Path('artifacts')/name/'results.json').exists():continue
        run(name,'pini_cbm.train_noise',['--clean','artifacts/spatial_region','--mode',mode,'--order','sequential',
            '--concept-local','--independent-heads','--fixed-foreground','--mean-budget',str(budget),
            '--sigma-budget',str(budget),'--suppress','--out','artifacts/'+name],gpu=True)
    run('background_fixed','pini_cbm.background_eval',['--runs','artifacts/fixed_mean','artifacts/fixed_noise','--out','artifacts/background_intervention_fixed'],gpu=True)
    run('foreground_fixed','pini_cbm.foreground_drift_audit',[])
    run('energy_fixed','pini_cbm.energy_audit',['--run','artifacts/fixed_noise'])
    run('localization_fixed','pini_cbm.localization_audit',['--run','artifacts/fixed_noise'])
    run('gallery_fixed','pini_cbm.visualize',['--run','artifacts/fixed_noise','--out','reports/spatial_noise_fixed'])
    run('concept_gallery_fixed','pini_cbm.concept_gallery',['--run','artifacts/fixed_noise','--out','reports/spatial_noise_fixed'])
    run('noise_region_fixed','pini_cbm.noise_region_intervention',['--runs','artifacts/fixed_noise'],gpu=True)
    run('summary_fixed','pini_cbm.summarize',[])


if __name__=='__main__':main()
