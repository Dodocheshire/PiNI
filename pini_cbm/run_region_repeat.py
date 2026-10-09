"""One fixed-protocol second adaptation seed; no hyperparameter search."""
import json
import subprocess
import sys
import time
from pathlib import Path

from .run_fixed_foreground import run


def main():
    started=time.time()
    while not Path('reports/spatial_noise_v1/fixed_noise_noise_region_intervention.json').exists():
        if time.time()-started>21600:raise SystemExit('Inspect fixed-stage prerequisites')
        time.sleep(30)
    budget=.02686347473647816
    for mode in ['mean','noise']:
        name='region_seed43_'+mode
        if (Path('artifacts')/name/'results.json').exists():continue
        run(name,'pini_cbm.train_noise',['--clean','artifacts/spatial_region','--mode',mode,'--order','sequential',
            '--concept-local','--mean-budget',str(budget),'--seed','43','--suppress','--out','artifacts/'+name],gpu=True)
    run('background_region_seed43','pini_cbm.background_eval',['--runs','artifacts/region_seed43_mean','artifacts/region_seed43_noise','--out','artifacts/background_intervention_region_seed43'],gpu=True)
    run('summary_region_seed43','pini_cbm.summarize',[])
    Path('artifacts/REGION_REPEAT_COMPLETE.json').write_text(json.dumps(dict(
        adaptation_seeds=[42,43],clean_initialization_shared=True,hyperparameters_changed=False,
        interpretation='Second adaptation seed, same split and pretrained concept/region model; not an independent end-to-end seed replication.'),indent=2))


if __name__=='__main__':main()
