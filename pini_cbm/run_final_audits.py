"""Complete mechanism evaluation and reports after the fixed queued models."""
import json
import subprocess
import sys
import time
from pathlib import Path


V1=['noise_independent','noise_sequential','noise_joint','noise_separate_heads']
LOCAL=['local_noise','full_joint_noise']


def run(name,args,gpu=False):
    command=[sys.executable,'-u','-m']+args
    if gpu:command=['flock',str(Path('../D-Pi/artifacts/resources/gpu_work.lock').resolve())]+command
    print(json.dumps(dict(start=name,command=command)),flush=True)
    with (Path('artifacts')/(name+'.log')).open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise SystemExit(result.returncode)


def main():
    required=V1+LOCAL+['local_mean','full_joint_mean','pets_image_mean_seed1','pets_main_fp16_seed1','pets_image_fp16_seed1']
    started=time.time()
    while True:
        pending=[n for n in required if not (Path('artifacts')/n/'results.json').exists()]
        if not pending:
            mc_pending=[]
            for group in [V1,LOCAL]:
                selected=max((json.loads((Path('artifacts')/n/'results.json').read_text())['concepts']['val']['mAP'],n) for n in group)[1]
                if 'MC8' not in json.loads((Path('artifacts')/selected/'results.json').read_text()):mc_pending.append(selected+' MC8')
            pending=mc_pending
        if not pending:break
        if time.time()-started>21600:raise SystemExit('Model/MC prerequisites require inspection')
        print(json.dumps(dict(pending=pending,elapsed_s=round(time.time()-started))),flush=True);time.sleep(30)
    run('background_local',['pini_cbm.background_eval','--runs','artifacts/local_mean','artifacts/local_noise','artifacts/full_joint_mean','artifacts/full_joint_noise',
                            '--out','artifacts/background_intervention_local'],gpu=True)
    run('background_direct_controls',['pini_cbm.background_controls'])
    run('official_direct_controls',['pini_cbm.official_controls'])
    run('semantic_readout_final',['pini_cbm.semantic_readout'])
    run('concept_audit_final',['pini_cbm.concept_audit'])
    run('uncertainty_final',['pini_cbm.uncertainty'])
    report=Path('reports/spatial_noise_final');report.mkdir(parents=True,exist_ok=True)
    candidates=[dict(run='artifacts/'+n,validation_mAP=json.loads((Path('artifacts')/n/'results.json').read_text())['concepts']['val']['mAP']) for n in V1+LOCAL]
    selection=max(candidates,key=lambda x:x['validation_mAP']);selection['rule']='highest complete native validation mAP among fixed stochastic variants; no test selection'
    selection['candidates']=candidates;(report/'selection.json').write_text(json.dumps(selection,indent=2))
    for name in list(dict.fromkeys(['noise_sequential','local_noise','full_joint_noise',Path(selection['run']).name])):
        run('energy_'+name,['pini_cbm.energy_audit','--run','artifacts/'+name])
        run('localization_'+name,['pini_cbm.localization_audit','--run','artifacts/'+name])
        summary=json.loads((Path('artifacts')/name/'localization_audit.json').read_text());summary.pop('per_pair')
        (report/(name+'_localization.json')).write_text(json.dumps(summary,indent=2))
    run('gallery_final',['pini_cbm.visualize','--run',selection['run'],'--out',str(report)])
    run('concept_gallery_final',['pini_cbm.concept_gallery','--run',selection['run'],'--out',str(report)])
    # Retain local proposal visuals even if the best validation model is V1.
    if Path(selection['run']).name!='local_noise':
        run('concept_gallery_local',['pini_cbm.concept_gallery','--run','artifacts/local_noise','--out',str(report/'local')])
    run('mechanism_figures_final',['pini_cbm.mechanism_figures'])
    run('summary_final',['pini_cbm.summarize'])
    (report/'COMPLETE.json').write_text(json.dumps(dict(selection=selection,all_required=required,all_mc_groups=[V1,LOCAL],
        background_intervention=True,concept_audit=True,paired_uncertainty=True,semantic_shared_head=True,
        note='Computational checks completed; scientific conclusions require reading actual results, not the presence of this marker.'),indent=2))
    print('Final computational audits completed; inspect scientific results before completing goal',flush=True)


if __name__=='__main__':main()
