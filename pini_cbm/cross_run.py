"""Resumable three-dataset, three-seed experiment queue (one GPU lease)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .cross_data import ROOT, write_json


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    state_path = ROOT/"RUN_STATE.json"
    old = json.loads(state_path.read_text()) if state_path.exists() else {}
    completed = old.get("completed_steps",[])
    state = dict(pid=os.getpid(),started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                 status="running",completed_steps=completed,current_step=None)
    steps = []
    names = ["cub","cifar10","cifar100"]
    for name in names:
        steps.append((name+"_cache",["cross_data","--dataset",name,"--stage","all"]))
    for name in names:
        steps.append((name+"_region",["cross_experiment","--dataset",name,"--phase","pretrain"]))
    for name in names:
        for method in (["lf_cbm","supervised_cbm"] if name=="cub" else ["lf_cbm"]):
            for seed in [42,43,44]:
                steps.append((f"{name}_{method}_{seed}",["cross_experiment","--dataset",name,"--phase","baseline",
                              "--method",method,"--seed",str(seed)]))
        steps.append((name+"_intervention",["cross_experiment","--dataset",name,"--phase","intervention"]))
    for seed in [42,43,44]:
        for name in names:
            for method in ["mean","noise"]:
                steps.append((f"{name}_{method}_{seed}",["cross_experiment","--dataset",name,"--phase","adapt",
                              "--method",method,"--seed",str(seed)]))
            for method in (["lf_cbm","supervised_cbm","mean","noise"] if name=="cub" else ["lf_cbm","mean","noise"]):
                steps.append((f"{name}_{method}_{seed}_background",["cross_experiment","--dataset",name,
                              "--phase","background","--method",method,"--seed",str(seed)]))
            steps.append((f"{name}_{seed}_spatial",["cross_experiment","--dataset",name,
                          "--phase","spatial","--seed",str(seed)]))
    write_json(ROOT/"QUEUE.json",dict(total_steps=len(steps),steps=[dict(name=n,arguments=a) for n,a in steps]))
    write_json(state_path,state)
    for number,(step,arguments) in enumerate(steps):
        if step in completed:
            continue
        command = [sys.executable,"-u","-m","pini_cbm."+arguments[0],*arguments[1:]]
        state.update(current_step=step,current_index=number,total_steps=len(steps),status="running",
                     updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
        write_json(state_path,state)
        print(json.dumps(dict(step=step,event="start",index=number,total_steps=len(steps))),flush=True)
        log_path = ROOT/(step+".log")
        with log_path.open("a") as log:
            child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            state["child_pid"] = child.pid
            state["current_log"] = str(log_path)
            write_json(state_path,state)
            code = child.wait()
        if code:
            state.update(status="failed",returncode=code)
            write_json(state_path,state)
            print(json.dumps(dict(step=step,event="failed",code=code,log=str(log_path))),flush=True)
            raise SystemExit(code)
        completed.append(step)
        state["completed_steps"] = completed
        write_json(state_path,state)
        print(json.dumps(dict(step=step,event="complete")),flush=True)
    state.update(status="experiments_complete",current_step=None,child_pid=None)
    write_json(state_path,state)
    write_json(ROOT/"ALL_COMPLETE.json",dict(datasets=names,seeds=[42,43,44],steps=completed))
    print("ALL THREE DATASET EXPERIMENTS COMPLETE",flush=True)


if __name__=="__main__":
    main()
