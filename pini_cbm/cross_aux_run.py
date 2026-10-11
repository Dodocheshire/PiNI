"""Advance independent repetitions alongside the main dataset queue."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .cross_data import ROOT, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid",type=int,default=0)
    parser.add_argument("--datasets",nargs="+",choices=["cub","cifar10","cifar100"],
                        default=["cub","cifar10","cifar100"])
    args = parser.parse_args()
    state_path = ROOT/"AUXILIARY_CUB_STATE.json"
    state = dict(pid=os.getpid(),status="waiting_for_prior_worker",wait_pid=args.wait_pid,
                 datasets=args.datasets,cifar_seeds=[44],
                 completed_steps=[],started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    write_json(state_path,state)
    waiting = Path("/proc")/str(args.wait_pid)/"cmdline"
    while waiting.exists():
        try:
            command = waiting.read_bytes()
        except FileNotFoundError:
            break
        if b"pini_cbm.cross_experiment" not in command:
            break
        time.sleep(5)
    if "cub" in args.datasets and not (ROOT/"cub/mean_seed43/original_results.json").exists():
        state.update(status="prior_worker_stopped_before_result")
        write_json(state_path,state)
        raise SystemExit("Inspect CUB mean seed 43 before resuming the auxiliary queue.")
    steps = [("background","mean",43),("adapt","noise",43),("background","noise",43),
             ("adapt","mean",44),("background","mean",44),
             ("adapt","noise",44),("background","noise",44)]
    steps = [("cub",*step) for step in steps]
    for dataset in ["cifar10","cifar100"]:
        for method in ["mean","noise"]:
            for phase in ["adapt","background"]:
                steps.append((dataset,phase,method,44))
    steps = [step for step in steps if step[0] in args.datasets]
    state["scheduled_steps"] = [f"{dataset}_{method}_{seed}_{phase}" for dataset,phase,method,seed in steps]
    for dataset,phase,method,seed in steps:
        step = f"{dataset}_{method}_{seed}_{phase}"
        command = [sys.executable,"-u","-m","pini_cbm.cross_experiment","--dataset",dataset,
                   "--phase",phase,"--method",method,"--seed",str(seed),"--eval-batch","256"]
        state.update(status="running",current_step=step,
                     updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
        with (ROOT/(step+"_aux.log")).open("a") as log:
            child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            state["child_pid"] = child.pid
            write_json(state_path,state)
            code = child.wait()
        if code:
            state.update(status="failed",returncode=code)
            write_json(state_path,state)
            raise SystemExit(code)
        state["completed_steps"].append(step)
        write_json(state_path,state)
        print(json.dumps(dict(step=step,status="complete")),flush=True)
    state.update(status="complete",current_step=None,child_pid=None,
                 completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    write_json(state_path,state)


if __name__=="__main__":
    main()
