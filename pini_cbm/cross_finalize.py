"""Generate and verify the paper when the live experiment queue finishes."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .cross_data import ROOT, write_json


def main():
    state_path = ROOT/"FINALIZE_STATE.json"
    state = dict(pid=os.getpid(),status="waiting_for_experiments",started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    write_json(state_path,state)
    progress_path = Path("reports/main/cross_progress.json")
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"datasets":{}}
    published_rounds = sum(len(row["seeds"]) for row in progress["datasets"].values())
    published_models = len(progress.get("original_models",[]))+sum(
        len(row["models"].get(method,[])) for row in progress["datasets"].values() for method in ["mean","noise"])
    published_stats = sum(len(row["spatial"]) for row in progress.get("spatial_datasets",progress["datasets"]).values())
    while not (ROOT/"ALL_COMPLETE.json").exists():
        run = json.loads((ROOT/"RUN_STATE.json").read_text())
        runner = Path("/proc")/str(run["pid"])
        live = runner.exists() and b"pini_cbm.cross_run" in (runner/"cmdline").read_bytes()
        if run["status"]=="failed" or not live:
            state.update(status="runner_stopped",runner_state=run)
            write_json(state_path,state)
            raise SystemExit("Experiment runner stopped before completion; inspect and resume the recorded step.")
        finished_rounds = 0
        for name in ["cub","cifar10","cifar100"]:
            methods = ["lf_cbm","supervised_cbm","mean","noise"] if name=="cub" else ["lf_cbm","mean","noise"]
            for seed in [42,43,44]:
                paths = [ROOT/name/f"{method}_seed{seed}/results.json" for method in methods]
                if ((ROOT/name/f"noise_seed{seed}/spatial_statistics.json").exists()
                        and all(p.exists() and "background" in json.loads(p.read_text()) for p in paths)):
                    finished_rounds += 1
        for trained in sorted(ROOT.glob("*/noise_seed*/TRAINED.json")):
            if (trained.parent/"spatial_statistics.json").exists():
                continue
            name = trained.parent.parent.name
            seed = int(trained.parent.name.removeprefix("noise_seed"))
            state.update(status="preparing_spatial_statistics",spatial_dataset=name,spatial_seed=seed)
            write_json(state_path,state)
            command = [sys.executable,"-u","-m","pini_cbm.cross_experiment","--dataset",name,
                       "--phase","spatial","--seed",str(seed),"--device","cpu"]
            result = subprocess.run(command)
            if result.returncode:
                state.update(status="failed",returncode=result.returncode)
                write_json(state_path,state)
                raise SystemExit(result.returncode)
        finished_stats = len(list(ROOT.glob("*/noise_seed*/spatial_statistics.json")))
        finished_models = sum(path.parent.name.startswith(("mean_seed","noise_seed"))
                              for path in ROOT.glob("*/*/original_results.json"))
        if finished_rounds>published_rounds or finished_models>published_models or finished_stats>published_stats:
            for arguments in [["pini_cbm.cross_reporting","--baseline-progress"],["pini_cbm.verify_paper_report"]]:
                state.update(status="publishing_completed_rounds",current_module=arguments[0])
                write_json(state_path,state)
                result = subprocess.run([sys.executable,"-u","-m",*arguments])
                if result.returncode:
                    state.update(status="failed",returncode=result.returncode)
                    write_json(state_path,state)
                    raise SystemExit(result.returncode)
            published_rounds = finished_rounds
            published_models = finished_models
            published_stats = finished_stats
        state.update(runner_pid=run["pid"],current_step=run["current_step"],
                     status="waiting_for_experiments",current_module=None,published_rounds=published_rounds,
                     published_spatial_models=published_models,
                     published_spatial_statistics=published_stats,
                     completed_steps=len(run["completed_steps"]),verified_runner_live=True,
                     updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
        write_json(state_path,state)
        time.sleep(30)
    for module in ["pini_cbm.summarize","pini_cbm.verify_paper_report","pini_cbm.cross_verification"]:
        state.update(status="generating_and_verifying",current_module=module)
        write_json(state_path,state)
        command = [sys.executable,"-u","-m",module]
        result = subprocess.run(command)
        if result.returncode:
            state.update(status="failed",current_module=module,returncode=result.returncode)
            write_json(state_path,state)
            raise SystemExit(result.returncode)
    state.update(status="report_ready_for_visual_review",current_module=None,
                 completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    write_json(state_path,state)
    print("All dataset experiments, figures and numerical verification are ready for visual review.",flush=True)


if __name__=="__main__":
    main()
