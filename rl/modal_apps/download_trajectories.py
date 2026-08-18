"""Download all raw trajectory.json files from Modal volumes to local workspace."""

import modal
import json
from pathlib import Path

app = modal.App("download-all-raw-trajectories")
image = modal.Image.debian_slim(python_version="3.12")
volume_outputs = modal.Volume.from_name("fable5-terminalbench-outputs-h200")

@app.function(image=image, volumes={"/outputs": volume_outputs})
def fetch_all_trajectories():
    trajectories = {}
    output_base = Path("/outputs")

    # 1. Search for all job directories in /outputs
    for run_dir in output_base.glob("tb2_*"):
        if not run_dir.is_dir():
            continue
        harbor_dirs = list(run_dir.rglob("harbor_output"))
        for h_dir in harbor_dirs:
            for job in h_dir.iterdir():
                if job.is_dir():
                    for task_dir in job.iterdir():
                        if task_dir.is_dir():
                            traj_file = task_dir / "agent" / "trajectory.json"
                            res_file = task_dir / "result.json"
                            if traj_file.exists():
                                key = f"{run_dir.name}/{task_dir.name}"
                                res_data = {}
                                if res_file.exists():
                                    try:
                                        res_data = json.loads(res_file.read_text(encoding="utf-8"))
                                    except Exception:
                                        pass
                                try:
                                    traj_data = json.loads(traj_file.read_text(encoding="utf-8"))
                                    trajectories[key] = {
                                        "task_name": task_dir.name,
                                        "run_name": run_dir.name,
                                        "result": res_data,
                                        "trajectory": traj_data,
                                    }
                                except Exception as e:
                                    print(f"Error loading {traj_file}: {e}")

    print(f"Successfully loaded {len(trajectories)} raw trajectory datasets.")
    return trajectories

@app.local_entrypoint()
def main():
    local_out = Path("D:/fable5_qwen3.7/eval_trajectories")
    local_out.mkdir(parents=True, exist_ok=True)
    
    print("[*] Fetching raw trajectories from Modal Volume...")
    data = fetch_all_trajectories.remote()
    
    for key, item in data.items():
        task_name = item["task_name"]
        run_name = item["run_name"]
        target_dir = local_out / run_name / task_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        (target_dir / "trajectory.json").write_text(json.dumps(item["trajectory"], indent=2), encoding="utf-8")
        if item["result"]:
            (target_dir / "result.json").write_text(json.dumps(item["result"], indent=2), encoding="utf-8")
        print(f"  ✓ Saved: {target_dir / 'trajectory.json'}")
        
    print(f"\n[+] All {len(data)} raw trajectories downloaded locally to {local_out.resolve()}")
