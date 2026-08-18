"""Extract all task outputs from kars07 volume."""

import modal
import json
from pathlib import Path

app = modal.App("extract-kars07-base-results")
image = modal.Image.debian_slim(python_version="3.12")
volume_outputs = modal.Volume.from_name("fable5-terminalbench-outputs-h200")

@app.function(image=image, volumes={"/outputs": volume_outputs})
def extract_all():
    output_base = Path("/outputs")
    print("=== DIRECTORIES IN /outputs (kars07) ===")
    for d in sorted(output_base.iterdir()):
        print(f"Directory: {d.name}")
        if d.is_dir():
            for sub in sorted(d.iterdir()):
                print(f"  └── {sub.name}")
                if sub.is_dir() and "eval_base" in sub.name:
                    harbor_out = sub / "harbor_output"
                    if harbor_out.exists():
                        for job in sorted(harbor_out.iterdir()):
                            print(f"      └── Job: {job.name}")
                            res_file = job / "result.json"
                            if res_file.exists():
                                try:
                                    data = json.loads(res_file.read_text(encoding="utf-8"))
                                    print("          Stats:", data.get("stats"))
                                except Exception as e:
                                    print("          Error reading result.json:", e)
                            for td in sorted(job.iterdir()):
                                if td.is_dir():
                                    t_res = td / "result.json"
                                    r_val = "N/A"
                                    if t_res.exists():
                                        try:
                                            r_val = json.loads(t_res.read_text(encoding="utf-8")).get("agent_result", {}).get("reward", "N/A")
                                        except Exception:
                                            pass
                                    print(f"          ├── Task: {td.name} | Reward: {r_val}")

@app.local_entrypoint()
def main():
    extract_all.remote()
