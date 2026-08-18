"""Inspect step_20 checkpoint in fable5-prime-rl-outputs in officialpathwiseai."""

import modal
from pathlib import Path

app = modal.App("inspect-step20-weights")
image = modal.Image.debian_slim(python_version="3.12")
volume_rl = modal.Volume.from_name("fable5-prime-rl-outputs")

@app.function(image=image, volumes={"/rl": volume_rl})
def check_weights():
    rl_base = Path("/rl")
    print("=== CONTENTS OF /rl ===")
    for p in rl_base.rglob("*"):
        if "step_20" in str(p) or "step_25" in str(p):
            if p.is_file():
                print(f"File: {p} ({p.stat().st_size / (1024*1024):.2f} MB)")

@app.local_entrypoint()
def main():
    check_weights.remote()
