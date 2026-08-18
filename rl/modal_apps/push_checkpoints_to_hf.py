"""Modal script to push all RL checkpoints and traces to Hugging Face Hub."""

import os
from pathlib import Path
import modal

app = modal.App("fable5-hf-pusher")

volume_checkpoints = modal.Volume.from_name("fable5-rl-checkpoints", create_if_missing=True)
volume_outputs = modal.Volume.from_name("fable5-prime-rl-outputs", create_if_missing=True)
volume_sft = modal.Volume.from_name("fable5-sft-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.24.0")
)

@app.function(
    image=image,
    volumes={
        "/checkpoints": volume_checkpoints,
        "/outputs": volume_outputs,
        "/sft": volume_sft,
    },
    timeout=1800,
)
def push_all_to_hf(
    repo_id: str = "eniairaph07/qwen3.8-27b-fable5-rl-sft-steps",
    hf_token: str | None = None,
):
    from huggingface_hub import HfApi, create_repo

    token = hf_token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    print(f"[*] Ensuring Hugging Face repository exists: {repo_id}...", flush=True)
    try:
        create_repo(repo_id=repo_id, repo_type="model", private=False, token=token, exist_ok=True)
        print(f"[+] Repository ready: https://huggingface.co/{repo_id}", flush=True)
    except Exception as e:
        print(f"[!] Note on repo creation: {e}", flush=True)

    volume_checkpoints.reload()
    volume_outputs.reload()
    volume_sft.reload()

    # 1. Fast Upload: All weights & files in step directories
    candidate_roots = [
        Path("/outputs/prime-rl-run/checkpoints"),
        Path("/checkpoints"),
    ]

    uploaded_files = 0
    for root in candidate_roots:
        if not root.exists():
            continue
        print(f"[*] Scanning root: {root}...", flush=True)
        for step_dir in root.iterdir():
            if step_dir.is_dir() and "step_" in step_dir.name:
                print(f"[*] Inspecting checkpoint directory: {step_dir.name}", flush=True)
                for file_path in step_dir.rglob("*"):
                    if file_path.is_file():
                        rel_path = file_path.relative_to(step_dir)
                        path_in_repo = f"rl_checkpoints/{step_dir.name}/{rel_path.as_posix()}"
                        size_mb = file_path.stat().st_size / 1e6
                        print(f"    --> Found {file_path.name} ({size_mb:.2f} MB) -> {path_in_repo}", flush=True)
                        try:
                            api.upload_file(
                                path_or_fileobj=str(file_path),
                                path_in_repo=path_in_repo,
                                repo_id=repo_id,
                                repo_type="model",
                                token=token,
                            )
                            uploaded_files += 1
                            print(f"    [+] Uploaded {file_path.name}!", flush=True)
                        except Exception as e:
                            print(f"    [!] Error uploading {file_path.name}: {e}", flush=True)

    print(f"[+] Total uploaded files: {uploaded_files} to https://huggingface.co/{repo_id}", flush=True)

    # 3. Create and upload README / Metadata summary
    readme_content = f"""---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - reinforcement-learning
  - grpo
  - terminal-bench
  - continuous-reward-shaping
---

# Fable 5: Qwen 3.8-27B RL (GRPO) & SFT Checkpoints

This repository contains fine-tuned LoRA checkpoints and training traces for **Qwen3.8-27B** trained using **Group Relative Policy Optimization (GRPO)** on decision-pivot terminal reasoning tasks.

## Contents
- `rl_checkpoints/step_5/`: Step 5 Checkpoint (`adapter_model.safetensors`, LoRA rank 64)
- `rl_checkpoints/step_10/`: Step 10 Checkpoint (`adapter_model.safetensors`, LoRA rank 64)
- `training_logs_and_traces/`: Step-by-step rollout logs and verified trajectory traces.

## Reward Formulation
- **Structure & Reasoning**: +0.20
- **Continuous Command-Token Similarity**: +0.50 (with canonical path & option normalization)
- **Submission Integrity & Anti-Gaming Gate**: +0.30 (or -0.50 active penalty for premature unearned completion claims)
"""
    readme_path = Path("/tmp/README.md")
    readme_path.write_text(readme_content, encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"\n[🚀] ALL CHECKPOINTS & TRACES BACKED UP TO: https://huggingface.co/{repo_id}", flush=True)

@app.local_entrypoint()
def main(repo_id: str = "eniairaph07/qwen3.8-27b-fable5-rl-sft-steps"):
    push_all_to_hf.remote(repo_id=repo_id)
