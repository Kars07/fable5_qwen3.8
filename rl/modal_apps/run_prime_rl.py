"""Modal runner for Prime-RL training and dry-runs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

PRIME_RL_REV = "8c1f196dd39699726ee8ff52f6ee2495c5fa38df"
PRIME_RL_DIR = "/opt/prime-rl"

app = modal.App("fable5-prime-rl-runner")

# Persistent Cloud Volumes
volume_hf = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
volume_vllm = modal.Volume.from_name("fable5-vllm-cache", create_if_missing=True)
volume_sft = modal.Volume.from_name("fable5-sft-checkpoints", create_if_missing=True)
volume_outputs = modal.Volume.from_name("fable5-prime-rl-outputs", create_if_missing=True)
volume_checkpoints = modal.Volume.from_name("fable5-rl-checkpoints", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("git", "curl", "build-essential", "tmux", "sqlite3", "xxd")
    .pip_install("uv==0.11.21")
    .run_commands(
        f"git clone https://github.com/PrimeIntellect-ai/prime-rl.git {PRIME_RL_DIR}",
        f"cd {PRIME_RL_DIR} && git checkout {PRIME_RL_REV}",
        (
            f"cd {PRIME_RL_DIR} && "
            "git -c url.https://github.com/.insteadOf=git@github.com: "
            "submodule update --init deps/pydantic-config deps/renderers deps/verifiers"
        ),
        f"cd {PRIME_RL_DIR} && uv sync --frozen --no-dev",
    )
    .add_local_dir("rl", remote_path="/opt/rl", copy=True)
    .add_local_dir("rl_dataset", remote_path="/opt/rl_dataset", copy=True)
    .add_local_dir("verifiers", remote_path="/opt/verifiers", copy=True)
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface/hub",
            "VLLM_CACHE_ROOT": "/cache/vllm",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/opt/rl:/opt/prime-rl:/opt/verifiers",
        }
    )
)


def _run(argv: list[str], *, cwd: str | None = None, timeout: int = 1800) -> dict[str, object]:
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    return {
        "command": " ".join(argv),
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@app.function(
    image=image,
    gpu="A10G",
    cpu=4,
    memory=16384,
    volumes={
        "/cache/huggingface": volume_hf,
        "/cache/vllm": volume_vllm,
        "/opt/artifacts": volume_sft,
        "/outputs": volume_outputs,
        "/checkpoints": volume_checkpoints,
    },
    timeout=3600,
)
def run_rl(config_path: str = "/opt/rl/configs/rl/repo_repair_smoke.toml", dry_run: bool = True) -> dict[str, object]:
    """Execute Prime-RL training or dry-run."""
    command = [
        "uv", "run", "--no-project", "rl",
        "@", config_path,
        "--output-dir", "/outputs/prime-rl-run",
        "--no-wandb",
    ]
    if dry_run:
        command.append("--dry-run")

    result = _run(command, cwd=PRIME_RL_DIR)
    volume_outputs.commit()
    volume_checkpoints.commit()
    return result


@app.local_entrypoint()
def main(config: str = "rl/configs/rl/repo_repair_smoke.toml", dry_run: bool = True) -> None:
    """Local entrypoint for running Prime-RL."""
    print("=" * 80)
    print(f"[*] Launching Prime-RL on Modal (Dry Run: {dry_run})")
    print(f"[*] Config: {config}")
    print("=" * 80)
    res = run_rl.remote(config_path=config, dry_run=dry_run)
    print("\n[Command]:", res.get("command"))
    print("[Exit Code]:", res.get("exit_code"))
    print("\n[STDOUT]:\n", res.get("stdout"))
    if res.get("stderr"):
        print("\n[STDERR]:\n", res.get("stderr"))
