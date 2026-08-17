"""Modal runner for Prime-RL training and dry-runs with live rollout logging."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

PRIME_RL_REV = "8c1f196dd39699726ee8ff52f6ee2495c5fa38df"
PRIME_RL_DIR = "/opt/prime-rl"
PYTHON = f"{PRIME_RL_DIR}/.venv/bin/python"

VLLM_ROUTER_WHEEL = (
    "https://github.com/PrimeIntellect-ai/router/releases/download/v0.1.26/"
    "vllm_router-0.1.26-cp38-abi3-manylinux_2_28_x86_64.whl"
)
FLASH_ATTN_WHEEL = (
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/"
    "download/v0.9.4/flash_attn-2.8.3+cu128torch2.11-"
    "cp312-cp312-linux_x86_64.whl"
)

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
        f"uv pip install --python {PYTHON} --no-deps {VLLM_ROUTER_WHEEL}",
        f"uv pip install --python {PYTHON} --no-deps {FLASH_ATTN_WHEEL}",
    )
    .add_local_dir("rl", remote_path="/opt/rl", copy=True)
    .add_local_dir("rl_dataset", remote_path="/opt/rl_dataset", copy=True)
    .add_local_dir("verifiers", remote_path="/opt/verifiers", copy=True)
    .run_commands(
        f"uv pip install --python {PYTHON} -e /opt/verifiers",
    )
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


@app.function(
    image=image,
    gpu="A10G:4",
    cpu=8,
    memory=32768,
    volumes={
        "/cache/huggingface": volume_hf,
        "/cache/vllm": volume_vllm,
        "/opt/artifacts": volume_sft,
        "/outputs": volume_outputs,
        "/checkpoints": volume_checkpoints,
    },
    timeout=7200,
)
def run_rl(
    config_path: str = "/opt/rl/configs/rl/nemotron_terminal_grpo.toml",
    dry_run: bool = False,
    steps: int | None = None,
) -> dict[str, object]:
    """Execute Prime-RL training or dry-run with live rollout streaming."""
    volume_hf.reload()
    volume_sft.reload()

    # Resolve config path inside container
    p = Path(config_path)
    if not p.is_absolute():
        if (Path("/opt") / config_path).exists():
            p = Path("/opt") / config_path
        elif (Path("/opt/rl") / config_path).exists():
            p = Path("/opt/rl") / config_path
        else:
            # strip leading rl/ if present
            stripped = config_path.replace("rl/", "", 1) if config_path.startswith("rl/") else config_path
            p = Path("/opt/rl") / stripped

    resolved_config = str(p)

    command = [
        "uv", "run", "--no-project", "rl",
        "@", resolved_config,
        "--output-dir", "/outputs/prime-rl-run",
        "--no-wandb",
    ]
    if dry_run:
        command.append("--dry-run")
    if steps is not None:
        command.extend(["--max-steps", str(steps)])

    print(f"[*] Executing Prime-RL: {' '.join(command)}\n", flush=True)

    process = subprocess.Popen(
        command,
        cwd=PRIME_RL_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    for line in process.stdout:
        print(line, end="", flush=True)

    process.wait()

    # Dump logs if any process failed
    log_dir = Path("/outputs/prime-rl-run/logs")
    if log_dir.exists():
        for log_file in log_dir.glob("*.log"):
            print(f"\n{'=' * 40} {log_file.name} {'=' * 40}\n", flush=True)
            print(log_file.read_text(encoding="utf-8", errors="replace"), flush=True)

    volume_outputs.commit()
    volume_checkpoints.commit()

    if process.returncode != 0:
        raise RuntimeError(f"Prime-RL exited with returncode {process.returncode}")

    return {
        "command": " ".join(command),
        "exit_code": process.returncode,
    }


@app.local_entrypoint()
def main(
    config: str = "rl/configs/rl/nemotron_terminal_grpo.toml",
    dry_run: bool = False,
    steps: int | None = None,
) -> None:
    """Local entrypoint for running Prime-RL."""
    print("=" * 80)
    print(f"[*] Launching Prime-RL on Modal (Dry Run: {dry_run})")
    print(f"[*] Config: {config}")
    if steps:
        print(f"[*] Max Steps Override: {steps}")
    print("=" * 80)
    run_rl.remote(config_path=config, dry_run=dry_run, steps=steps)
