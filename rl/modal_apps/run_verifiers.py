"""Modal runner for Verifiers v1 evaluations."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

VERIFIERS_REV = "7251c60934d2c42af85d42a1da3da62269b7957e"

app = modal.App("fable5-verifiers-v1-runner")

# Persistent Cloud Volumes
volume_hf = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
volume_sft = modal.Volume.from_name("fable5-sft-checkpoints", create_if_missing=True)
volume_outputs = modal.Volume.from_name("fable5-prime-rl-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "tmux", "sqlite3", "xxd")
    .pip_install("uv==0.11.21")
    .add_local_dir("rl", remote_path="/opt/rl", copy=True)
    .add_local_dir("rl_dataset", remote_path="/opt/rl_dataset", copy=True)
    .add_local_dir("verifiers", remote_path="/opt/verifiers", copy=True)
    .run_commands(
        "uv pip install --system /opt/verifiers",
    )
    .env({
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/opt/rl:/opt/verifiers",
    })
)


def _run(argv: list[str], timeout: int = 900, env: dict[str, str] | None = None) -> dict[str, object]:
    result = subprocess.run(argv, text=True, capture_output=True, timeout=timeout, env=env)
    return {
        "command": " ".join(argv),
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    volumes={
        "/cache/huggingface": volume_hf,
        "/opt/artifacts": volume_sft,
        "/outputs": volume_outputs,
    },
    timeout=1200,
)
def verifiers_eval(
    taskset: str = "rl_pivot_terminal",
    harness: str = "null",
    num_tasks: int = 5,
    dry_run: bool = True,
) -> dict[str, object]:
    """Run Verifiers v1 evaluation command."""
    command = [
        "uv", "run", "--no-project", "eval", taskset,
        "-n", str(num_tasks),
        "--env.agent.harness.id", harness,
        "--env.agent.runtime.type", "subprocess",
        "--output-dir", "/outputs/verifiers-eval",
        "--no-push",
    ]
    if dry_run:
        command.append("--dry-run")

    result = _run(command)
    volume_outputs.commit()
    return result


@app.local_entrypoint()
def main(
    taskset: str = "rl_pivot_terminal",
    harness: str = "null",
    tasks: int = 5,
    dry_run: bool = True,
) -> None:
    """Local entrypoint for running Verifiers evaluation."""
    print("=" * 80)
    print(f"[*] Running Verifiers Evaluation (Taskset: {taskset}, Harness: {harness})")
    print("=" * 80)
    res = verifiers_eval.remote(
        taskset=taskset,
        harness=harness,
        num_tasks=tasks,
        dry_run=dry_run,
    )
    print("\n[Command]:", res.get("command"))
    print("[Exit Code]:", res.get("exit_code"))
    print("\n[STDOUT]:\n", res.get("stdout"))
    if res.get("stderr"):
        print("\n[STDERR]:\n", res.get("stderr"))
