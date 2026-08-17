"""Training orchestrator for Prime-RL with GRPO, PPO, and MAX-RL on Verifiers environments."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from rl.rollout.config import OUTPUTS_DIR, PRIME_RL_DIR, PYTHON


def run_prime_rl_training(
    config_path: str,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
    use_wandb: bool = False,
    timeout: int = 7200,
) -> Dict[str, Any]:
    """Execute Prime-RL training loop in container."""
    if output_dir is None:
        output_dir = f"{OUTPUTS_DIR}/prime-rl-train-{int(time.time())}"

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv", "run", "--no-project", "rl", "@", config_path,
        "--output-dir", str(out_path),
    ]

    if dry_run:
        cmd.append("--dry-run")
    if not use_wandb:
        cmd.append("--no-wandb")

    print(f"[*] Executing Prime-RL: {' '.join(cmd)}", flush=True)
    start = time.time()
    res = subprocess.run(
        cmd,
        cwd=PRIME_RL_DIR,
        text=True,
        capture_output=True,
        timeout=timeout,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    elapsed = time.time() - start

    return {
        "command": " ".join(cmd),
        "exit_code": res.returncode,
        "stdout": res.stdout,
        "stderr": res.stderr,
        "output_dir": str(out_path),
        "elapsed_seconds": round(elapsed, 2),
    }


def resume_prime_rl_training(
    checkpoint_path: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Resume Prime-RL training from a saved checkpoint."""
    cmd = [
        "uv", "run", "--no-project", "rl",
        "--resume", checkpoint_path,
    ]
    if output_dir:
        cmd.extend(["--output-dir", output_dir])

    print(f"[*] Resuming Prime-RL from {checkpoint_path}: {' '.join(cmd)}", flush=True)
    res = subprocess.run(
        cmd,
        cwd=PRIME_RL_DIR,
        text=True,
        capture_output=True,
        timeout=7200,
    )
    return {
        "command": " ".join(cmd),
        "exit_code": res.returncode,
        "stdout": res.stdout,
        "stderr": res.stderr,
    }
