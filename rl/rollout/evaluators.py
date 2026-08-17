"""Self-hosted evaluation runners for Harbor, Terminal-Bench 2.0, Verifiers v1, and Harnesses."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rl.rollout.config import DEFAULT_INFERENCE_MODEL, OUTPUTS_DIR, PRIME_RL_DIR, PYTHON, SFT_ADAPTER_PATH
from rl.rollout.inference import launch_prime_inference_server


def run_harbor_e2b_eval(
    e2b_api_key: str,
    dataset: str = "terminal-bench/terminal-bench-2",
    harness: str = "terminus_2",
    num_tasks: int = 5,
    model: str = DEFAULT_INFERENCE_MODEL,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute Harbor Terminal-Bench evaluation on E2B microVMs using self-hosted Prime inference."""
    if output_dir is None:
        output_dir = f"{OUTPUTS_DIR}/harbor-e2b-{int(time.time())}"

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Write E2B API Key
    key_file = Path("/tmp/e2b_api_key")
    key_file.write_text(e2b_api_key.strip(), encoding="utf-8")
    key_file.chmod(0o600)

    # 2. Launch Local Prime Inference Server
    inference_log = out_path / "inference.log"
    proc, _ = launch_prime_inference_server(model=model, log_file=inference_log)

    runner_log = out_path / "runner.log"
    results_json = out_path / "results.json"

    try:
        # 3. Build Harbor Command
        cmd = [
            "harbor", "run",
            "-d", dataset,
            "-m", f"openai/{model}",
            "-a", harness,
            "--env", "e2b",
            "-n", str(num_tasks),
            "--output-dir", str(out_path),
        ]

        env = {
            **os.environ,
            "E2B_API_KEY": e2b_api_key,
            "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
            "OPENAI_API_KEY": "task2-prime-token",
            "PYTHONUNBUFFERED": "1",
        }

        print(f"[*] Executing Harbor evaluation: {' '.join(cmd)}", flush=True)
        with runner_log.open("w", encoding="utf-8") as handle:
            harbor_res = subprocess.run(
                cmd,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                timeout=3600,
            )

        return {
            "exit_code": harbor_res.returncode,
            "output_dir": str(out_path),
            "log": runner_log.read_text(encoding="utf-8")[-2000:] if runner_log.exists() else "",
            "results_path": str(results_json),
        }
    finally:
        proc.terminate()
        proc.wait()


def run_verifiers_evaluation(
    taskset_id: str = "rl_pivot_terminal",
    harness_id: str = "terminus_2",
    runtime_type: str = "subprocess",
    num_tasks: int = 10,
    num_rollouts: int = 1,
    base_url: str = "http://127.0.0.1:8000/v1",
    model: str = DEFAULT_INFERENCE_MODEL,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute Verifiers v1 evaluation pipeline."""
    if output_dir is None:
        output_dir = f"{OUTPUTS_DIR}/verifiers-eval-{int(time.time())}"

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv", "run", "--no-project", "eval", taskset_id,
        "-n", str(num_tasks),
        "--num-rollouts", str(num_rollouts),
        "--env.agent.harness.id", harness_id,
        "--env.agent.runtime.type", runtime_type,
        "--model", model,
        "--client.base-url", base_url,
        "--client.api-key", "test-key",
        "--output-dir", str(out_path),
        "--no-push",
    ]

    print(f"[*] Running Verifiers v1 Eval: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, text=True, capture_output=True, timeout=1800)

    return {
        "command": " ".join(cmd),
        "exit_code": res.returncode,
        "stdout": res.stdout,
        "stderr": res.stderr,
        "output_dir": str(out_path),
    }
