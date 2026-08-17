"""Self-hosted vLLM & Prime-RL inference server management within Modal containers."""

from __future__ import annotations

import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from rl.rollout.config import DEFAULT_INFERENCE_MODEL, PRIME_RL_DIR, PYTHON


def get_local_e2b_key() -> str:
    """Retrieve E2B API key from environment variable or .env file."""
    key = os.environ.get("E2B_API_KEY")
    if key:
        return key

    env_paths = [Path(".env"), Path("../.env"), Path(__file__).resolve().parents[2] / ".env"]
    for env_path in env_paths:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                name, separator, value = line.partition("=")
                if separator and name.strip() == "E2B_API_KEY":
                    val = value.strip().strip("'\"")
                    if val:
                        return val

    raise RuntimeError("E2B_API_KEY environment variable or .env entry is required for sandbox rollouts.")


def wait_for_inference_ready(
    process: subprocess.Popen[str],
    health_url: str = "http://127.0.0.1:8000/v1/models",
    timeout: int = 900,
) -> None:
    """Poll the local inference server endpoint until it returns HTTP 200."""
    start = time.time()
    last_error: Exception | None = None

    print(f"[*] Polling inference server at {health_url} (timeout: {timeout}s)...", flush=True)

    while time.time() - start < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"Inference server process exited unexpectedly with code: {process.returncode}")

        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    print(f"[+] Inference server is healthy and responding (elapsed: {time.time() - start:.1f}s).", flush=True)
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionResetError) as e:
            last_error = e

        time.sleep(2)

    raise TimeoutError(f"Inference server failed to become healthy within {timeout}s. Last error: {last_error}")


def launch_prime_inference_server(
    model: str = DEFAULT_INFERENCE_MODEL,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.85,
    tensor_parallel_size: int = 1,
    log_file: Optional[Path] = None,
) -> tuple[subprocess.Popen[str], Path]:
    """Launch background vLLM / Prime-RL inference daemon on container GPU."""
    if log_file is None:
        log_file = Path("/tmp/prime_inference.log")

    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv", "run", "--frozen", "inference",
        "--vllm.model", model,
        "--vllm.max-model-len", str(max_model_len),
        "--vllm.gpu-memory-utilization", str(gpu_memory_utilization),
        "--vllm.tensor-parallel-size", str(tensor_parallel_size),
    ]

    print(f"[*] Starting Prime-RL inference daemon: {' '.join(cmd)}", flush=True)
    handle = log_file.open("w", encoding="utf-8")

    process = subprocess.Popen(
        cmd,
        cwd=PRIME_RL_DIR,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    try:
        wait_for_inference_ready(process)
    except Exception:
        process.terminate()
        process.wait()
        if log_file.exists():
            print("\n[-] Inference Server Log Dump:")
            print(log_file.read_text(encoding="utf-8")[-2000:])
        raise

    return process, log_file
