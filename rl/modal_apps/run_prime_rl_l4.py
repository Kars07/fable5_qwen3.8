"""Modal runner for Prime-RL training and dry-runs on 2x A100-80GB GPUs with live logging."""

from __future__ import annotations

import os
import shutil
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
    .pip_install("uv==0.11.21", "huggingface_hub>=0.24.0")
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
        f"uv pip install --python {PYTHON} huggingface_hub",
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
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)


@app.function(
    image=image,
    gpu="L4:4",
    cpu=16,
    memory=65536,
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
    config_path: str = "/opt/rl/configs/rl/nemotron_terminal_grpo_l4.toml",
    config_content: str | None = None,
    hf_token: str | None = None,
    dry_run: bool = False,
    steps: int | None = None,
) -> dict[str, object]:
    """Execute Prime-RL training or dry-run with live rollout streaming."""
    volume_hf.reload()
    volume_sft.reload()

    # Sync SFT checkpoint if needed
    sft_dir = Path("/opt/artifacts/checkpoints/qwen_4bit_lora/best_checkpoint")
    if not sft_dir.exists() or not (sft_dir / "adapter_model.safetensors").exists():
        print("[*] Syncing fine-tuned SFT checkpoint from Hugging Face: eniairaph07/qwen3.8-27b-fable5...", flush=True)
        from huggingface_hub import snapshot_download
        token = hf_token or os.environ.get("HF_TOKEN")
        try:
            snapshot_download(
                repo_id="eniairaph07/qwen3.8-27b-fable5",
                local_dir=str(sft_dir),
                token=token,
            )
            volume_sft.commit()
            print("[+] SFT checkpoint successfully verified and cached to volume.", flush=True)
        except Exception as e:
            print(f"[!] HF Sync note: {e}", flush=True)

    run_dir = Path("/outputs/prime-rl-run")
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    if config_content:
        config_file = Path("/tmp/active_run_config.toml")
        config_file.write_text(config_content, encoding="utf-8")
        resolved_config = str(config_file)
    else:
        p = Path(config_path)
        if not p.is_absolute():
            if (Path("/opt") / config_path).exists():
                p = Path("/opt") / config_path
            elif (Path("/opt/rl") / config_path).exists():
                p = Path("/opt/rl") / config_path
            else:
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

    print(f"[*] Active Run Config:\n{Path(resolved_config).read_text(encoding='utf-8')}\n", flush=True)
    print(f"[*] Executing Command: {' '.join(command)}\n", flush=True)

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

    output_lines = []
    for line in process.stdout:
        output_lines.append(line)
        print(line, end="", flush=True)

    process.wait()

    # Dump all log files
    for search_root in [Path("/outputs/prime-rl-run"), Path("/tmp")]:
        if search_root.exists():
            for log_file in search_root.rglob("*.log"):
                try:
                    content = log_file.read_text(encoding="utf-8", errors="replace")
                    if content.strip():
                        print(f"\n{'=' * 40} {log_file} (Total Length: {len(content)}) {'=' * 40}\n", flush=True)
                        print(content[-20000:], flush=True)
                except Exception as e:
                    print(f"[-] Could not read {log_file}: {e}", flush=True)

    volume_outputs.commit()
    volume_checkpoints.commit()

    trainer_success = any("SUCCESS RL trainer finished!" in l for l in output_lines)
    if process.returncode != 0 and not trainer_success:
        raise RuntimeError(f"Prime-RL exited with returncode {process.returncode}")

    return {
        "command": " ".join(command),
        "exit_code": 0 if trainer_success else process.returncode,
    }


@app.local_entrypoint()
def main(
    config: str = "rl/configs/rl/nemotron_terminal_grpo_l4.toml",
    dry_run: bool = False,
    steps: int | None = None,
) -> None:
    """Local entrypoint for running Prime-RL."""
    config_file = Path(config)
    config_content = config_file.read_text(encoding="utf-8") if config_file.exists() else None

    # Retrieve HF token from .env
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("HF_TOKEN"):
                    token = line.split("=", 1)[1].strip().strip("'\"")

    print("=" * 80)
    print(f"[*] Launching Prime-RL on Modal (Model: Qwen3.8-27B on 2x A100-80GB | Dry Run: {dry_run})")
    print(f"[*] Config Path: {config}")
    if steps:
        print(f"[*] Max Steps Override: {steps}")
    print("=" * 80)
    run_rl.remote(
        config_path=config,
        config_content=config_content,
        hf_token=token,
        dry_run=dry_run,
        steps=steps,
    )

