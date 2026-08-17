"""Modal application, container image, and persistent volume definitions."""

from __future__ import annotations

import modal
from rl.rollout.config import (
    CONFIGS_DIR,
    FLASH_ATTN_WHEEL,
    HARNESS_DIR,
    PRIME_RL_DIR,
    PRIME_RL_REV,
    PYTHON,
    RL_DATASET_DIR,
    RL_DIR,
    TASKSETS_DIR,
    VERIFIERS_DIR,
    VERIFIERS_REV,
    VLLM_ROUTER_WHEEL,
    VOLUME_HF_CACHE_NAME,
    VOLUME_OUTPUTS_NAME,
    VOLUME_RL_CHECKPOINTS_NAME,
    VOLUME_SFT_CHECKPOINTS_NAME,
    VOLUME_VLLM_CACHE_NAME,
)

# 1. Define Modal App
app = modal.App("fable5-prime-rl-rollout")

# 2. Define Persistent Cloud Volumes
hf_cache = modal.Volume.from_name(VOLUME_HF_CACHE_NAME, create_if_missing=True)
vllm_cache = modal.Volume.from_name(VOLUME_VLLM_CACHE_NAME, create_if_missing=True)
sft_checkpoints = modal.Volume.from_name(VOLUME_SFT_CHECKPOINTS_NAME, create_if_missing=True)
rl_outputs = modal.Volume.from_name(VOLUME_OUTPUTS_NAME, create_if_missing=True)
rl_checkpoints = modal.Volume.from_name(VOLUME_RL_CHECKPOINTS_NAME, create_if_missing=True)

# 3. Define Comprehensive GPU / RL Container Image
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git", "curl", "build-essential", "tmux", "sqlite3", "xxd")
    .pip_install("uv==0.11.21")
    .run_commands(
        f"git clone https://github.com/PrimeIntellect-ai/prime-rl.git {PRIME_RL_DIR}",
        f"cd {PRIME_RL_DIR} && git checkout {PRIME_RL_REV}",
        (
            f"cd {PRIME_RL_DIR} && "
            "git -c url.https://github.com/.insteadOf=git@github.com: "
            "submodule update --init deps/prime-envs deps/pydantic-config "
            "deps/renderers deps/verifiers"
        ),
        f"cd {PRIME_RL_DIR} && uv sync --frozen --no-dev",
        f"uv pip install --python {PYTHON} --no-deps {VLLM_ROUTER_WHEEL}",
        f"uv pip install --python {PYTHON} --no-deps {FLASH_ATTN_WHEEL}",
        f"uv pip install --python {PYTHON} e2b==2.35.0",
        f"uv pip install --python {PYTHON} harbor==0.20.0",
        f"uv pip install --python {PYTHON} pydantic>=2.0 pyyaml rich tabulate",
    )
    .add_local_dir("rl", remote_path=RL_DIR, copy=True)
    .add_local_dir("rl_dataset", remote_path=RL_DATASET_DIR, copy=True)
    .add_local_dir("verifiers", remote_path=VERIFIERS_DIR, copy=True)
    .run_commands(
        f"uv pip install --python {PYTHON} -e {VERIFIERS_DIR}",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface/hub",
            "VLLM_CACHE_ROOT": "/cache/vllm",
            "HF_HUB_DOWNLOAD_TIMEOUT": "300",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{RL_DIR}:{PRIME_RL_DIR}:{VERIFIERS_DIR}",
        }
    )
)
