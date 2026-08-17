"""Central configuration constants and path definitions for Self-Hosted Rollout & RL."""

from __future__ import annotations

import os
from pathlib import Path

# Git Revisions
PRIME_RL_REV = "8c1f196dd39699726ee8ff52f6ee2495c5fa38df"
VERIFIERS_REV = "7251c60934d2c42af85d42a1da3da62269b7957e"

# Model Defaults
BASE_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_INFERENCE_MODEL = "Qwen/Qwen3-1.7B"
SFT_ADAPTER_PATH = "/opt/artifacts/checkpoints/qwen_4bit_lora/best_checkpoint"

# Remote Container Directories
PRIME_RL_DIR = "/opt/prime-rl"
VERIFIERS_DIR = "/opt/verifiers"
RL_DIR = "/opt/rl"
RL_DATASET_DIR = "/opt/rl_dataset"
HARNESS_DIR = "/opt/rl/harnesses"
TASKSETS_DIR = "/opt/rl/tasksets"
CONFIGS_DIR = "/opt/rl/configs"
OUTPUTS_DIR = "/outputs"

# Python Binary in Container
PYTHON = f"{PRIME_RL_DIR}/.venv/bin/python"

# Wheel URLs
VLLM_ROUTER_WHEEL = (
    "https://github.com/PrimeIntellect-ai/router/releases/download/v0.1.26/"
    "vllm_router-0.1.26-cp38-abi3-manylinux_2_28_x86_64.whl"
)
FLASH_ATTN_WHEEL = (
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/"
    "download/v0.9.4/flash_attn-2.8.3+cu128torch2.11-"
    "cp312-cp312-linux_x86_64.whl"
)

# Modal Volume Names
VOLUME_HF_CACHE_NAME = "hf-model-cache"
VOLUME_VLLM_CACHE_NAME = "fable5-vllm-cache"
VOLUME_OUTPUTS_NAME = "fable5-prime-rl-outputs"
VOLUME_SFT_CHECKPOINTS_NAME = "fable5-sft-checkpoints"
VOLUME_RL_CHECKPOINTS_NAME = "fable5-rl-checkpoints"
