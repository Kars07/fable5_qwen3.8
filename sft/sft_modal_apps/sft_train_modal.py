"""Modal Application for Distributed SFT Training of Qwen on Fable-5 Traces with Model Parallelism on 8x L4 GPUs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

# 1. Define Modal App
app = modal.App("fable5-qwen-sft-training")

# 2. Persistent Volumes
volume_checkpoints = modal.Volume.from_name("fable5-sft-checkpoints", create_if_missing=True)
volume_hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

# 3. Base Training Image with Baked Project Layers
train_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "torch",
        "transformers>=4.45.0",
        "datasets>=2.19.0",
        "accelerate>=0.33.0",
        "peft>=0.12.0",
        "bitsandbytes>=0.43.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "rich>=13.7.0",
        "tabulate>=0.9.0",
        "sentencepiece",
        "tiktoken",
        "numpy",
        "huggingface_hub",
    )
    .env({
        "HF_HOME": "/cache/huggingface",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir("sft", remote_path="/opt/sft", copy=True)
    .add_local_dir("dataset", remote_path="/opt/dataset", copy=True)
    .add_local_dir("configs", remote_path="/opt/configs", copy=True)
)


@app.function(
    image=train_image,
    gpu="L4:8",  # 8x NVIDIA L4 Ada Lovelace GPUs (192 GB total VRAM)
    cpu=16,
    memory=128 * 1024,  # 128 GB Host RAM
    timeout=86400,
    volumes={
        "/opt/artifacts": volume_checkpoints,
        "/cache": volume_hf_cache,
    },
)
def run_model_parallel_training(
    config_path: str = "configs/sft_train.yaml",
    smoke_test: bool = True,
    max_seq_length: int = 8192,
    lora_r: int = 64,
    lora_alpha: int = 128,
):
    """Execute multi-GPU model-parallel training/smoke testing across 8x L4 GPUs."""
    os.chdir("/opt")
    print("=" * 70, flush=True)
    mode_desc = "SMOKE TEST (10 Steps)" if smoke_test else "FULL SFT RUN"
    print(f"[*] Starting {mode_desc} on 8x NVIDIA L4 Cluster...", flush=True)
    print(f"[*] Working Directory: {os.getcwd()}", flush=True)

    import torch
    print(f"[*] PyTorch Version: {torch.__version__}", flush=True)
    print(f"[*] CUDA Available:  {torch.cuda.is_available()}", flush=True)
    print(f"[*] GPU Count:       {torch.cuda.device_count()}", flush=True)
    for i in range(torch.cuda.device_count()):
        print(f"    - GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f} GB)", flush=True)

    # 1. Ensure model files exist in /cache/models/Qwen3.8-27B
    model_dir = "/cache/models/Qwen3.8-27B"
    os.makedirs(model_dir, exist_ok=True)
    print(f"\n[*] Ensuring model files exist in {model_dir}...", flush=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="Qwen/Qwen3.8-27B",
        local_dir=model_dir,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"],
        max_workers=8,
    )
    volume_hf_cache.commit()
    print("[+] Model snapshot verified and committed to volume.\n", flush=True)

    cmd = [
        "python",
        "sft/train_model_parallel.py",
        "--config", config_path,
        "--model-id", model_dir,
        "--lora",
        "--lora-r", str(lora_r),
        "--lora-alpha", str(lora_alpha),
        "--max-seq-length", str(max_seq_length),
    ]

    if smoke_test:
        cmd.append("--smoke-test")

    print(f"[*] Executing command: {' '.join(cmd)}\n", flush=True)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    for line in process.stdout:
        print(line, end="", flush=True)

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Training/Smoke Test failed with exit code: {process.returncode}")

    if not smoke_test:
        volume_checkpoints.commit()
        print("[+] Training completed. Checkpoints committed to persistent volume.", flush=True)
    else:
        print("[+] Smoke test completed successfully in < 2 minutes!", flush=True)


@app.local_entrypoint()
def main(
    config: str = "configs/sft_train.yaml",
    smoke_test: bool = True,
    seq_len: int = 8192,
    r: int = 64,
    alpha: int = 128,
):
    """Local entrypoint for launching Modal training / smoke test."""
    print("=" * 70, flush=True)
    mode_str = "SMOKE TEST (10 Steps)" if smoke_test else "FULL SFT TRAINING"
    print(f"LAUNCHING MODAL RUN: {mode_str}", flush=True)
    print(f"GPU Setup:       8x NVIDIA L4 (192 GB total VRAM)")
    print(f"Host Memory:     128 GB RAM")
    print(f"Parallelism:     Multi-GPU Pipeline Model Parallelism (device_map='auto')")
    print(f"Context Length:  {seq_len} tokens")
    print(f"LoRA Config:     r={r}, alpha={alpha}")
    print("=" * 70, flush=True)
    run_model_parallel_training.remote(
        config_path=config,
        smoke_test=smoke_test,
        max_seq_length=seq_len,
        lora_r=r,
        lora_alpha=alpha,
    )
