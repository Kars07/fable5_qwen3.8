"""Modal Application for Running Generation Benchmarks on Fine-Tuned SFT Checkpoints."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

# 1. Define Modal App
app = modal.App("fable5-qwen-generate-eval")

# 2. Persistent Volumes
volume_checkpoints = modal.Volume.from_name("fable5-sft-checkpoints", create_if_missing=True)
volume_hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

# 3. Image
eval_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "torch",
        "transformers>=4.45.0",
        "peft>=0.12.0",
        "bitsandbytes>=0.43.0",
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
    image=eval_image,
    gpu="A10G:4",
    cpu=8,
    memory=64 * 1024,
    timeout=3600,
    volumes={
        "/opt/artifacts": volume_checkpoints,
        "/cache": volume_hf_cache,
    },
)
def run_generation_eval(
    adapter_path: str = "/opt/artifacts/checkpoints/qwen_4bit_lora/best_checkpoint",
    max_new_tokens: int = 512,
    temperature: float = 0.6,
):
    """Run generation benchmark testing on Modal."""
    os.chdir("/opt")
    volume_checkpoints.reload()
    volume_hf_cache.reload()

    model_dir = "/cache/models/Qwen3.8-27B"

    print(f"[*] Checking adapter directory: {adapter_path}", flush=True)
    if os.path.exists(adapter_path):
        print(f"[+] Found adapter contents in {adapter_path}: {os.listdir(adapter_path)}", flush=True)
    else:
        print(f"[!] Warning: Adapter path {adapter_path} does not exist locally, will evaluate base model.", flush=True)

    cmd = [
        "python",
        "sft/evaluate_generation.py",
        "--base-model", model_dir,
        "--max-new-tokens", str(max_new_tokens),
        "--temperature", str(temperature),
    ]

    if adapter_path and os.path.exists(adapter_path):
        cmd.extend(["--adapter-path", adapter_path])

    print(f"[*] Executing generation evaluation: {' '.join(cmd)}\n", flush=True)

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
        raise RuntimeError(f"Generation evaluation failed with code: {process.returncode}")


@app.local_entrypoint()
def main(
    adapter: str = "/opt/artifacts/checkpoints/qwen_4bit_lora/best_checkpoint",
    tokens: int = 512,
    temp: float = 0.6,
):
    """Local entrypoint for generation benchmarks."""
    print("=" * 70, flush=True)
    print("LAUNCHING GENERATION EVALUATION BENCHMARK ON MODAL")
    print(f"Adapter:    {adapter}")
    print(f"Max Tokens: {tokens}")
    print(f"Temp:       {temp}")
    print("=" * 70, flush=True)
    run_generation_eval.remote(
        adapter_path=adapter,
        max_new_tokens=tokens,
        temperature=temp,
    )
