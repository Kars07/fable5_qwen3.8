"""Extract clean, lightweight PEFT LoRA adapters (~300 MB) from 113 GB PyTorch Distributed Checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import modal

app = modal.App("fable5-extract-lora-adapter")

volume_outputs = modal.Volume.from_name("fable5-prime-rl-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4.0",
        "safetensors>=0.4.5",
        "huggingface_hub>=0.24.0",
        "peft>=0.11.0",
        "transformers>=4.48.0",
    )
    .env({"PYTHONUNBUFFERED": "1"})
)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    volumes={"/outputs": volume_outputs},
    timeout=1800,
)
def extract_and_upload_all_loras(
    repo_id: str = "eniairaph07/qwen3.8-27b-fable5-rl-sft-steps",
    hf_token: str | None = None,
) -> dict[str, Any]:
    import torch
    import torch.distributed.checkpoint as dcp
    from huggingface_hub import HfApi
    from safetensors.torch import save_file

    token = hf_token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    volume_outputs.reload()
    ckpt_root = Path("/outputs/prime-rl-run/checkpoints")
    if not ckpt_root.exists():
        return {"error": f"Checkpoint root {ckpt_root} not found"}

    step_dirs = sorted([d for d in ckpt_root.iterdir() if d.is_dir() and "step_" in d.name], key=lambda x: x.name)
    print(f"[*] Found {len(step_dirs)} step directories: {[d.name for d in step_dirs]}", flush=True)

    lora_config = {
        "auto_mapping": None,
        "base_model_name_or_path": "Qwen/Qwen3.8-27B",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": 128.0,
        "lora_dropout": 0.05,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 64,
        "rank_pattern": {},
        "revision": None,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }

    uploaded = []

    for s_dir in step_dirs:
        step_name = s_dir.name
        trainer_dir = s_dir / "trainer"
        if not trainer_dir.exists():
            print(f"[!] No trainer dir in {step_name}, skipping...", flush=True)
            continue

        print(f"\n=======================================================", flush=True)
        print(f"[*] Processing {step_name} at {trainer_dir}...", flush=True)

        # 1. Load state dict via PyTorch DCP FileSystemReader
        try:
            from torch.distributed.checkpoint import FileSystemReader
            reader = FileSystemReader(trainer_dir)
            metadata = reader.read_metadata()
            print(f"[+] Loaded DCP metadata for {step_name}: {len(metadata.state_dict_metadata)} tensor keys found", flush=True)

            # Filter only LoRA keys to avoid loading 113 GB of master weights
            lora_keys = [k for k in metadata.state_dict_metadata.keys() if "lora" in k.lower()]
            print(f"[+] Found {len(lora_keys)} LoRA tensor keys in checkpoint", flush=True)

            # Load only LoRA tensors
            state_dict = {k: torch.empty(metadata.state_dict_metadata[k].size, dtype=metadata.state_dict_metadata[k].properties.dtype) for k in lora_keys}
            dcp.load(state_dict=state_dict, storage_reader=reader)

            # Standardize key names for PEFT (base_model.model...)
            clean_state_dict = {}
            for k, v in state_dict.items():
                clean_k = k
                if not clean_k.startswith("base_model.model."):
                    clean_k = f"base_model.model.{clean_k}"
                clean_state_dict[clean_k] = v.contiguous().to(torch.bfloat16)

            # Save clean ~300 MB safetensors adapter locally
            out_dir = Path(f"/tmp/clean_lora/{step_name}")
            out_dir.mkdir(parents=True, exist_ok=True)
            adapter_file = out_dir / "adapter_model.safetensors"
            config_file = out_dir / "adapter_config.json"

            save_file(clean_state_dict, str(adapter_file))
            config_file.write_text(json.dumps(lora_config, indent=2), encoding="utf-8")

            size_mb = adapter_file.stat().st_size / 1e6
            print(f"[+] Successfully extracted clean LoRA adapter: {size_mb:.2f} MB (vs 113 GB raw)", flush=True)

            # Upload to Hugging Face
            print(f"[*] Uploading clean adapter to {repo_id}/rl_checkpoints/{step_name}/...", flush=True)
            api.upload_file(
                path_or_fileobj=str(adapter_file),
                path_in_repo=f"rl_checkpoints/{step_name}/adapter_model.safetensors",
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )
            api.upload_file(
                path_or_fileobj=str(config_file),
                path_in_repo=f"rl_checkpoints/{step_name}/adapter_config.json",
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )
            print(f"[🚀] {step_name} LoRA adapter is LIVE on Hugging Face Hub!", flush=True)
            uploaded.append(step_name)

        except Exception as e:
            print(f"[!] Error extracting LoRA for {step_name}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    return {"uploaded_steps": uploaded}


@app.local_entrypoint()
def main():
    res = extract_and_upload_all_loras.remote()
    print("Result:", res)
