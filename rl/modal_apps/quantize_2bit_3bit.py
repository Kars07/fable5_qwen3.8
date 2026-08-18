"""Quantize Step 20 merged model to 2-bit (Q2_K) and 3-bit (Q3_K_M) GGUF and upload to Hugging Face."""

import modal
import os
import subprocess
from pathlib import Path

app = modal.App("quantize-2bit-3bit-fable")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "cmake", "curl", "wget")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.48.0",
        "peft>=0.12.0",
        "safetensors>=0.4.3",
        "accelerate>=0.33.0",
        "huggingface_hub>=0.24.0",
        "sentencepiece",
        "protobuf",
        "numpy",
        "rich",
    )
    .run_commands(
        "git clone https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp",
        "cd /opt/llama.cpp && cmake -B build && cmake --build build --config Release -j$(nproc)",
        "pip install -r /opt/llama.cpp/requirements.txt",
        "pip install --upgrade git+https://github.com/huggingface/transformers.git peft>=0.14.0 accelerate>=1.0.0",
    )
)

volume_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

@app.function(
    image=image,
    cpu=16,
    memory=128000,
    timeout=7200,
    volumes={"/cache/huggingface": volume_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def quantize_and_upload(
    checkpoint_step: str = "step_20",
    repo_name: str = "Qwen3.8-27b-FABLE-GGUF",
    quant_types: list[str] = ["Q2_K", "Q3_K_M"],
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import HfApi, snapshot_download, create_repo
    from safetensors.torch import load_file, save_file
    import json
    from rich.console import Console

    console = Console()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN secret not found in environment!")

    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info.get("name", "eniairaph07")
    orgs = [o.get("name") for o in user_info.get("orgs", []) if isinstance(o, dict)]
    if "officialpathwiseai" in orgs:
        target_repo = f"officialpathwiseai/{repo_name}"
    else:
        target_repo = f"{username}/{repo_name}"
    console.print(f"[bold green][*] Target Hugging Face Repo: {target_repo}[/bold green]")

    # 1. Download base model & LoRA adapter
    console.print("[bold cyan][1/5] Downloading Qwen/Qwen3.8-27B & Step 20 LoRA adapter...[/bold cyan]")
    lora_raw_dir = Path(f"/tmp/lora_raw_{checkpoint_step}")
    lora_raw_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot_download(
        repo_id="eniairaph07/qwen3.8-27b-fable5-rl-sft-steps",
        allow_patterns=[
            f"rl_checkpoints/{checkpoint_step}/*",
            f"rl_checkpoints/{checkpoint_step}/**/*",
            f"rl_checkpoints/{checkpoint_step}/trainer/*",
        ],
        local_dir=str(lora_raw_dir),
        token=token,
    )
    src_adapter_dir = lora_raw_dir / "rl_checkpoints" / checkpoint_step
    
    adapter_file = src_adapter_dir / "adapter_model.safetensors"
    if not adapter_file.exists():
        adapter_file = src_adapter_dir / "model.safetensors"
        
    raw_weights = load_file(str(adapter_file))
    clean_state_dict = {}
    for k, v in raw_weights.items():
        clean_k = k
        if any(opt in clean_k.lower() for opt in ["optimizer", "optimizers", "exp_avg", "step", "state."]):
            continue
        for prefix in [
            "actor.", "critic.", "module.", "policy.", "app.model.layers.",
            "language_model.model.", "language_model.", "model.model."
        ]:
            if clean_k.startswith(prefix):
                clean_k = clean_k[len(prefix):]
        if not clean_k.startswith("base_model.model."):
            clean_k = "base_model.model." + clean_k
        if clean_k.endswith(".0"):
            clean_k = clean_k[:-2]
        clean_state_dict[clean_k] = v.contiguous().to(torch.bfloat16)

    sanitized_lora_dir = Path("/tmp/sanitized_lora")
    sanitized_lora_dir.mkdir(parents=True, exist_ok=True)
    save_file(clean_state_dict, str(sanitized_lora_dir / "adapter_model.safetensors"))
    
    lora_cfg = {
        "base_model_name_or_path": "Qwen/Qwen3.8-27B",
        "bias": "none",
        "inference_mode": True,
        "peft_type": "LORA",
        "r": 64,
        "lora_alpha": 128.0,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "task_type": "CAUSAL_LM",
    }
    (sanitized_lora_dir / "adapter_config.json").write_text(json.dumps(lora_cfg, indent=2), encoding="utf-8")
    console.print("[bold green][+] Sanitized LoRA adapter ready.[/bold green]")

    # 2. Merge LoRA with Base Model
    console.print("[bold cyan][2/5] Merging LoRA weights in bfloat16 (128 GB RAM)...[/bold cyan]")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B", trust_remote_code=True, cache_dir="/cache/huggingface")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.8-27B",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        cache_dir="/cache/huggingface",
        trust_remote_code=True,
    )
    
    model = PeftModel.from_pretrained(base_model, str(sanitized_lora_dir))
    merged_model = model.merge_and_unload()
    console.print("[bold green][+] Model merged successfully.[/bold green]")

    merged_hf_dir = Path("/tmp/merged_hf")
    merged_hf_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_hf_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_hf_dir))

    del merged_model
    del base_model
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 3. Convert to base FP16 GGUF
    console.print("[bold cyan][3/5] Converting merged HF model to GGUF format via llama.cpp...[/bold cyan]")
    gguf_dir = Path("/tmp/gguf_output")
    gguf_dir.mkdir(parents=True, exist_ok=True)
    f16_gguf = gguf_dir / f"{repo_name}-f16.gguf"

    conv_cmd = [
        "python3", "/opt/llama.cpp/convert_hf_to_gguf.py",
        str(merged_hf_dir),
        "--outfile", str(f16_gguf),
        "--outtype", "f16",
    ]
    console.print(f"Running: {' '.join(conv_cmd)}")
    subprocess.run(conv_cmd, check=True)
    console.print(f"[bold green][+] Base GGUF created: {f16_gguf}[/bold green]")

    # 4. Quantize to 2-bit (Q2_K) and 3-bit (Q3_K_M)
    console.print(f"[bold cyan][4/5] Quantizing GGUF models ({', '.join(quant_types)})...[/bold cyan]")
    quant_files = []
    quant_binary = "/opt/llama.cpp/build/bin/llama-quantize"
    if not Path(quant_binary).exists():
        quant_binary = "/opt/llama.cpp/build/bin/quantize"

    for qtype in quant_types:
        out_gguf = gguf_dir / f"{repo_name}-{qtype}.gguf"
        console.print(f"[*] Quantizing to {qtype} -> {out_gguf.name}...")
        subprocess.run([quant_binary, str(f16_gguf), str(out_gguf), qtype], check=True)
        console.print(f"[+] Finished {qtype}: {out_gguf.stat().st_size / (1024**3):.2f} GB")
        quant_files.append(out_gguf)

    # 5. Upload 2-bit & 3-bit GGUF binaries to Hugging Face
    console.print(f"[bold cyan][5/5] Uploading 2-bit and 3-bit GGUFs to {target_repo}...[/bold cyan]")
    for qf in quant_files:
        console.print(f"[*] Uploading {qf.name} ({qf.stat().st_size / (1024**3):.2f} GB)...")
        api.upload_file(
            path_or_fileobj=str(qf),
            path_in_repo=qf.name,
            repo_id=target_repo,
            repo_type="model",
        )

    console.print(f"\n[bold green]🎉 SUCCESS! Uploaded {', '.join(quant_types)} to Hugging Face:[/bold green]")
    console.print(f"👉 https://huggingface.co/{target_repo}")

@app.local_entrypoint()
def main():
    quantize_and_upload.remote(
        checkpoint_step="step_20",
        repo_name="Qwen3.8-27b-FABLE-GGUF",
        quant_types=["Q2_K", "Q3_K_M"],
    )
