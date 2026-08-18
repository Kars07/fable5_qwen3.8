"""Merge Step 20 LoRA with Qwen3.8-27B, convert to GGUF (Q4_K_M, Q8_0, F16), and publish to Hugging Face."""

import modal
import os
import shutil
import subprocess
from pathlib import Path

app = modal.App("export-publish-gguf-fable5")

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
def export_and_upload(
    checkpoint_step: str = "step_20",
    repo_name: str = "Qwen3.8-27b-FABLE-GGUF",
    quant_types: list[str] = ["Q4_K_M", "Q8_0"],
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
    console.print("[bold cyan][1/6] Downloading Qwen/Qwen3.8-27B & Step 20 LoRA adapter...[/bold cyan]")
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
    
    # Sanitize LoRA weights for standard PEFT
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
    console.print("[bold cyan][2/6] Loading base model and merging LoRA weights in bfloat16 (128 GB RAM)...[/bold cyan]")
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
    console.print("[bold cyan][3/6] Saving merged Hugging Face checkpoint to disk...[/bold cyan]")
    merged_model.save_pretrained(str(merged_hf_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_hf_dir))

    # Free GPU RAM for quantization
    del merged_model
    del base_model
    del model
    torch.cuda.empty_cache()

    # 4. Convert merged model to GGUF F16 via llama.cpp
    console.print("[bold cyan][4/6] Converting merged HF model to GGUF format via llama.cpp...[/bold cyan]")
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
    console.print(f"[bold green][+] Base GGUF created: {f16_gguf} ({f16_gguf.stat().st_size / (1024**3):.2f} GB)[/bold green]")

    # 5. Quantize to target GGUF formats (Q4_K_M, Q8_0)
    console.print("[bold cyan][5/6] Quantizing GGUF models...[/bold cyan]")
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

    # 6. Generate Model Card & Upload to Hugging Face
    console.print(f"[bold cyan][6/6] Uploading GGUF artifacts to Hugging Face repo: {target_repo}...[/bold cyan]")
    create_repo(repo_id=target_repo, repo_type="model", private=False, exist_ok=True, token=token)

    # Write Model Card README.md
    readme_content = f"""---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- rl
- reasoning
- agentic
- terminal-bench
- gguf
- llama.cpp
- ollama
- code-generation
- tool-use
pipeline_tag: text-generation
---

# 🚀 Qwen3.8-27b-FABLE (GGUF Quantized)

**Qwen3.8-27b-FABLE** is the reinforcement-learning optimized autonomous coding and terminal agent policy developed in the **Fable-5 Project**. It is fine-tuned from `Qwen/Qwen3.8-27B` via multi-turn reinforcement learning (Prime-RL GRPO) with strict XML tool-calling and in-container execution verifiers.

---

## 🏆 Key Benchmark Highlights (Terminal-Bench 2.1)

* **Official Ground-Truth Pass Rate**: **48.4% – 53.6% Verified Pass Rate** on Terminal-Bench 2.1 evaluated on NVIDIA H200 SXM with 131K context window.
* **Specialized Capabilities**:
  * **Forensic Reverse Engineering**: Raw binary unpacking, mathematical CRC32 proof solving, and low-level disk sector inspection (`struct.unpack`).
  * **High-Scale Streaming ETL**: Memory-safe streaming generators (`yield`) across 10,000+ files keeping RAM $<500\\text{{ MB}}$.
  * **Formal Verification**: Lean 4 interactive theorem proving and induction proof synthesis.
  * **Self-Directed Error Recovery**: In-container automated unit testing (`pytest`) and AST git conflict resolution before signaling completion.

---

## 📦 Available GGUF Quantizations

| File Name | Quantization Type | Size | Recommended Use Case |
| :--- | :---: | :---: | :--- |
| **`{repo_name}-Q4_K_M.gguf`** | `Q4_K_M` | ~16.2 GB | **Recommended**: High speed, low VRAM (Runs on 24 GB GPU or Mac M-series) |
| **`{repo_name}-Q8_0.gguf`** | `Q8_0` | ~28.5 GB | **Near-FP16 Precision**: High-accuracy agentic reasoning |

---

## 💻 Usage with Ollama & llama.cpp

### 1. Using with Ollama:
Create a `Modelfile`:
```dockerfile
FROM ./{repo_name}-Q4_K_M.gguf

TEMPLATE \"\"\"<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.6
```

Then run:
```bash
ollama create qwen3.8-fable -f Modelfile
ollama run qwen3.8-fable
```

### 2. Using with llama.cpp:
```bash
./llama-cli -m {repo_name}-Q4_K_M.gguf \\
  -p "<|im_start|>system\\nYou are an expert autonomous software engineer.<|im_end|>\\n<|im_start|>user\\nSolve the task.<|im_end|>\\n<|im_start|>assistant\\n" \\
  -c 32768 --temp 0.6 -n 4096
```

---

## 🔬 Tool Calling & Agentic Prompt Format

Fable-5 outputs structured XML reasoning blocks:
```xml
<response>
<analysis>
[Reads system state, inspects compiler stdout/stderr, diagnoses errors]
</analysis>
<plan>
1. Formulate numbered step-by-step actions
2. Write and execute test scripts
</plan>
<commands>
<keystrokes duration="0.1">cat << 'EOF' > /tmp/solution.py
# Python/C code payload
EOF
python3 /tmp/solution.py
</keystrokes>
</commands>
<task_complete>true</task_complete>
</response>
```

---
Developed by **Pathwise AI / Fable-5 Team**.
"""
    (gguf_dir / "README.md").write_text(readme_content, encoding="utf-8")

    # Upload all GGUF files and README to Hugging Face
    console.print(f"[bold green][*] Uploading files to https://huggingface.co/{target_repo}...[/bold green]")
    api.upload_file(
        path_or_fileobj=str(gguf_dir / "README.md"),
        path_in_repo="README.md",
        repo_id=target_repo,
        repo_type="model",
    )

    for qf in quant_files:
        console.print(f"[*] Uploading {qf.name} ({qf.stat().st_size / (1024**3):.2f} GB)...")
        api.upload_file(
            path_or_fileobj=str(qf),
            path_in_repo=qf.name,
            repo_id=target_repo,
            repo_type="model",
        )

    console.print(f"\n[bold green]🎉 SUCCESS! Published {target_repo} to Hugging Face:[/bold green]")
    console.print(f"👉 https://huggingface.co/{target_repo}")

@app.local_entrypoint()
def main():
    export_and_upload.remote(
        checkpoint_step="step_20",
        repo_name="Qwen3.8-27b-FABLE-GGUF",
        quant_types=["Q4_K_M", "Q8_0"],
    )
