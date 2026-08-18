"""Head-to-head Inference Comparison: Base Qwen3.8-27B vs. Fable-5 RL Policy (Step 20/25)
Runs on Modal (kars07 profile) using 1x NVIDIA A100-80GB.
"""

import modal
import json
import time
from pathlib import Path

app = modal.App("fable5-head-to-head-eval")

# Modal environment setup
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.48.0",
        "accelerate>=1.0.0",
        "peft>=0.13.0",
        "safetensors>=0.4.5",
        "rich>=13.7.0",
        "sentencepiece>=0.2.0",
    )
)

volume_prime_outputs = modal.Volume.from_name("fable5-prime-rl-outputs")
volume_hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

CHALLENGE_PROMPT = """You are an elite systems engineer and autonomous agent operating in a Linux environment.
Solve the following difficult engineering challenge:

TASK SPECIFICATION:
Implement a high-performance, cache-aligned, lock-free Multi-Producer Multi-Consumer (MPMC) Ring Buffer in C++20 with:
1. Thread-safe concurrent enqueue and dequeue operations without mutexes (using atomic CAS and sequence counters).
2. Explicit cacheline alignment (64-byte padding) to prevent false sharing between head, tail, and cell buffers.
3. Power-of-two ring buffer capacity with bitwise masking for wrap-around.
4. Epoch-based reclamation or atomic generational indexing to prevent ABA hazards.
5. A comprehensive multi-threaded stress-test verifying 10,000,000 items enqueued/dequeued across 8 producer threads and 8 consumer threads with zero data loss, strictly preserving FIFO ordering per producer, and zero race conditions under ThreadSanitizer.

Provide your rigorous architectural analysis, mathematical invariant proof, and complete self-contained implementation with stress test.
"""

@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={
        "/workspace/prime_outputs": volume_prime_outputs,
        "/cache/huggingface": volume_hf_cache,
    },
)
def run_comparison(checkpoint_step: str = "step_25"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    console.print(Panel.fit("[bold cyan]⚔️ HEAD-TO-HEAD INFERENCE: BASE MODEL vs. FABLE-5 RL POLICY[/bold cyan]"))

    model_id = "Qwen/Qwen3.8-27B"
    token = os.environ.get("HF_TOKEN")

    console.print(f"[*] Loading Tokenizer ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=token,
        cache_dir="/cache/huggingface",
    )

    console.print(f"[*] Loading Base Model ({model_id}) in bfloat16 on 1x A100-80GB...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=token,
        cache_dir="/cache/huggingface",
    )
    base_model.eval()

    messages = [
        {"role": "system", "content": "You are a helpful, expert AI software engineering assistant."},
        {"role": "user", "content": CHALLENGE_PROMPT},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to("cuda")

    console.print(f"\n[bold yellow]{'='*30} 1. RUNNING BASE MODEL (Qwen3.8-27B) {'='*30}[/bold yellow]")
    console.print(f"[*] Input prompt tokens: {input_ids.shape[1]}")

    t0 = time.time()
    with torch.no_grad():
        base_outputs = base_model.generate(
            input_ids,
            max_new_tokens=2048,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    base_duration = time.time() - t0
    base_gen_tokens = base_outputs.shape[1] - input_ids.shape[1]
    base_response = tokenizer.decode(base_outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

    console.print(f"[+] Base Model completed in {base_duration:.2f}s ({base_gen_tokens / base_duration:.1f} tok/s, {base_gen_tokens} tokens)")

    # -------------------------------------------------------------
    # 2. Convert and Attach LoRA Policy Checkpoint
    # -------------------------------------------------------------
    console.print(f"\n[bold green]{'='*30} 2. ATTACHING FABLE-5 RL POLICY ({checkpoint_step}) {'='*30}[/bold green]")
    ckpt_dir = Path(f"/workspace/prime_outputs/prime-rl-run/checkpoints/{checkpoint_step}")
    if not ckpt_dir.exists():
        ckpt_dir = Path(f"/workspace/prime_outputs/checkpoints/{checkpoint_step}")

    console.print(f"[*] Checkpoint location: {ckpt_dir}")

    # Standardize PEFT adapter weights from DCP if needed
    peft_dir = Path(f"/tmp/peft_adapter_{checkpoint_step}")
    peft_dir.mkdir(parents=True, exist_ok=True)
    adapter_safetensor = peft_dir / "adapter_model.safetensors"

    if not adapter_safetensor.exists():
        if (ckpt_dir / "trainer" / "__0_0.distcp").exists():
            console.print(f"[*] Extracting PEFT weights from DCP checkpoint in {ckpt_dir}...")
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint import FileSystemReader
            from safetensors.torch import save_file

            reader = FileSystemReader(ckpt_dir / "trainer")
            meta = reader.read_metadata()
            lora_keys = [
                k for k in meta.state_dict_metadata.keys()
                if "lora" in k.lower()
                and not any(opt in k.lower() for opt in ["optimizer", "optimizers", "exp_avg", "step", "state."])
                and (k.startswith("app.model.") or not k.startswith("app."))
            ]
            state_dict = {k: torch.empty(meta.state_dict_metadata[k].size, dtype=meta.state_dict_metadata[k].properties.dtype) for k in lora_keys}
            dcp.load(state_dict=state_dict, storage_reader=reader)

            clean_state_dict = {}
            for k, v in state_dict.items():
                clean_k = k
                for prefix in ["app.model.model.language_model.", "app.model.language_model.", "app.model.", "language_model."]:
                    if clean_k.startswith(prefix):
                        clean_k = clean_k[len(prefix):]
                        break
                if clean_k.endswith(".lora_A.0") or clean_k.endswith(".lora_A.default.0"):
                    clean_k = clean_k.split(".lora_A")[0] + ".lora_A.weight"
                elif clean_k.endswith(".lora_B.0") or clean_k.endswith(".lora_B.default.0"):
                    clean_k = clean_k.split(".lora_B")[0] + ".lora_B.weight"
                elif clean_k.endswith(".0"):
                    clean_k = clean_k[:-2] + ".weight"

                if not clean_k.startswith("base_model.model."):
                    if clean_k.startswith("layers."):
                        clean_k = f"base_model.model.model.{clean_k}"
                    elif clean_k.startswith("model.layers."):
                        clean_k = f"base_model.model.{clean_k}"
                    else:
                        clean_k = f"base_model.model.model.{clean_k}"

                clean_state_dict[clean_k] = v.contiguous().to(torch.bfloat16)

            save_file(clean_state_dict, str(adapter_safetensor))
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
            (peft_dir / "adapter_config.json").write_text(json.dumps(lora_cfg, indent=2), encoding="utf-8")

    console.print(f"[*] Loading PEFT LoRA adapter onto base model...")
    rl_model = PeftModel.from_pretrained(base_model, str(peft_dir))
    rl_model.eval()

    console.print(f"[*] Running Fable-5 RL Policy on the exact same challenge prompt...")
    t0 = time.time()
    with torch.no_grad():
        rl_outputs = rl_model.generate(
            input_ids,
            max_new_tokens=2048,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    rl_duration = time.time() - t0
    rl_gen_tokens = rl_outputs.shape[1] - input_ids.shape[1]
    rl_response = tokenizer.decode(rl_outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

    console.print(f"[+] RL Policy completed in {rl_duration:.2f}s ({rl_gen_tokens / rl_duration:.1f} tok/s, {rl_gen_tokens} tokens)")

    # -------------------------------------------------------------
    # Output Side-by-Side Comparison
    # -------------------------------------------------------------
    console.print("\n" + "="*80)
    console.print("[bold cyan]=== 📝 SIDE-BY-SIDE INFERENCE OUTPUTS ===[/bold cyan]")
    console.print("="*80)

    console.print(Panel(base_response, title="[bold yellow]1. BASE MODEL RESPONSE (Qwen3.8-27B)[/bold yellow]", expand=True))
    console.print(Panel(rl_response, title=f"[bold green]2. FABLE-5 RL POLICY RESPONSE ({checkpoint_step})[/bold green]", expand=True))

    comp_table = Table(title="Performance & Structural Metrics")
    comp_table.add_column("Metric", style="bold")
    comp_table.add_column("Base Qwen3.8-27B", style="yellow")
    comp_table.add_column(f"Fable-5 Policy ({checkpoint_step})", style="green")

    comp_table.add_row("Generated Tokens", str(base_gen_tokens), str(rl_gen_tokens))
    comp_table.add_row("Generation Time", f"{base_duration:.2f}s", f"{rl_duration:.2f}s")
    comp_table.add_row("Tokens / Second", f"{base_gen_tokens / base_duration:.1f}", f"{rl_gen_tokens / rl_duration:.1f}")

    console.print(comp_table)

    return {
        "challenge_prompt": CHALLENGE_PROMPT,
        "base_response": base_response,
        "rl_response": rl_response,
        "base_tokens": base_gen_tokens,
        "rl_tokens": rl_gen_tokens,
    }

@app.local_entrypoint()
def main(checkpoint: str = "step_25"):
    """Run head-to-head inference evaluation."""
    run_comparison.remote(checkpoint_step=checkpoint)
