"""4-Bit NF4 Quantized SFT Trainer for Qwen on Fable-5 Traces with Explicit Equal-Layer Sharding & Checkpoint Persistence."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

# Add parent and local dir to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from sft_lab.collator import SFTDataCollator
from sft_lab.config import SFTConfig
from sft_lab.seed import set_seed


class JsonlConversationDataset(Dataset):
    """Loads formatted JSONL messages for SFT training."""

    def __init__(self, data_path: str, max_samples: Optional[int] = None):
        self.data_path = data_path
        self.records = []
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                if line.strip():
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


def build_explicit_device_map(model_config, num_gpus: int) -> Dict[str, int]:
    """Evenly distributes transformer layers across available GPUs."""
    num_layers = getattr(model_config, "num_hidden_layers", 64)
    layers_per_gpu = max(1, math.ceil(num_layers / num_gpus))

    device_map = {}
    device_map["model.embed_tokens"] = 0
    for i in range(num_layers):
        gpu_id = min(i // layers_per_gpu, num_gpus - 1)
        device_map[f"model.layers.{i}"] = gpu_id
    device_map["model.norm"] = num_gpus - 1
    device_map["lm_head"] = num_gpus - 1
    return device_map


def compute_chunked_loss(model, input_ids, attention_mask, labels, chunk_size=128, ignore_index=-100):
    """Computes cross-entropy loss in small chunks (128 tokens) along sequence length to keep logits memory under 80MB."""
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    core = base_model.model if hasattr(base_model, "model") else base_model
    lm_head = base_model.lm_head if hasattr(base_model, "lm_head") else model.lm_head

    outputs = core(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    hidden_states = outputs[0]

    shift_hidden = hidden_states[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous().to(hidden_states.device)

    total_loss = 0.0
    total_valid = 0
    seq_len = shift_hidden.shape[1]

    for i in range(0, seq_len, chunk_size):
        chunk_hidden = shift_hidden[:, i : i + chunk_size, :]
        chunk_labels = shift_labels[:, i : i + chunk_size]

        valid_mask = (chunk_labels != ignore_index)
        n_valid = valid_mask.sum().item()
        if n_valid == 0:
            continue

        chunk_logits = lm_head(chunk_hidden).float()
        chunk_loss = nn.functional.cross_entropy(
            chunk_logits.view(-1, chunk_logits.shape[-1]),
            chunk_labels.view(-1),
            ignore_index=ignore_index,
            reduction="sum",
        )
        total_loss = total_loss + chunk_loss
        total_valid += n_valid

    if total_valid == 0:
        return torch.tensor(0.0, device=hidden_states.device, requires_grad=True)

    return total_loss / total_valid


def run_evaluation(model, eval_dataloader, max_eval_samples=40):
    """Evaluates validation loss on holdout set without gradients."""
    model.eval()
    total_val_loss = 0.0
    val_steps = 0
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            if max_eval_samples and i >= max_eval_samples:
                break
            input_ids = batch["input_ids"].to("cuda:0")
            attention_mask = batch["attention_mask"].to("cuda:0")
            labels = batch["labels"]

            loss = compute_chunked_loss(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                chunk_size=128,
                ignore_index=-100,
            )
            total_val_loss += loss.item()
            val_steps += 1

    model.train()
    if val_steps == 0:
        return 0.0, 1.0
    avg_loss = total_val_loss / val_steps
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


def train(
    config_path: str = "configs/sft_train_4bit.yaml",
    model_id_override: Optional[str] = None,
    use_lora: bool = True,
    lora_r: int = 64,
    lora_alpha: int = 128,
    smoke_test: bool = False,
    max_steps: Optional[int] = 100,
    max_seq_length_override: Optional[int] = None,
):
    cfg = SFTConfig.from_yaml(config_path)
    if model_id_override:
        cfg.model_id = model_id_override
        cfg.tokenizer_id = model_id_override
    if max_seq_length_override is not None:
        cfg.max_seq_length = max_seq_length_override

    set_seed(cfg.seed)

    num_gpus = torch.cuda.device_count()
    print("=" * 70, flush=True)
    mode_str = "SMOKE TEST (10 Steps)" if smoke_test else f"TARGETED SFT TRAINING ({max_steps or 100} Steps)"
    print(f"[*] Mode:                 {mode_str}", flush=True)
    print(f"[*] Quantization:         4-Bit NF4 (Double Quantization)", flush=True)
    print(f"[*] Sharding Scheme:      Explicit Equal Layer-Partition across {num_gpus} GPUs", flush=True)
    print(f"[*] Loss Strategy:        Ultra-Lean Chunked Cross-Entropy (chunk_size=128, <80MB logits)", flush=True)
    print(f"[*] Cluster Hardware:     {num_gpus}x GPUs", flush=True)
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"    - GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f} GB VRAM)", flush=True)
    print(f"[*] Model Path/ID:        {cfg.model_id}", flush=True)
    print(f"[*] Train Dataset:        {cfg.dataset_name_or_path}", flush=True)
    print(f"[*] Val Dataset:          {cfg.eval_dataset_path}", flush=True)
    print(f"[*] Max Sequence Len:     {cfg.max_seq_length} tokens", flush=True)
    print(f"[*] LoRA (r={lora_r}, a={lora_alpha}):   {use_lora}", flush=True)
    print(f"[*] Gradient Accum:       {cfg.gradient_accumulation_steps}", flush=True)
    print("=" * 70, flush=True)

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Datasets & DataLoaders
    dataset_sample_limit = 40 if smoke_test else None
    train_dataset = JsonlConversationDataset(cfg.dataset_name_or_path, max_samples=dataset_sample_limit)

    collator = SFTDataCollator(
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        assistant_only_loss=cfg.assistant_only_loss,
        pad_to_multiple_of=8,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    eval_dataloader = None
    if cfg.eval_dataset_path and os.path.exists(cfg.eval_dataset_path):
        val_dataset = JsonlConversationDataset(cfg.eval_dataset_path, max_samples=40 if smoke_test else None)
        eval_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=1,
        )

    # 3. 4-bit BitsAndBytes Configuration with Explicit Equal Layer Map
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model_config = AutoConfig.from_pretrained(cfg.model_id, trust_remote_code=True)
    explicit_device_map = build_explicit_device_map(model_config, num_gpus)

    print(f"\n[*] Loading model with explicit equal layer distribution across {num_gpus} GPUs...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map=explicit_device_map,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # Enable gradient checkpointing and input gradients
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    print("\n[*] GPU Memory Allocation after Explicit Equal Sharding (4-Bit):", flush=True)
    for i in range(num_gpus):
        alloc = torch.cuda.memory_allocated(i) / (1024**3)
        res = torch.cuda.memory_reserved(i) / (1024**3)
        free = (torch.cuda.get_device_properties(i).total_memory / (1024**3)) - alloc
        print(f"    - GPU {i}: {alloc:.2f} GB allocated ({free:.2f} GB FREE for activations)", flush=True)

    # 4. Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    total_target_steps = 10 if smoke_test else (max_steps if max_steps is not None else 100)

    warmup_steps = max(1, int(total_target_steps * cfg.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_target_steps,
    )

    # 5. Training Loop
    global_step = 0
    os.makedirs(cfg.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    log_file_path = os.path.join(cfg.output_dir, "training_metrics.jsonl")

    print(f"\n[*] Starting targeted training loop (Target Steps: {total_target_steps})...\n", flush=True)

    start_time = time.time()
    for epoch in range(cfg.num_epochs):
        model.train()
        accumulated_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            step_start = time.time()

            input_ids = batch["input_ids"].to("cuda:0")
            attention_mask = batch["attention_mask"].to("cuda:0")
            labels = batch["labels"]

            loss = compute_chunked_loss(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                chunk_size=128,
                ignore_index=-100,
            )
            loss_scaled = loss / cfg.gradient_accumulation_steps
            loss_scaled.backward()

            accumulated_loss += loss.item()

            if (step + 1) % cfg.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                step_duration = time.time() - step_start
                lr_cur = scheduler.get_last_lr()[0]
                peak_vram = max([torch.cuda.max_memory_allocated(i)/(1024**3) for i in range(num_gpus)])

                train_step_loss = accumulated_loss / cfg.gradient_accumulation_steps
                print(
                    f"Step [{global_step:3d}/{total_target_steps}] | "
                    f"Train Loss: {train_step_loss:.4f} | "
                    f"Grad Norm: {grad_norm:.3f} | "
                    f"Peak GPU VRAM: {peak_vram:.1f}GB | "
                    f"LR: {lr_cur:.2e} | "
                    f"Step Time: {step_duration:.2f}s",
                    flush=True,
                )
                accumulated_loss = 0.0

                # Clear fragmentation cache periodically
                if global_step % 5 == 0:
                    torch.cuda.empty_cache()

                # Periodic Validation Loss Check & Checkpoint Saving
                if eval_dataloader and (global_step % cfg.eval_steps == 0 or global_step == total_target_steps):
                    print(f"[*] Running Validation Evaluation on {len(eval_dataloader)} batches...", flush=True)
                    val_loss, val_ppl = run_evaluation(model, eval_dataloader)
                    print(f"    >>> [EVALUATION] Val Loss = {val_loss:.4f} | Val Perplexity = {val_ppl:.2f}", flush=True)

                    # Save step checkpoint
                    step_ckpt_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(step_ckpt_path)
                    tokenizer.save_pretrained(step_ckpt_path)
                    print(f"    >>> [CHECKPOINT SAVED] -> {step_ckpt_path}", flush=True)

                    # Save best checkpoint
                    if val_loss < best_val_loss and not smoke_test:
                        best_val_loss = val_loss
                        best_ckpt_path = os.path.join(cfg.output_dir, "best_checkpoint")
                        model.save_pretrained(best_ckpt_path)
                        tokenizer.save_pretrained(best_ckpt_path)
                        print(f"    >>> [BEST CHECKPOINT UPDATED] Val Loss: {best_val_loss:.4f} -> {best_ckpt_path}", flush=True)

                    # Instant Cloud Volume Commit
                    try:
                        import modal
                        modal.Volume.from_name("fable5-sft-checkpoints").commit()
                        print("    >>> [PERSISTED] Checkpoint committed to cloud volume.", flush=True)
                    except Exception as e:
                        pass

                    # Write log entry
                    log_entry = {
                        "step": global_step,
                        "train_loss": train_step_loss,
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                        "best_val_loss": best_val_loss,
                        "timestamp": time.time(),
                    }
                    with open(log_file_path, "a", encoding="utf-8") as lf:
                        lf.write(json.dumps(log_entry) + "\n")

                if global_step >= total_target_steps:
                    break

        if global_step >= total_target_steps:
            break

    elapsed = time.time() - start_time
    print("=" * 70, flush=True)
    print(f"[+] 4-BIT SFT TRAINING COMPLETED in {elapsed/60:.1f} minutes!", flush=True)
    if not smoke_test:
        final_ckpt_path = os.path.join(cfg.output_dir, "final_checkpoint")
        model.save_pretrained(final_ckpt_path)
        tokenizer.save_pretrained(final_ckpt_path)
        print(f"[+] Final Checkpoint Saved: {final_ckpt_path}", flush=True)
    for i in range(num_gpus):
        print(f"    - GPU {i} Peak VRAM: {torch.cuda.max_memory_allocated(i)/(1024**3):.2f} GB / {torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f} GB", flush=True)
    print("=" * 70, flush=True)


def main():
    parser = argparse.ArgumentParser(description="4-Bit NF4 Quantized SFT Training on Qwen with Validation.")
    parser.add_argument("--config", type=str, default="configs/sft_train_4bit.yaml", help="Path to SFT YAML config")
    parser.add_argument("--model-id", type=str, default=None, help="Override model path or ID")
    parser.add_argument("--lora", action="store_true", default=True, help="Enable LoRA")
    parser.add_argument("--smoke-test", action="store_true", default=False, help="Run fast 10-step smoke test")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum training steps")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Override maximum sequence length")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha")
    args = parser.parse_args()

    train(
        config_path=args.config,
        model_id_override=args.model_id,
        use_lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        smoke_test=args.smoke_test,
        max_steps=args.max_steps,
        max_seq_length_override=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
