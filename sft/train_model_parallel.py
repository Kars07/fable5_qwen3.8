"""High-Performance Multi-GPU Model-Parallel SFT Trainer for Qwen on 4x/8x L4 GPUs with Chunked Cross Entropy & Validation."""

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
    AutoModelForCausalLM,
    AutoTokenizer,
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


def compute_chunked_loss(model, input_ids, attention_mask, labels, chunk_size=1024, ignore_index=-100):
    """Computes cross-entropy loss in chunks along sequence length to avoid allocating 7.5GB logits at once."""
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
    """Evaluates validation loss on validation set without gradients."""
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
                chunk_size=1024,
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
    config_path: str = "configs/sft_train.yaml",
    model_id_override: Optional[str] = None,
    use_lora: bool = True,
    lora_r: int = 64,
    lora_alpha: int = 128,
    smoke_test: bool = False,
    max_steps: Optional[int] = None,
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
    mode_str = "SMOKE TEST (10 Steps)" if smoke_test else "FULL SFT TRAINING"
    print(f"[*] Mode:                 {mode_str}", flush=True)
    print(f"[*] Parallelism:          Multi-GPU Pipeline Model Parallelism (device_map='auto')", flush=True)
    print(f"[*] Loss Strategy:        Chunked Cross-Entropy (Zero 152k-Vocab OOM)", flush=True)
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
        val_dataset = JsonlConversationDataset(cfg.eval_dataset_path, max_samples=50 if smoke_test else None)
        eval_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=1,
        )

    # 3. Model Loading with Auto Device Mapping across GPUs
    torch_dtype = torch.bfloat16 if cfg.dtype == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float16
    print(f"\n[*] Sharding model layers across all {num_gpus} GPUs (dtype: {torch_dtype})...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

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

    print("\n[*] GPU Memory Allocation after Model Sharding:", flush=True)
    for i in range(num_gpus):
        alloc = torch.cuda.memory_allocated(i) / (1024**3)
        res = torch.cuda.memory_reserved(i) / (1024**3)
        free = (torch.cuda.get_device_properties(i).total_memory / (1024**3)) - alloc
        print(f"    - GPU {i}: {alloc:.2f} GB allocated, {res:.2f} GB reserved ({free:.2f} GB FREE for activations)", flush=True)

    # 4. Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    total_target_steps = 10 if smoke_test else (len(train_dataloader) // cfg.gradient_accumulation_steps) * cfg.num_epochs
    if max_steps:
        total_target_steps = min(total_target_steps, max_steps)

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
    print(f"\n[*] Starting training loop (Target Steps: {total_target_steps})...\n", flush=True)

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
                chunk_size=1024,
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

                train_step_loss = accumulated_loss * cfg.gradient_accumulation_steps
                print(
                    f"Step [{global_step:2d}/{total_target_steps}] | "
                    f"Train Loss: {train_step_loss:.4f} | "
                    f"Grad Norm: {grad_norm:.3f} | "
                    f"Peak GPU VRAM: {peak_vram:.1f}GB | "
                    f"LR: {lr_cur:.2e} | "
                    f"Step Time: {step_duration:.2f}s",
                    flush=True,
                )
                accumulated_loss = 0.0

                # Periodic Validation Loss Check
                if eval_dataloader and (global_step % cfg.eval_steps == 0 or global_step == total_target_steps):
                    print(f"[*] Running Validation Evaluation on {len(eval_dataloader)} batches...", flush=True)
                    val_loss, val_ppl = run_evaluation(model, eval_dataloader)
                    print(f"    >>> EVALUATION: Val Loss = {val_loss:.4f} | Val Perplexity = {val_ppl:.2f}", flush=True)

                    if val_loss < best_val_loss and not smoke_test:
                        best_val_loss = val_loss
                        best_ckpt_path = os.path.join(cfg.output_dir, "best_checkpoint")
                        model.save_pretrained(best_ckpt_path)
                        tokenizer.save_pretrained(best_ckpt_path)
                        print(f"    >>> [BEST CHECKPOINT SAVED] Val Loss: {best_val_loss:.4f} -> {best_ckpt_path}", flush=True)

                if global_step >= total_target_steps:
                    break

        if global_step >= total_target_steps:
            break

    elapsed = time.time() - start_time
    print("=" * 70, flush=True)
    print(f"[+] TRAINING COMPLETED in {elapsed:.1f} seconds!", flush=True)
    for i in range(num_gpus):
        print(f"    - GPU {i} Peak VRAM: {torch.cuda.max_memory_allocated(i)/(1024**3):.2f} GB / {torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f} GB", flush=True)
    print("=" * 70, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Model Parallel SFT Training on Qwen with Validation.")
    parser.add_argument("--config", type=str, default="configs/sft_train.yaml", help="Path to SFT YAML config")
    parser.add_argument("--model-id", type=str, default=None, help="Override model path or ID")
    parser.add_argument("--lora", action="store_true", default=True, help="Enable LoRA")
    parser.add_argument("--smoke-test", action="store_true", default=False, help="Run fast 10-step smoke test")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum training steps")
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
