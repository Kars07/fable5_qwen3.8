"""Distributed SFT Training & Smoke Testing using Hugging Face Accelerate with FSDP Model Sharding & CPU Offload."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
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
    # 1. Initialize Accelerator with FSDP
    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    cfg = SFTConfig.from_yaml(config_path)
    if model_id_override:
        cfg.model_id = model_id_override
        cfg.tokenizer_id = model_id_override
    if max_seq_length_override is not None:
        cfg.max_seq_length = max_seq_length_override

    set_seed(cfg.seed)

    if is_main:
        print("=" * 70, flush=True)
        mode_str = "SMOKE TEST (10 Steps)" if smoke_test else "FULL SFT TRAINING"
        print(f"[*] Mode:                 {mode_str}", flush=True)
        print(f"[*] Parallelism:          Accelerate + FSDP (FULL_SHARD + Offload)", flush=True)
        print(f"[*] Cluster Devices:      {accelerator.num_processes} GPUs", flush=True)
        print(f"[*] Model Path/ID:        {cfg.model_id}", flush=True)
        print(f"[*] Dataset:              {cfg.dataset_name_or_path}", flush=True)
        print(f"[*] Max Sequence Len:     {cfg.max_seq_length} tokens", flush=True)
        print(f"[*] LoRA (r={lora_r}, a={lora_alpha}):   {use_lora}", flush=True)
        print(f"[*] Batch Size (Local):   {cfg.batch_size} (Effective: {cfg.batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps})", flush=True)
        print("=" * 70, flush=True)

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 3. Dataset & DataLoader
    dataset_sample_limit = 40 if smoke_test else None
    dataset = JsonlConversationDataset(cfg.dataset_name_or_path, max_samples=dataset_sample_limit)

    collator = SFTDataCollator(
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        assistant_only_loss=cfg.assistant_only_loss,
        pad_to_multiple_of=8,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=1,
        pin_memory=True,
    )

    # 4. Model Loading with RAM-efficient progressive sharding on CPU
    torch_dtype = torch.bfloat16 if cfg.dtype == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float16

    if is_main:
        print(f"[*] Loading model onto CPU RAM for FSDP sharding (dtype: {torch_dtype})...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
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
        model = model.to(torch_dtype)
        for param in model.parameters():
            if param.dtype != torch_dtype:
                param.data = param.data.to(torch_dtype)

        if is_main:
            model.print_trainable_parameters()

    # 5. Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    total_target_steps = 10 if smoke_test else (len(dataloader) // cfg.gradient_accumulation_steps) * cfg.num_epochs
    if max_steps:
        total_target_steps = min(total_target_steps, max_steps)

    warmup_steps = max(1, int(total_target_steps * cfg.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_target_steps)

    # 6. Accelerate Prepare
    if is_main:
        print("[*] Preparing model, optimizer, dataloader with Accelerate FSDP...", flush=True)

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    device_id = accelerator.local_process_index
    mem_alloc = torch.cuda.memory_allocated(device_id) / (1024**3)
    print(f"[+] GPU {device_id}: Model prepared with FSDP ({mem_alloc:.2f} GB VRAM allocated, {(24.0 - mem_alloc):.2f} GB free)", flush=True)

    # 7. Training Loop
    global_step = 0
    os.makedirs(cfg.output_dir, exist_ok=True)

    if is_main:
        print(f"[*] Starting execution: Target Steps = {total_target_steps}...", flush=True)

    start_time = time.time()
    for epoch in range(cfg.num_epochs):
        model.train()
        accumulated_loss = 0.0

        for step, batch in enumerate(dataloader):
            step_start = time.time()

            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                accelerator.backward(loss)

                accumulated_loss += loss.item()

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, cfg.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    step_duration = time.time() - step_start
                    mem_cur = torch.cuda.memory_allocated(device_id) / (1024**3)
                    mem_peak = torch.cuda.max_memory_allocated(device_id) / (1024**3)

                    if is_main:
                        lr_cur = scheduler.get_last_lr()[0]
                        print(
                            f"Step [{global_step}/{total_target_steps}] | "
                            f"Loss: {accumulated_loss / cfg.gradient_accumulation_steps:.4f} | "
                            f"VRAM: {mem_cur:.1f}GB (Peak: {mem_peak:.1f}GB) | "
                            f"LR: {lr_cur:.2e} | "
                            f"Time: {step_duration:.2f}s",
                            flush=True,
                        )
                    accumulated_loss = 0.0

                    if global_step >= total_target_steps:
                        break

        if global_step >= total_target_steps:
            break

    elapsed = time.time() - start_time
    if is_main:
        print("=" * 70, flush=True)
        print(f"[+] Smoke test completed successfully in {elapsed:.1f} seconds!", flush=True)
        print(f"[+] Peak VRAM usage per GPU: {torch.cuda.max_memory_allocated(0)/(1024**3):.2f} GB / 22.0 GB", flush=True)
        print("=" * 70, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Distributed SFT Training on Qwen with Accelerate FSDP.")
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
