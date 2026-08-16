"""4-Bit NF4 Distributed Data Parallel (DDP) SFT Trainer for Qwen on 4x A10G GPUs."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
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


def setup_distributed():
    """Initialize distributed process group."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def compute_chunked_loss(model, input_ids, attention_mask, labels, chunk_size=256, ignore_index=-100):
    """Computes cross-entropy loss in small chunks (256 tokens) along sequence length to avoid allocating large logits."""
    # Unwrap DDP & PeftModel if needed
    raw_model = model.module if hasattr(model, "module") else model
    base_model = raw_model.get_base_model() if hasattr(raw_model, "get_base_model") else raw_model
    core = base_model.model if hasattr(base_model, "model") else base_model
    lm_head = base_model.lm_head if hasattr(base_model, "lm_head") else raw_model.lm_head

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


def run_evaluation(model, eval_dataloader, local_rank, max_eval_samples=25):
    """Evaluates validation loss across the holdout set without gradients."""
    model.eval()
    total_val_loss = 0.0
    val_steps = 0
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            if max_eval_samples and i >= max_eval_samples:
                break
            input_ids = batch["input_ids"].to(local_rank)
            attention_mask = batch["attention_mask"].to(local_rank)
            labels = batch["labels"].to(local_rank)

            loss = compute_chunked_loss(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                chunk_size=256,
                ignore_index=-100,
            )
            total_val_loss += loss.item()
            val_steps += 1

    model.train()
    if val_steps == 0:
        return 0.0, 1.0

    avg_loss = total_val_loss / val_steps

    # Aggregate across distributed ranks
    loss_tensor = torch.tensor([avg_loss], device=local_rank)
    if dist.is_initialized():
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    global_avg_loss = loss_tensor.item()
    ppl = math.exp(min(global_avg_loss, 20.0))
    return global_avg_loss, ppl


def train(
    config_path: str = "configs/sft_train_4bit_ddp.yaml",
    model_id_override: Optional[str] = None,
    use_lora: bool = True,
    lora_r: int = 64,
    lora_alpha: int = 128,
    smoke_test: bool = False,
    max_steps: Optional[int] = None,
    max_seq_length_override: Optional[int] = None,
):
    rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)

    cfg = SFTConfig.from_yaml(config_path)
    if model_id_override:
        cfg.model_id = model_id_override
        cfg.tokenizer_id = model_id_override
    if max_seq_length_override is not None:
        cfg.max_seq_length = max_seq_length_override

    set_seed(cfg.seed + rank)

    if is_main:
        print("=" * 70, flush=True)
        mode_str = "SMOKE TEST (10 Steps)" if smoke_test else "FULL 4-BIT SFT TRAINING (3 Epochs)"
        print(f"[*] Mode:                 {mode_str}", flush=True)
        print(f"[*] Parallelism:          Distributed Data Parallel (DDP) - 4x Concurrent Batches", flush=True)
        print(f"[*] Quantization:         4-Bit NF4 (Double Quantization)", flush=True)
        print(f"[*] Cluster World Size:   {world_size} GPUs (Local Rank: {local_rank})", flush=True)
        print(f"[*] Model Path/ID:        {cfg.model_id}", flush=True)
        print(f"[*] Train Dataset:        {cfg.dataset_name_or_path}", flush=True)
        print(f"[*] Val Dataset:          {cfg.eval_dataset_path}", flush=True)
        print(f"[*] Max Sequence Len:     {cfg.max_seq_length} tokens", flush=True)
        print(f"[*] LoRA (r={lora_r}, a={lora_alpha}):   {use_lora}", flush=True)
        print(f"[*] Batch (Per GPU):      {cfg.batch_size} (Effective Batch Size: {cfg.batch_size * world_size * cfg.gradient_accumulation_steps})", flush=True)
        print("=" * 70, flush=True)

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Datasets & Distributed Samplers
    dataset_sample_limit = 40 if smoke_test else None
    train_dataset = JsonlConversationDataset(cfg.dataset_name_or_path, max_samples=dataset_sample_limit)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.seed)

    collator = SFTDataCollator(
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        assistant_only_loss=cfg.assistant_only_loss,
        pad_to_multiple_of=8,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    eval_dataloader = None
    if cfg.eval_dataset_path and os.path.exists(cfg.eval_dataset_path):
        val_dataset = JsonlConversationDataset(cfg.eval_dataset_path, max_samples=40 if smoke_test else None)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        eval_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            sampler=val_sampler,
            collate_fn=collator,
            num_workers=1,
        )

    # 3. 4-bit BitsAndBytes Model on local GPU
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    if is_main:
        print(f"\n[*] Loading 4-bit model onto GPU {local_rank}...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map={"": local_rank},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

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
        if is_main:
            model.print_trainable_parameters()

    # Wrap with DDP
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    alloc = torch.cuda.memory_allocated(local_rank) / (1024**3)
    free = (torch.cuda.get_device_properties(local_rank).total_memory / (1024**3)) - alloc
    print(f"[+] GPU {local_rank}: 4-Bit model loaded ({alloc:.2f} GB allocated, {free:.2f} GB FREE for activations)", flush=True)

    # 4. Optimizer & Scheduler
    trainable_params = [p for p in ddp_model.parameters() if p.requires_grad]
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

    if is_main:
        print(f"\n[*] Starting 4-GPU DDP training loop (Target Steps: {total_target_steps})...\n", flush=True)

    start_time = time.time()
    for epoch in range(cfg.num_epochs):
        train_sampler.set_epoch(epoch)
        ddp_model.train()
        accumulated_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            step_start = time.time()

            input_ids = batch["input_ids"].to(local_rank)
            attention_mask = batch["attention_mask"].to(local_rank)
            labels = batch["labels"].to(local_rank)

            loss = compute_chunked_loss(
                model=ddp_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                chunk_size=256,
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
                peak_vram = torch.cuda.max_memory_allocated(local_rank) / (1024**3)

                train_step_loss = accumulated_loss / cfg.gradient_accumulation_steps

                if is_main:
                    print(
                        f"Step [{global_step:4d}/{total_target_steps}] | "
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
                    if is_main:
                        print(f"[*] Running Validation Evaluation across 4 GPUs...", flush=True)
                    val_loss, val_ppl = run_evaluation(ddp_model, eval_dataloader, local_rank)
                    if is_main:
                        print(f"    >>> [EVALUATION] Val Loss = {val_loss:.4f} | Val Perplexity = {val_ppl:.2f}", flush=True)

                        if val_loss < best_val_loss and not smoke_test:
                            best_val_loss = val_loss
                            best_ckpt_path = os.path.join(cfg.output_dir, "best_checkpoint")
                            raw_model = ddp_model.module if hasattr(ddp_model, "module") else ddp_model
                            raw_model.save_pretrained(best_ckpt_path)
                            tokenizer.save_pretrained(best_ckpt_path)
                            print(f"    >>> [BEST CHECKPOINT SAVED] Val Loss: {best_val_loss:.4f} -> {best_ckpt_path}", flush=True)

                if global_step >= total_target_steps:
                    break

        if global_step >= total_target_steps:
            break

    elapsed = time.time() - start_time
    if is_main:
        print("=" * 70, flush=True)
        print(f"[+] 4-BIT DDP TRAINING COMPLETED in {elapsed/60:.1f} minutes!", flush=True)
        if not smoke_test:
            final_ckpt_path = os.path.join(cfg.output_dir, "final_checkpoint")
            raw_model = ddp_model.module if hasattr(ddp_model, "module") else ddp_model
            raw_model.save_pretrained(final_ckpt_path)
            tokenizer.save_pretrained(final_ckpt_path)
            print(f"[+] Final Checkpoint Saved: {final_ckpt_path}", flush=True)
        for i in range(world_size):
            print(f"    - GPU {i} Peak VRAM: {torch.cuda.max_memory_allocated(i)/(1024**3):.2f} GB", flush=True)
        print("=" * 70, flush=True)

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="4-Bit NF4 Distributed Data Parallel (DDP) Training on Qwen.")
    parser.add_argument("--config", type=str, default="configs/sft_train_4bit_ddp.yaml", help="Path to SFT YAML config")
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
