"""Distributed SFT Training & Smoke Testing with PyTorch FSDP Model Sharding for Qwen on 4x L4 GPUs."""

import argparse
import functools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_cosine_schedule_with_warmup,
)

# Add parent and local dir to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from sft_lab.checkpointing import save_checkpoint
from sft_lab.collator import SFTDataCollator
from sft_lab.config import SFTConfig
from sft_lab.seed import generate_environment_report, set_seed


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


def get_qwen_decoder_layer_class():
    """Identify the transformer block class for Qwen architecture auto-wrapping in FSDP."""
    classes = set()
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
        classes.add(Qwen3_5DecoderLayer)
    except Exception:
        pass
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        classes.add(Qwen2DecoderLayer)
    except Exception:
        pass
    return classes


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
        mode_str = "SMOKE TEST (10 Steps)" if smoke_test else "FULL SFT TRAINING"
        print(f"[*] Mode:                 {mode_str}", flush=True)
        print(f"[*] Parallelism:          PyTorch FSDP (FULL_SHARD Model Sharding)", flush=True)
        print(f"[*] Cluster:              {world_size} GPUs (Local Rank: {local_rank})", flush=True)
        print(f"[*] Model Path/ID:        {cfg.model_id}", flush=True)
        print(f"[*] Dataset:              {cfg.dataset_name_or_path}", flush=True)
        print(f"[*] Max Sequence Len:     {cfg.max_seq_length} tokens", flush=True)
        print(f"[*] LoRA (r={lora_r}, a={lora_alpha}):   {use_lora}", flush=True)
        print(f"[*] Batch Size (Local):   {cfg.batch_size} (Effective: {cfg.batch_size * world_size * cfg.gradient_accumulation_steps})", flush=True)
        print("=" * 70, flush=True)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Dataset and Distributed Sampler
    dataset_sample_limit = 40 if smoke_test else None
    dataset = JsonlConversationDataset(cfg.dataset_name_or_path, max_samples=dataset_sample_limit)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.seed)
    
    collator = SFTDataCollator(
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        assistant_only_loss=cfg.assistant_only_loss,
        pad_to_multiple_of=8,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=1,
        pin_memory=True,
    )

    # 3. Model Loading with FSDP Model Sharding
    torch_dtype = torch.bfloat16 if cfg.dtype == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float16
    
    if is_main:
        print(f"[*] Loading model on CPU RAM before FSDP sharding (dtype: {torch_dtype})...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    from peft import LoraConfig, TaskType, get_peft_model
    if use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
        # Ensure all LoRA parameters strictly match torch_dtype (bfloat16) for uniform FSDP flattening
        model = model.to(torch_dtype)
        for param in model.parameters():
            if param.dtype != torch_dtype:
                param.data = param.data.to(torch_dtype)
        if is_main:
            model.print_trainable_parameters()

    # 4. Wrap with PyTorch FSDP FULL_SHARD
    if is_main:
        print("[*] Sharding model across 4 GPUs with FSDP FULL_SHARD...", flush=True)

    auto_wrap_classes = get_qwen_decoder_layer_class()
    wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=auto_wrap_classes) if auto_wrap_classes else None
    mp_policy = MixedPrecision(
        param_dtype=torch_dtype,
        reduce_dtype=torch_dtype,
        buffer_dtype=torch_dtype,
    )

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
    )

    mem_alloc = torch.cuda.memory_allocated(local_rank) / (1024**3)
    mem_res = torch.cuda.memory_reserved(local_rank) / (1024**3)
    print(f"[+] GPU {local_rank}: Model sharded to {mem_alloc:.2f} GB VRAM (Free: {(24.0 - mem_alloc):.2f} GB)", flush=True)

    # 5. Optimizer & Scheduler
    trainable_params = [p for p in fsdp_model.parameters() if p.requires_grad]
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

    # 6. Training Loop
    global_step = 0
    os.makedirs(cfg.output_dir, exist_ok=True)

    if is_main:
        print(f"[*] Starting execution: Target Steps = {total_target_steps}...", flush=True)

    start_time = time.time()
    for epoch in range(cfg.num_epochs):
        sampler.set_epoch(epoch)
        fsdp_model.train()
        accumulated_loss = 0.0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(local_rank)
            labels = batch["labels"].to(local_rank)
            attention_mask = batch["attention_mask"].to(local_rank)

            step_start = time.time()
            outputs = fsdp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / cfg.gradient_accumulation_steps
            loss.backward()

            accumulated_loss += loss.item()

            if (step + 1) % cfg.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                grad_norm = fsdp_model.clip_grad_norm_(cfg.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                step_duration = time.time() - step_start
                mem_cur = torch.cuda.memory_allocated(local_rank) / (1024**3)
                mem_peak = torch.cuda.max_memory_allocated(local_rank) / (1024**3)

                if is_main:
                    lr_cur = scheduler.get_last_lr()[0]
                    print(
                        f"Step [{global_step}/{total_target_steps}] | "
                        f"Loss: {accumulated_loss * cfg.gradient_accumulation_steps:.4f} | "
                        f"Grad Norm: {grad_norm:.3f} | "
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

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Distributed SFT Training / Smoke Testing on Qwen with PyTorch FSDP.")
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
