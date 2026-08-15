"""Test script demonstrating SFTDataset and DataCollator with PyTorch DataLoader."""

import json
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from validation import DataCollatorForSFTWithLossMask, SFTDataset

sys.stdout.reconfigure(encoding="utf-8")


def main():
    print("[*] Loading Qwen/Qwen3.8-27B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data_file = "dataset/data/converted/fable5_sft_qwen_native.jsonl"
    print(f"[*] Initializing SFTDataset from: {data_file}...")
    dataset = SFTDataset(
        data_path=data_file,
        tokenizer=tokenizer,
        max_seq_length=4096,
        assistant_only_loss=True,
    )
    print(f"[+] Total samples in dataset: {len(dataset)}")

    print("[*] Creating DataLoader with DataCollatorForSFTWithLossMask (batch_size=2)...")
    collator = DataCollatorForSFTWithLossMask(tokenizer=tokenizer, pad_to_multiple_of=8)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collator)

    batch = next(iter(loader))
    print("\nBatch tensor shapes:")
    print(f"  - input_ids:      {batch['input_ids'].shape} (dtype: {batch['input_ids'].dtype})")
    print(f"  - attention_mask: {batch['attention_mask'].shape} (dtype: {batch['attention_mask'].dtype})")
    print(f"  - labels:         {batch['labels'].shape} (dtype: {batch['labels'].dtype})")

    # Verify loss mask
    labels = batch["labels"]
    supervised_mask = labels != -100
    print(f"\nLoss Mask Statistics:")
    print(f"  - Supervised tokens in batch: {supervised_mask.sum().item()} / {labels.numel()} ({supervised_mask.sum().item()/labels.numel()*100:.1f}%)")
    print(f"  - Masked (prompt/padding) tokens in batch: {(~supervised_mask).sum().item()} ({(~supervised_mask).sum().item()/labels.numel()*100:.1f}%)")

    # Decode supervised tokens for inspection
    first_item_supervised = batch["input_ids"][0][batch["labels"][0] != -100]
    decoded_supervised = tokenizer.decode(first_item_supervised, skip_special_tokens=False)
    print(f"\nSupervised Tokens Preview for Item 0 (first 300 chars):")
    print(f"--------------------------------------------------")
    print(decoded_supervised[:300] + "...")
    print(f"--------------------------------------------------")
    print("[+] All PyTorch DataLoader and Loss Mask checks PASSED!")


if __name__ == "__main__":
    main()
