"""Dataset validation, inspection, and PyTorch Dataset classes for SFT with Qwen models."""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from parser import (
    build_target_assistant_turn,
    parse_context_into_messages,
    parse_fable_record_to_messages,
)

# Ensure UTF-8 console output
sys.stdout.reconfigure(encoding="utf-8")

VALID_ROLES = {"system", "user", "assistant", "tool"}


@dataclass
class ValidationIssue:
    example_id: str
    issue_type: str
    message: str


def validate_conversation_structure(example: Dict[str, Any], index: int) -> List[ValidationIssue]:
    """Validate structure of a conversation record."""
    issues = []
    ex_id = str(example.get("id", example.get("uid", f"record_{index}")))

    # Check if this is a raw fable record
    if "messages" not in example and "context" in example:
        # Validate raw fable record structure
        if "cot" not in example or not example.get("cot"):
            issues.append(ValidationIssue(ex_id, "missing_cot", "Missing or empty 'cot' (Chain of Thought)"))
        if "output_type" not in example or example.get("output_type") not in {"tool_use", "text"}:
            issues.append(ValidationIssue(ex_id, "invalid_output_type", f"Invalid output_type: {example.get('output_type')}"))
        if "output" not in example:
            issues.append(ValidationIssue(ex_id, "missing_output", "Missing 'output' field"))
        return issues

    if "messages" not in example:
        issues.append(ValidationIssue(ex_id, "missing_field", "Missing 'messages' field"))
        return issues

    messages = example["messages"]
    if not isinstance(messages, list) or len(messages) == 0:
        issues.append(ValidationIssue(ex_id, "empty_conversation", "'messages' is empty or not a list"))
        return issues

    has_assistant = False
    prev_role = None

    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            issues.append(ValidationIssue(ex_id, "malformed_record", f"Message at index {msg_idx} is not a dict"))
            continue

        role = msg.get("role")
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")
        reasoning = msg.get("reasoning_content")

        if role not in VALID_ROLES:
            issues.append(ValidationIssue(ex_id, "unknown_role", f"Unknown role '{role}' at index {msg_idx}"))

        # In tool-calling / reasoning turns, content can be empty string if tool_calls or reasoning are present
        if role == "assistant":
            if not content and not tool_calls and not reasoning:
                issues.append(ValidationIssue(ex_id, "empty_assistant_turn", f"Assistant message at index {msg_idx} has no content, tool_calls, or reasoning"))
        else:
            if content is None or (isinstance(content, str) and len(content.strip()) == 0):
                issues.append(ValidationIssue(ex_id, "empty_message", f"Empty content for role '{role}' at index {msg_idx}"))

        if role == "system" and msg_idx != 0:
            issues.append(
                ValidationIssue(
                    ex_id,
                    "role_transition_error",
                    f"System role at index {msg_idx} (must be first)",
                )
            )

        if prev_role is not None and prev_role == role and role in {"user"}:
            # Consecutive user messages are flagged as warnings/anomalies
            issues.append(
                ValidationIssue(
                    ex_id,
                    "consecutive_user_messages",
                    f"Consecutive messages with role '{role}' at index {msg_idx}",
                )
            )

        if role == "assistant":
            has_assistant = True

        prev_role = role

    if not has_assistant:
        issues.append(ValidationIssue(ex_id, "no_assistant_response", "Conversation contains no assistant response"))

    return issues


def compute_stats(values: List[float | int]) -> Dict[str, float]:
    """Compute summary statistics for numeric list."""
    if not values:
        return {
            "min": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    arr = np.array(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def inspect_dataset_file(
    file_path: str,
    tokenizer_name_or_path: str = "Qwen/Qwen3.8-27B",
    max_seq_length: int = 4096,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Perform comprehensive validation and inspection of any dataset JSON/JSONL file."""
    fpath = Path(file_path)
    if not fpath.exists():
        alt = Path("dataset/data") / fpath.name
        if alt.exists():
            fpath = alt
        else:
            raise FileNotFoundError(f"File not found: {file_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct", trust_remote_code=True)

    records = []
    if str(fpath).endswith(".jsonl"):
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    else:
        with open(fpath, "r", encoding="utf-8") as f:
            records = json.load(f)

    if limit and limit > 0:
        records = records[:limit]

    total_examples = len(records)
    issues_list: List[ValidationIssue] = []

    turn_counts = []
    user_counts = []
    assistant_counts = []
    system_counts = []
    tool_counts = []
    char_counts = []
    token_counts = []
    prompt_token_counts = []
    assistant_token_counts = []
    assistant_ratios = []

    truncated_count = 0
    assistant_truncated_count = 0
    supervised_tokens_lost = 0
    zero_supervised_count = 0

    seen_ids = set()
    duplicate_ids = set()
    exact_duplicate_convs = 0
    conv_hashes = set()

    for i, ex in enumerate(records):
        ex_id = str(ex.get("id", ex.get("uid", f"record_{i}")))
        if ex_id in seen_ids:
            duplicate_ids.add(ex_id)
        seen_ids.add(ex_id)

        ex_issues = validate_conversation_structure(ex, i)
        issues_list.extend(ex_issues)

        # Normalize to messages format if raw fable record
        if "messages" not in ex and "context" in ex:
            messages = parse_fable_record_to_messages(ex)
        else:
            messages = ex.get("messages", [])

        if not messages:
            continue

        conv_str = json.dumps(messages, sort_keys=True)
        if conv_str in conv_hashes:
            exact_duplicate_convs += 1
        conv_hashes.add(conv_str)

        user_c = sum(1 for m in messages if m.get("role") == "user")
        asst_c = sum(1 for m in messages if m.get("role") == "assistant")
        sys_c = sum(1 for m in messages if m.get("role") == "system")
        tool_c = sum(1 for m in messages if m.get("role") == "tool")

        turn_counts.append(len(messages))
        user_counts.append(user_c)
        assistant_counts.append(asst_c)
        system_counts.append(sys_c)
        tool_counts.append(tool_c)

        total_chars = sum(len(str(m.get("content", ""))) + len(str(m.get("reasoning_content", ""))) for m in messages)
        char_counts.append(total_chars)

        try:
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
            total_toks = len(full_tokens)
            token_counts.append(total_toks)

            # Prompt tokens: everything before the final assistant response
            prompt_msgs = messages[:-1] if messages and messages[-1].get("role") == "assistant" else messages
            if prompt_msgs:
                prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
                prompt_toks = len(tokenizer.encode(prompt_text, add_special_tokens=False))
            else:
                prompt_toks = 0

            asst_toks = max(0, total_toks - prompt_toks)
            prompt_token_counts.append(prompt_toks)
            assistant_token_counts.append(asst_toks)

            ratio = asst_toks / total_toks if total_toks > 0 else 0.0
            assistant_ratios.append(ratio)

            if total_toks > max_seq_length:
                truncated_count += 1
                if prompt_toks >= max_seq_length:
                    assistant_truncated_count += 1
                    supervised_tokens_lost += asst_toks
                    zero_supervised_count += 1
                else:
                    supervised_in_window = max_seq_length - prompt_toks
                    lost = asst_toks - supervised_in_window
                    if lost > 0:
                        assistant_truncated_count += 1
                        supervised_tokens_lost += lost
            else:
                if asst_toks == 0:
                    zero_supervised_count += 1

        except Exception as err:
            issues_list.append(ValidationIssue(ex_id, "template_rendering_error", str(err)))

    issues_by_type: Dict[str, List[str]] = {}
    for issue in issues_list:
        if issue.issue_type not in issues_by_type:
            issues_by_type[issue.issue_type] = []
        issues_by_type[issue.issue_type].append(f"{issue.example_id}: {issue.message}")

    report = {
        "dataset_file": str(fpath.resolve()),
        "tokenizer_name": tokenizer_name_or_path,
        "max_seq_length": max_seq_length,
        "total_examples": total_examples,
        "validation_issues_count": len(issues_list),
        "issues_by_type": {k: len(v) for k, v in issues_by_type.items()},
        "issues_samples": {k: v[:5] for k, v in issues_by_type.items()},
        "duplicates": {
            "duplicate_ids_count": len(duplicate_ids),
            "exact_duplicate_conversations": exact_duplicate_convs,
        },
        "stats": {
            "turns": compute_stats(turn_counts),
            "user_messages": compute_stats(user_counts),
            "assistant_messages": compute_stats(assistant_counts),
            "system_messages": compute_stats(system_counts),
            "tool_messages": compute_stats(tool_counts),
            "characters": compute_stats(char_counts),
            "total_tokens": compute_stats(token_counts),
            "prompt_tokens": compute_stats(prompt_token_counts),
            "assistant_tokens": compute_stats(assistant_token_counts),
            "assistant_ratio": compute_stats(assistant_ratios),
        },
        "truncation": {
            "max_seq_length": max_seq_length,
            "truncated_examples": truncated_count,
            "truncated_percentage": (truncated_count / total_examples * 100.0) if total_examples > 0 else 0.0,
            "assistant_truncated_examples": assistant_truncated_count,
            "assistant_truncated_percentage": (assistant_truncated_count / total_examples * 100.0) if total_examples > 0 else 0.0,
            "supervised_tokens_lost": supervised_tokens_lost,
            "zero_supervised_examples": zero_supervised_count,
        },
    }
    return report


class SFTDataset(Dataset):
    """PyTorch Dataset for SFT conversations with tokenization and loss masking."""

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 4096,
        assistant_only_loss: bool = True,
        system_prompt: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.assistant_only_loss = assistant_only_loss
        self.system_prompt = system_prompt

        path = Path(data_path)
        if not path.exists():
            alt = Path("dataset/data") / path.name
            if alt.exists():
                path = alt
            else:
                raise FileNotFoundError(f"File not found: {data_path}")

        raw_records = []
        if str(path).endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        raw_records.append(json.loads(line))
        else:
            with open(path, "r", encoding="utf-8") as f:
                raw_records = json.load(f)

        self.records = []
        for i, ex in enumerate(raw_records):
            # Normalize record
            if "messages" not in ex and "context" in ex:
                msgs = parse_fable_record_to_messages(ex)
            else:
                msgs = ex.get("messages", [])

            if self.system_prompt and msgs and msgs[0].get("role") != "system":
                msgs.insert(0, {"role": "system", "content": self.system_prompt})

            if msgs:
                self.records.append({
                    "id": ex.get("id", ex.get("uid", f"sample_{i}")),
                    "messages": msgs,
                    "session": ex.get("session", "unknown"),
                    "output_type": ex.get("output_type", "unknown"),
                })

    def __len__(self) -> int:
        return len(self.records)

    def encode_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Encode messages into input_ids, attention_mask, and labels with loss masking."""
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False, max_length=self.max_seq_length, truncation=True)

        if not self.assistant_only_loss:
            labels = list(full_ids)
        else:
            # Mask out prompt tokens with -100
            prompt_msgs = messages[:-1] if messages and messages[-1].get("role") == "assistant" else messages
            if prompt_msgs:
                prompt_text = self.tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
                prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False, max_length=self.max_seq_length, truncation=True)
                prompt_len = min(len(prompt_ids), len(full_ids))
            else:
                prompt_len = 0

            labels = [-100] * prompt_len + full_ids[prompt_len:]

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        encoded = self.encode_messages(item["messages"])
        encoded["id"] = item["id"]
        return encoded


@dataclass
class DataCollatorForSFTWithLossMask:
    """Collator that pads input_ids, attention_mask, and labels."""
    tokenizer: PreTrainedTokenizer
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        first = features[0]
        batch = {}

        # Pad input_ids
        input_ids = [f["input_ids"] for f in features]
        max_len = max(len(ids) for ids in input_ids)
        if self.pad_to_multiple_of and max_len % self.pad_to_multiple_of != 0:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for f in features:
            ids = f["input_ids"]
            lbls = f["labels"]
            pad_len = max_len - len(ids)

            padded_input_ids.append(torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)]))
            padded_attention_mask.append(torch.cat([f["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
            padded_labels.append(torch.cat([lbls, torch.full((pad_len,), -100, dtype=torch.long)]))

        batch["input_ids"] = torch.stack(padded_input_ids)
        batch["attention_mask"] = torch.stack(padded_attention_mask)
        batch["labels"] = torch.stack(padded_labels)
        return batch


def main():
    parser = argparse.ArgumentParser(description="Validate and inspect dataset file for SFT.")
    parser.add_argument("--file", type=str, default="dataset/data/fable5_cot_merged.jsonl", help="Dataset file to validate")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3.8-27B", help="Tokenizer model name or path")
    parser.add_argument("--max-len", type=int, default=4096, help="Target maximum sequence length")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records to check")

    args = parser.parse_args()
    print(f"[*] Validating '{args.file}' with tokenizer '{args.tokenizer}' (max_seq_length={args.max_len})...")
    rep = inspect_dataset_file(args.file, tokenizer_name_or_path=args.tokenizer, max_seq_length=args.max_len, limit=args.limit)

    print("\n" + "=" * 70)
    print(f"VALIDATION REPORT: {Path(rep['dataset_file']).name}")
    print("=" * 70)
    print(f"Total Examples: {rep['total_examples']}")
    print(f"Validation Issues: {rep['validation_issues_count']}")
    if rep["issues_by_type"]:
        print("Issues Breakdown:")
        for k, count in rep["issues_by_type"].items():
            print(f"  - {k}: {count}")

    print("\nSequence Token Statistics:")
    ts = rep["stats"]["total_tokens"]
    print(f"  - Mean: {ts['mean']:.1f} ± {ts.get('std', 0):.1f}")
    print(f"  - Median: {ts['median']:.1f}")
    print(f"  - P90: {ts['p90']:.1f}, P95: {ts['p95']:.1f}, P99: {ts['p99']:.1f}")
    print(f"  - Min: {ts['min']:.0f}, Max: {ts['max']:.0f}")

    print(f"\nTruncation Analysis at max_seq_length={args.max_len}:")
    tr = rep["truncation"]
    print(f"  - Truncated Examples: {tr['truncated_examples']} ({tr['truncated_percentage']:.2f}%)")
    print(f"  - Examples with Assistant Loss Truncated: {tr['assistant_truncated_examples']}")
    print(f"  - Supervised Tokens Lost: {tr['supervised_tokens_lost']}")
    print(f"  - Zero Supervised Examples: {tr['zero_supervised_examples']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
