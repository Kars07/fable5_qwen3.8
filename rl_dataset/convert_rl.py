"""Conversion utility for transforming Nemotron RL records into Qwen RL / GRPO / Verifier datasets."""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure UTF-8 console output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))
from rl_dataset.parser import (
    NemotronRLRecord,
    format_as_qwen_rl_prompt,
    format_expected_as_assistant_response,
    stream_nemotron_dataset,
)


def convert_record_to_qwen_rl(
    record: NemotronRLRecord,
) -> Dict[str, Any]:
    """Convert a Nemotron RL decision pivot into a Qwen RLVR rollout sample."""
    messages = format_as_qwen_rl_prompt(record, include_system_prompt=True)
    target_completion = format_expected_as_assistant_response(record)

    return {
        "uuid": record.uuid,
        "task_name": record.task_name,
        "harness": record.metadata.harness,
        "trajectory_uid": record.metadata.source_trajectory_uid,
        "turn_index": record.metadata.pivot_agent_turn_index,
        "total_turns": record.metadata.total_source_agent_turns,
        "messages": messages,
        "reference_completion": target_completion,
        "expected_answer_raw": record.expected_answer.model_dump(),
        "agent_ref": record.agent_ref,
    }


def convert_dataset(
    input_path: str = "rl_dataset/data/atcb_terminal_pivot_release_final_v2.jsonl",
    output_dir: str = "rl_dataset/data/converted",
    val_split_ratio: float = 0.05,
    seed: int = 42,
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Convert dataset with trajectory-aware train/val splitting."""
    in_file = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"[*] CONVERTING DATASET: {in_file.resolve()}")
    print(f"[*] Output Directory:  {out_dir.resolve()}")
    print(f"[*] Val Split Ratio:   {val_split_ratio * 100:.1f}% (Trajectory Grouped)")
    print("=" * 80)

    # 1. Group records by source_trajectory_uid to prevent train/val leakage
    trajectories: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    total_records = 0

    start_time = time.time()
    for rec in stream_nemotron_dataset(in_file, max_records=sample_limit):
        total_records += 1
        converted = convert_record_to_qwen_rl(rec)
        traj_id = rec.metadata.source_trajectory_uid or rec.uuid
        trajectories[traj_id].append(converted)

    print(f"[+] Loaded {total_records} records across {len(trajectories)} unique trajectories.")

    # 2. Shuffle trajectories deterministically
    traj_keys = sorted(list(trajectories.keys()))
    rng = random.Random(seed)
    rng.shuffle(traj_keys)

    # 3. Split by trajectory
    val_traj_count = max(1, int(len(traj_keys) * val_split_ratio))
    val_keys = set(traj_keys[:val_traj_count])
    train_keys = set(traj_keys[val_traj_count:])

    train_records = []
    val_records = []

    for k in traj_keys:
        recs = trajectories[k]
        if k in val_keys:
            val_records.extend(recs)
        else:
            train_records.extend(recs)

    # 4. Save converted files
    train_out = out_dir / "nemotron_terminal_rl_train.jsonl"
    val_out = out_dir / "nemotron_terminal_rl_val.jsonl"
    full_out = out_dir / "nemotron_terminal_rl_all.jsonl"

    print(f"[*] Saving {len(train_records)} train records to {train_out.resolve()}...")
    with open(train_out, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[*] Saving {len(val_records)} validation records to {val_out.resolve()}...")
    with open(val_out, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[*] Saving {total_records} full records to {full_out.resolve()}...")
    with open(full_out, "w", encoding="utf-8") as f:
        for r in train_records + val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("✨ CONVERSION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"[+] Total Processed:     {total_records:,}")
    print(f"[+] Train Set:           {len(train_records):,} records ({len(train_keys):,} trajectories) -> {train_out.name}")
    print(f"[+] Validation Set:      {len(val_records):,} records ({len(val_keys):,} trajectories) -> {val_out.name}")
    print(f"[+] Elapsed Time:        {elapsed:.2f}s")
    print("=" * 80 + "\n")

    return {
        "total_records": total_records,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "train_trajectories": len(train_keys),
        "val_trajectories": len(val_keys),
        "train_path": str(train_out),
        "val_path": str(val_out),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert Nemotron RL dataset to Qwen RL / GRPO format.")
    parser.add_argument("--input", type=str, default="rl_dataset/data/atcb_terminal_pivot_release_final_v2.jsonl", help="Input raw JSONL file")
    parser.add_argument("--output-dir", type=str, default="rl_dataset/data/converted", help="Output directory for converted files")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation trajectory split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for trajectory splitting")
    parser.add_argument("--sample-limit", type=int, default=None, help="Sample limit for testing")

    args = parser.parse_args()
    convert_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        val_split_ratio=args.val_ratio,
        seed=args.seed,
        sample_limit=args.sample_limit,
    )


if __name__ == "__main__":
    main()
