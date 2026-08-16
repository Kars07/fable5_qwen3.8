"""Rich terminal visualizer for inspecting individual Nemotron RL decision points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Ensure UTF-8 console output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))
from rl_dataset.parser import NemotronRLRecord, stream_nemotron_dataset


def visualize_record(rec: NemotronRLRecord, record_idx: int = 1, show_full_history: bool = False):
    """Print beautifully formatted view of a decision point."""
    print("=" * 80)
    print(f"📌 RECORD #{record_idx} | TASK: {rec.task_name} | UUID: {rec.uuid}")
    print("=" * 80)

    # 1. Metadata Block
    m = rec.metadata
    print(f"📁 Environment:    {m.harness}")
    print(f"🤖 Teacher Model:  {m.teacher_model}")
    print(f"🔄 Trajectory UID: {m.source_trajectory_uid}")
    print(f"📍 Pivot Turn:     Turn {m.pivot_agent_turn_index} of {m.total_source_agent_turns} (Total Prompt Messages: {len(rec.input_messages)})")
    print(f"🏁 Task Complete:  {rec.expected_answer.task_complete}")
    print("-" * 80)

    # 2. Initial Task Description
    if rec.input_messages:
        print("\n📝 [INITIAL TASK INSTRUCTION & PROMPT]:")
        first_content = rec.input_messages[0].content
        print(first_content[:1000] + ("..." if len(first_content) > 1000 else ""))

    # 3. Context History
    if show_full_history and len(rec.input_messages) > 1:
        print("\n📜 [PRIOR INTERACTION TRANSCRIPT]:")
        for i, msg in enumerate(rec.input_messages[1:], 2):
            role_label = f"[{msg.role.upper()} TURN #{i}]"
            content = msg.content
            snippet = content if len(content) < 800 else content[:400] + "\n...[truncated]...\n" + content[-400:]
            print(f"\n{role_label}:")
            print(snippet)
    elif len(rec.input_messages) > 1:
        print(f"\n📜 [PRIOR TURNS]: {len(rec.input_messages) - 1} prior terminal interaction messages (use --full to display).")

    # 4. Teacher Reasoning (CoT)
    ans = rec.expected_answer
    print("\n" + "=" * 80)
    print("🧠 [TEACHER CHAIN-OF-THOUGHT ANALYSIS (<think>)]:")
    print("=" * 80)
    print(ans.analysis if ans.analysis else "<None provided>")

    print("\n" + "=" * 80)
    print("📋 [TEACHER NEXT ACTION PLAN]:")
    print("=" * 80)
    print(ans.plan if ans.plan else "<None provided>")

    # 5. Executed Commands
    print("\n" + "=" * 80)
    print(f"💻 [EXECUTED TERMINAL COMMANDS ({len(ans.commands)} actions)]:")
    print("=" * 80)
    if not ans.commands:
        print("  <No terminal commands executed (Task Complete marked)>")
    else:
        for c_idx, cmd in enumerate(ans.commands, 1):
            print(f"  [{c_idx}] (duration: {cmd.duration:.1f}s)")
            for line in cmd.keystrokes.strip().split("\n"):
                print(f"      $ {line}")

    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize individual Nemotron RL decision points.")
    parser.add_argument("--dataset", type=str, default="rl_dataset/data/atcb_terminal_pivot_release_final_v2.jsonl", help="Path to JSONL dataset")
    parser.add_argument("--index", type=int, default=1, help="1-indexed record number to view")
    parser.add_argument("--uuid", type=str, default=None, help="Find and view specific record by UUID")
    parser.add_argument("--task", type=str, default=None, help="Find and view first record matching task name substring")
    parser.add_argument("--count", type=int, default=1, help="Number of records to display sequentially")
    parser.add_argument("--full", action="store_true", help="Display full prior message transcript history")

    args = parser.parse_args()

    shown = 0
    cur_idx = 0

    for rec in stream_nemotron_dataset(args.dataset):
        cur_idx += 1

        if args.uuid and rec.uuid != args.uuid:
            continue

        if args.task and args.task.lower() not in rec.task_name.lower():
            continue

        if not args.uuid and not args.task and cur_idx < args.index:
            continue

        visualize_record(rec, record_idx=cur_idx, show_full_history=args.full)
        shown += 1

        if shown >= args.count:
            break

    if shown == 0:
        print(f"[-] No matching records found for query.")


if __name__ == "__main__":
    main()
