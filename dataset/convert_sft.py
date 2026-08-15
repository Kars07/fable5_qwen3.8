"""Convert Glint-Research/Fable-5-traces into standard SFT formats for Qwen models."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.progress import track
from transformers import AutoTokenizer

from parser import (
    build_target_assistant_turn,
    extract_tools_from_records,
    parse_context_into_messages,
    parse_fable_record_to_messages,
)

# Ensure UTF-8 console output
sys.stdout.reconfigure(encoding="utf-8")
console = Console()

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert AI coding assistant. When solving tasks, think carefully about the architecture, "
    "diagnose root causes before editing, write clean maintainable code, and verify all changes through tests or commands."
)


def convert_step_level(
    records: List[Dict[str, Any]],
    output_path: Path,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    format_type: str = "qwen_native",  # 'qwen_native' or 'chatml_inline'
) -> int:
    """
    Convert each Fable-5 row into an independent SFT instance.
    Handles left-truncated contexts by ensuring a valid leading user query.
    """
    valid_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, r in enumerate(records):
            msgs = parse_context_into_messages(
                r.get("context", ""),
                fallback_user_prompt="Continue the coding task with the current state and files.",
            )

            # Insert system prompt if provided and not already present
            if system_prompt and (not msgs or msgs[0].get("role") != "system"):
                msgs.insert(0, {"role": "system", "content": system_prompt})

            target_turn = build_target_assistant_turn(r)

            if format_type == "chatml_inline":
                # Convert reasoning_content and tool_calls into inline string representation
                cot = target_turn.get("reasoning_content", "")
                content = target_turn.get("content", "")
                tool_calls = target_turn.get("tool_calls")

                asst_text = ""
                if cot:
                    asst_text += f"<think>\n{cot}\n</think>\n\n"

                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        asst_text += f"<tool_call>\n<function={fn.get('name')}>\n"
                        args = fn.get("arguments", {})
                        if isinstance(args, dict):
                            for k, v in args.items():
                                val_str = v if isinstance(v, str) else json.dumps(v)
                                asst_text += f"<parameter={k}>\n{val_str}\n</parameter>\n"
                        else:
                            asst_text += f"{args}\n"
                        asst_text += "</function>\n</tool_call>\n"
                elif content:
                    asst_text += content

                msgs.append({"role": "assistant", "content": asst_text.strip()})
            else:
                # Qwen native format with separate reasoning_content and structured tool_calls
                msgs.append(target_turn)

            sft_item = {
                "id": r.get("uid", f"fable_{idx}"),
                "session": r.get("session", "unknown"),
                "output_type": r.get("output_type", "unknown"),
                "origin": r.get("origin", "unknown"),
                "messages": msgs,
            }

            f.write(json.dumps(sft_item, ensure_ascii=False) + "\n")
            valid_count += 1

    return valid_count


def convert_session_level(
    records: List[Dict[str, Any]],
    output_path: Path,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
) -> int:
    """
    Reconstruct full multi-turn session trajectories from individual turns.
    Preserves complete dialogue context across the entire coding task.
    """
    session_map: Dict[str, List[Dict[str, Any]]] = {}

    for r in records:
        sess_id = r.get("session", "unknown")
        if sess_id not in session_map:
            session_map[sess_id] = []
        session_map[sess_id].append(r)

    # Sort each session's turns by uid index
    session_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for sess_id, sess_records in session_map.items():
            # Sort turns
            def get_turn_idx(rec):
                uid = rec.get("uid", "")
                if "#" in uid:
                    try:
                        return int(uid.split("#")[-1])
                    except ValueError:
                        pass
                return 0

            sess_records.sort(key=get_turn_idx)

            # Build full multi-turn conversation from the sequence of records
            full_msgs: List[Dict[str, Any]] = []
            if system_prompt:
                full_msgs.append({"role": "system", "content": system_prompt})

            # The first record's context contains the original user prompt
            initial_msgs = parse_context_into_messages(
                sess_records[0].get("context", ""),
                fallback_user_prompt="Start coding task.",
            )
            for m in initial_msgs:
                if m.get("role") != "system":
                    full_msgs.append(m)

            for rec in sess_records:
                target = build_target_assistant_turn(rec)
                full_msgs.append(target)

            session_item = {
                "id": sess_id,
                "session": sess_id,
                "num_turns": len(sess_records),
                "messages": full_msgs,
            }
            f.write(json.dumps(session_item, ensure_ascii=False) + "\n")
            session_count += 1

    return session_count


def validate_converted_dataset(
    file_path: Path,
    tokenizer_name: str = "Qwen/Qwen3.8-27B",
    max_check: int = 100,
) -> Tuple[int, int]:
    """Validate that converted SFT records pass tokenizer chat template rendering."""
    console.print(f"[*] Validating converted file '{file_path.name}' with tokenizer '{tokenizer_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    passed = 0
    failed = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_check and i >= max_check:
                break
            item = json.loads(line)
            msgs = item.get("messages", [])
            try:
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                passed += 1
            except Exception as e:
                failed += 1
                if failed <= 3:
                    console.print(f"[bold red]Validation error on item {item.get('id')}: {e}[/bold red]")

    return passed, failed


def main():
    parser = argparse.ArgumentParser(description="Convert Glint-Research/Fable-5-traces to SFT datasets.")
    parser.add_argument("--data-path", type=str, default="dataset/data/fable5_cot_merged.jsonl", help="Input dataset path")
    parser.add_argument("--output-dir", type=str, default="dataset/data/converted", help="Output directory")
    parser.add_argument("--format", type=str, choices=["qwen_native", "chatml_inline", "all"], default="all", help="SFT format type")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3.8-27B", help="Tokenizer for validation")

    args = parser.parse_args()
    in_path = Path(args.data_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        console.print(f"[bold red]Input file not found: {in_path}[/bold red]")
        sys.exit(1)

    with open(in_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    console.print(f"[bold green]Loaded {len(records)} records from {in_path.resolve()}[/bold green]")

    if args.format in ("qwen_native", "all"):
        step_out = out_dir / "fable5_sft_qwen_native.jsonl"
        console.print(f"[*] Converting to Qwen Native format -> {step_out}...")
        n = convert_step_level(records, step_out, format_type="qwen_native")
        console.print(f"[+] Wrote {n} records to {step_out.resolve()}")
        p, f = validate_converted_dataset(step_out, tokenizer_name=args.tokenizer, max_check=200)
        console.print(f"[bold cyan]Validation Result: {p} passed, {f} failed[/bold cyan]")

    if args.format in ("chatml_inline", "all"):
        inline_out = out_dir / "fable5_sft_chatml_inline.jsonl"
        console.print(f"[*] Converting to ChatML Inline format -> {inline_out}...")
        n = convert_step_level(records, inline_out, format_type="chatml_inline")
        console.print(f"[+] Wrote {n} records to {inline_out.resolve()}")
        p, f = validate_converted_dataset(inline_out, tokenizer_name=args.tokenizer, max_check=200)
        console.print(f"[bold cyan]Validation Result: {p} passed, {f} failed[/bold cyan]")

    # Session level
    sess_out = out_dir / "fable5_sft_sessions.jsonl"
    console.print(f"[*] Converting to full multi-turn sessions -> {sess_out}...")
    n_sess = convert_session_level(records, sess_out)
    console.print(f"[+] Wrote {n_sess} complete sessions to {sess_out.resolve()}")


if __name__ == "__main__":
    main()
