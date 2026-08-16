"""Comprehensive inspection tool for Glint-Research/Fable-5-traces dataset with Qwen tokenizer support."""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.syntax import Syntax
from rich.table import Table
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


def compute_distribution_stats(values: List[float | int]) -> Dict[str, float]:
    """Compute detailed quantile and summary statistics."""
    if not values:
        return {k: 0.0 for k in ["count", "min", "p10", "p25", "p50_median", "p75", "p90", "p95", "p99", "max", "mean", "std"]}
    arr = np.array(values, dtype=float)
    return {
        "count": len(arr),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50_median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def load_records(data_path: str) -> List[Dict[str, Any]]:
    """Load JSONL or JSON dataset records."""
    path = Path(data_path)
    if not path.exists():
        # Fallback to local default path if user passed a relative name
        alt_path = Path("dataset/data") / path.name
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Dataset file not found at: {data_path}")

    records = []
    if str(path).endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
    return records


def view_sample(record: Dict[str, Any], index: int, tokenizer: Optional[AutoTokenizer] = None):
    """Render a single sample in a clean, colorized terminal UI."""
    uid = record.get("uid", f"sample_{index}")
    out_type = record.get("output_type", "unknown")
    origin = record.get("origin", "unknown")
    session = record.get("session", "unknown")

    console.print()
    console.rule(f"[bold cyan]Inspection: Record #{index} | UID: {uid} | Type: {out_type} | Session: {session[:8]}... | Origin: {origin}[/bold cyan]")

    # Context Messages
    raw_context = record.get("context", "")
    parsed_msgs = parse_context_into_messages(raw_context)

    console.print(f"[bold yellow]=== Conversation Context ({len(parsed_msgs)} turns) ===[/bold yellow]")
    for m_idx, m in enumerate(parsed_msgs):
        role = m.get("role", "unknown")
        content = m.get("content", "")
        if role == "user":
            console.print(Panel(content, title=f"[bold green]Turn {m_idx}: USER[/bold green]", border_style="green"))
        elif role == "system":
            console.print(Panel(content, title=f"[bold cyan]Turn {m_idx}: SYSTEM[/bold cyan]", border_style="cyan"))
        elif role == "tool":
            preview = content if len(content) < 400 else content[:400] + f"\n... [+{len(content)-400} chars truncated]"
            console.print(Panel(preview, title=f"[bold magenta]Turn {m_idx}: TOOL RESULT[/bold magenta]", border_style="magenta"))
        elif role == "assistant":
            if m.get("tool_calls"):
                tc = m["tool_calls"][0]["function"]
                call_str = f"Tool: {tc['name']}\nArguments:\n{json.dumps(tc['arguments'], indent=2)}"
                console.print(Panel(call_str, title=f"[bold blue]Turn {m_idx}: ASSISTANT (Tool Call)[/bold blue]", border_style="blue"))
            else:
                console.print(Panel(content, title=f"[bold blue]Turn {m_idx}: ASSISTANT[/bold blue]", border_style="blue"))

    # CoT (Reasoning)
    cot = record.get("cot", "")
    if cot:
        console.print(Panel(cot, title="[bold purple]=== Chain of Thought (<think>) ===[/bold purple]", border_style="purple"))
    else:
        console.print("[yellow][!] No Chain of Thought (<think>) present in this record.[/yellow]")

    # Target Action / Output
    out_data = record.get("output", {})
    if out_type == "tool_use":
        out_str = f"Tool: {out_data.get('tool')}\nArguments:\n{json.dumps(out_data.get('input', {}), indent=2)}"
        console.print(Panel(out_str, title="[bold red]=== Target Action: TOOL USE ===[/bold red]", border_style="red"))
    else:
        console.print(Panel(out_data.get("text", str(out_data)), title="[bold red]=== Target Action: FINAL TEXT ===[/bold red]", border_style="red"))

    # Tokenizer Rendering Preview
    if tokenizer is not None:
        all_msgs = parse_fable_record_to_messages(record)
        try:
            rendered = tokenizer.apply_chat_template(all_msgs, tokenize=False, add_generation_prompt=False)
            toks = tokenizer.encode(rendered, add_special_tokens=False)
            console.print(f"[bold cyan]Tokenizer Applied Template Preview ({len(toks)} total tokens):[/bold cyan]")
            console.print(Panel(rendered[:600] + ("\n... [truncated] ...\n" + rendered[-600:] if len(rendered) > 1200 else ""), border_style="dim white"))
        except Exception as e:
            console.print(f"[bold red]Tokenizer render error: {e}[/bold red]")


def inspect_dataset(
    data_path: str,
    tokenizer_name: str = "Qwen/Qwen3.8-27B",
    sample_limit: Optional[int] = None,
    export_dir: str = "dataset/reports",
) -> Dict[str, Any]:
    """Perform full dataset profiling, token analysis, tool statistics, and anomaly detection."""
    records = load_records(data_path)
    if sample_limit and sample_limit > 0:
        records = records[:sample_limit]

    total_records = len(records)
    console.print(f"[bold green][*] Profiling {total_records} records using tokenizer '{tokenizer_name}'...[/bold green]")

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception as e:
        console.print(f"[bold red]Failed to load tokenizer '{tokenizer_name}': {e}. Falling back to Qwen/Qwen2.5-32B-Instruct[/bold red]")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct", trust_remote_code=True)

    # Metric collections
    output_types = Counter()
    tool_counts = Counter()
    origins = Counter()
    models = Counter()
    sessions = Counter()

    # Length collections
    char_context = []
    char_cot = []
    char_comp = []
    
    tokens_context = []
    tokens_cot = []
    tokens_target_action = []
    tokens_total_sequence = []
    cot_ratios = []

    # Quality and anomaly tracking
    missing_cot_count = 0
    empty_context_count = 0
    missing_user_in_context = 0
    json_parse_errors_in_tools = 0
    template_errors = []

    # Target context windows to evaluate
    context_windows = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
    window_truncations = {w: 0 for w in context_windows}

    for idx, r in enumerate(track(records, description="Inspecting dataset records...")):
        out_type = r.get("output_type", "unknown")
        output_types[out_type] += 1
        origins[r.get("origin", "unknown")] += 1
        models[r.get("model", "unknown")] += 1
        sessions[r.get("session", "unknown")] += 1

        out_data = r.get("output", {})
        if out_type == "tool_use":
            tool_name = out_data.get("tool", "unknown")
            tool_counts[tool_name] += 1
            if isinstance(out_data.get("input"), str):
                try:
                    json.loads(out_data["input"])
                except Exception:
                    json_parse_errors_in_tools += 1

        ctx = r.get("context", "")
        cot = r.get("cot", "")
        comp = r.get("completion", "")

        if not ctx:
            empty_context_count += 1
        if not cot:
            missing_cot_count += 1

        # Check raw context headers
        parsed_ctx = parse_context_into_messages(ctx, fallback_user_prompt=None)
        if not any(m.get("role") == "user" for m in parsed_ctx):
            missing_user_in_context += 1

        char_context.append(len(ctx))
        char_cot.append(len(cot))
        char_comp.append(len(comp))

        # Tokenization analysis
        t_ctx = len(tokenizer.encode(ctx, add_special_tokens=False))
        t_cot = len(tokenizer.encode(cot, add_special_tokens=False)) if cot else 0
        
        # Target action tokens (excluding reasoning)
        if out_type == "tool_use":
            target_str = f"<tool_call>\n<function={out_data.get('tool')}>\n{json.dumps(out_data.get('input', {}))}\n</function>\n</tool_call>"
        else:
            target_str = str(out_data.get("text", out_data))
        t_action = len(tokenizer.encode(target_str, add_special_tokens=False))

        # Test full chat template rendering
        try:
            full_msgs = parse_fable_record_to_messages(r)
            rendered_prompt = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
            t_total = len(tokenizer.encode(rendered_prompt, add_special_tokens=False))
        except Exception as err:
            template_errors.append({"index": idx, "uid": r.get("uid"), "error": str(err)})
            t_total = t_ctx + t_cot + t_action + 50  # approximate fallback

        tokens_context.append(t_ctx)
        tokens_cot.append(t_cot)
        tokens_target_action.append(t_action)
        tokens_total_sequence.append(t_total)

        ratio = t_cot / (t_cot + t_action) if (t_cot + t_action) > 0 else 0.0
        cot_ratios.append(ratio)

        for w in context_windows:
            if t_total > w:
                window_truncations[w] += 1

    # Compile report structure
    report = {
        "dataset_path": str(Path(data_path).resolve()),
        "tokenizer": tokenizer_name,
        "total_records": total_records,
        "unique_sessions": len(sessions),
        "output_type_distribution": dict(output_types),
        "tool_usage_distribution": dict(tool_counts),
        "origin_distribution": dict(origins),
        "model_distribution": dict(models),
        "session_turn_stats": compute_distribution_stats(list(sessions.values())),
        "character_stats": {
            "context": compute_distribution_stats(char_context),
            "cot_reasoning": compute_distribution_stats(char_cot),
            "completion": compute_distribution_stats(char_comp),
        },
        "token_stats": {
            "context_prompt": compute_distribution_stats(tokens_context),
            "cot_reasoning": compute_distribution_stats(tokens_cot),
            "target_action": compute_distribution_stats(tokens_target_action),
            "total_sequence": compute_distribution_stats(tokens_total_sequence),
            "cot_ratio": compute_distribution_stats(cot_ratios),
        },
        "context_window_budget": {
            f"{w}_tokens": {
                "truncated_count": window_truncations[w],
                "fit_count": total_records - window_truncations[w],
                "fit_percentage": round((1 - (window_truncations[w] / total_records)) * 100, 2) if total_records else 0,
            }
            for w in context_windows
        },
        "anomalies": {
            "missing_cot_count": missing_cot_count,
            "empty_context_count": empty_context_count,
            "missing_user_in_raw_context": missing_user_in_context,
            "json_parse_errors_in_tools": json_parse_errors_in_tools,
            "template_rendering_errors_count": len(template_errors),
            "template_error_samples": template_errors[:10],
        },
    }

    # Print Rich CLI summary
    _print_cli_tables(report)

    # Save reports
    rep_dir = Path(export_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    json_path = rep_dir / "inspection_report.json"
    md_path = rep_dir / "inspection_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _write_markdown_report(report, md_path)
    console.print(f"[bold green][+] Inspection reports saved to:[/bold green]\n  - JSON: {json_path.resolve()}\n  - Markdown: {md_path.resolve()}")

    return report


def _print_cli_tables(report: Dict[str, Any]):
    """Print beautifully styled summary tables in the terminal."""
    console.print()
    console.rule("[bold cyan]DATASET OVERVIEW & METRICS[/bold cyan]")

    # Overview Table
    ov_table = Table(title="Dataset Summary", show_header=True, header_style="bold magenta")
    ov_table.add_column("Metric", style="cyan")
    ov_table.add_column("Value", style="bold white")
    ov_table.add_row("Total Records", str(report["total_records"]))
    ov_table.add_row("Unique Sessions", str(report["unique_sessions"]))
    ov_table.add_row("Tokenizer Tested", report["tokenizer"])
    ov_table.add_row("Tool Call Rows", f"{report['output_type_distribution'].get('tool_use', 0)} ({report['output_type_distribution'].get('tool_use', 0)/report['total_records']*100:.1f}%)")
    ov_table.add_row("Final Text Rows", f"{report['output_type_distribution'].get('text', 0)} ({report['output_type_distribution'].get('text', 0)/report['total_records']*100:.1f}%)")
    ov_table.add_row("Left-Truncated Contexts (Missing Initial Prompt)", str(report["anomalies"]["missing_user_in_raw_context"]))
    console.print(ov_table)

    # Token Distribution Table
    tok_table = Table(title=f"Token Quantiles ({report['tokenizer']})", show_header=True, header_style="bold cyan")
    tok_table.add_column("Field", style="bold yellow")
    tok_table.add_column("Min", justify="right")
    tok_table.add_column("P25", justify="right")
    tok_table.add_column("Median (P50)", justify="right", style="bold green")
    tok_table.add_column("P75", justify="right")
    tok_table.add_column("P90", justify="right")
    tok_table.add_column("P95", justify="right")
    tok_table.add_column("P99", justify="right", style="bold red")
    tok_table.add_column("Max", justify="right")
    tok_table.add_column("Mean ± Std", justify="right")

    for key, name in [
        ("context_prompt", "Context (Prompt)"),
        ("cot_reasoning", "CoT (<think>)"),
        ("target_action", "Action / Target"),
        ("total_sequence", "Total Sequence"),
    ]:
        s = report["token_stats"][key]
        tok_table.add_row(
            name,
            f"{s['min']:.0f}",
            f"{s['p25']:.0f}",
            f"{s['p50_median']:.0f}",
            f"{s['p75']:.0f}",
            f"{s['p90']:.0f}",
            f"{s['p95']:.0f}",
            f"{s['p99']:.0f}",
            f"{s['max']:.0f}",
            f"{s['mean']:.0f} ± {s['std']:.0f}",
        )
    console.print(tok_table)

    # Context Budget Table
    ctx_table = Table(title="Context Window Budget Matrix", show_header=True, header_style="bold blue")
    ctx_table.add_column("Max Window Size", style="cyan")
    ctx_table.add_column("Fit Percentage", justify="right", style="bold green")
    ctx_table.add_column("Fit Records", justify="right")
    ctx_table.add_column("Truncated Records", justify="right", style="bold red")

    for k, v in report["context_window_budget"].items():
        ctx_table.add_row(
            k.replace("_", " ").upper(),
            f"{v['fit_percentage']}%",
            str(v["fit_count"]),
            str(v["truncated_count"]),
        )
    console.print(ctx_table)

    # Tool Usage Table
    tool_table = Table(title="Tool Invocation Breakdown", show_header=True, header_style="bold green")
    tool_table.add_column("Tool Name", style="bold white")
    tool_table.add_column("Count", justify="right")
    tool_table.add_column("Share (%)", justify="right", style="cyan")

    total_tool_calls = sum(report["tool_usage_distribution"].values())
    for t_name, count in sorted(report["tool_usage_distribution"].items(), key=lambda x: x[1], reverse=True):
        tool_table.add_row(t_name, str(count), f"{(count/total_tool_calls*100):.2f}%")
    console.print(tool_table)


def _write_markdown_report(report: Dict[str, Any], md_path: Path):
    """Generate comprehensive markdown inspection documentation."""
    content = f"""# Glint-Research/Fable-5-traces Inspection Report

**Dataset Path:** `{report['dataset_path']}`  
**Tokenizer Evaluated:** `{report['tokenizer']}`  
**Total Records:** `{report['total_records']}`  
**Unique Sessions:** `{report['unique_sessions']}`  

---

## 1. High-Level Summary

- **Tool Use Actions:** {report['output_type_distribution'].get('tool_use', 0)} ({report['output_type_distribution'].get('tool_use', 0)/report['total_records']*100:.2f}%)
- **Assistant Text Responses:** {report['output_type_distribution'].get('text', 0)} ({report['output_type_distribution'].get('text', 0)/report['total_records']*100:.2f}%)
- **Mean Reasoning Tokens per Turn:** {report['token_stats']['cot_reasoning']['mean']:.1f} tokens (Median: {report['token_stats']['cot_reasoning']['p50_median']:.1f})
- **Mean Total Sequence Length:** {report['token_stats']['total_sequence']['mean']:.1f} tokens (Median: {report['token_stats']['total_sequence']['p50_median']:.1f})

---

## 2. Token Length Distribution (`{report['tokenizer']}`)

| Field | Min | P25 | Median (P50) | P75 | P90 | P95 | P99 | Max | Mean ± Std |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Context (Prompt)** | {report['token_stats']['context_prompt']['min']:.0f} | {report['token_stats']['context_prompt']['p25']:.0f} | {report['token_stats']['context_prompt']['p50_median']:.0f} | {report['token_stats']['context_prompt']['p75']:.0f} | {report['token_stats']['context_prompt']['p90']:.0f} | {report['token_stats']['context_prompt']['p95']:.0f} | {report['token_stats']['context_prompt']['p99']:.0f} | {report['token_stats']['context_prompt']['max']:.0f} | {report['token_stats']['context_prompt']['mean']:.0f} ± {report['token_stats']['context_prompt']['std']:.0f} |
| **CoT Reasoning (`<think>`)** | {report['token_stats']['cot_reasoning']['min']:.0f} | {report['token_stats']['cot_reasoning']['p25']:.0f} | {report['token_stats']['cot_reasoning']['p50_median']:.0f} | {report['token_stats']['cot_reasoning']['p75']:.0f} | {report['token_stats']['cot_reasoning']['p90']:.0f} | {report['token_stats']['cot_reasoning']['p95']:.0f} | {report['token_stats']['cot_reasoning']['p99']:.0f} | {report['token_stats']['cot_reasoning']['max']:.0f} | {report['token_stats']['cot_reasoning']['mean']:.0f} ± {report['token_stats']['cot_reasoning']['std']:.0f} |
| **Target Action / Output** | {report['token_stats']['target_action']['min']:.0f} | {report['token_stats']['target_action']['p25']:.0f} | {report['token_stats']['target_action']['p50_median']:.0f} | {report['token_stats']['target_action']['p75']:.0f} | {report['token_stats']['target_action']['p90']:.0f} | {report['token_stats']['target_action']['p95']:.0f} | {report['token_stats']['target_action']['p99']:.0f} | {report['token_stats']['target_action']['max']:.0f} | {report['token_stats']['target_action']['mean']:.0f} ± {report['token_stats']['target_action']['std']:.0f} |
| **Total Sequence Length** | {report['token_stats']['total_sequence']['min']:.0f} | {report['token_stats']['total_sequence']['p25']:.0f} | {report['token_stats']['total_sequence']['p50_median']:.0f} | {report['token_stats']['total_sequence']['p75']:.0f} | {report['token_stats']['total_sequence']['p90']:.0f} | {report['token_stats']['total_sequence']['p95']:.0f} | {report['token_stats']['total_sequence']['p99']:.0f} | {report['token_stats']['total_sequence']['max']:.0f} | {report['token_stats']['total_sequence']['mean']:.0f} ± {report['token_stats']['total_sequence']['std']:.0f} |

---

## 3. Context Window Truncation Budget

| Window Size | Fit Percentage | Samples Fit | Truncated Samples |
|:---|---:|---:|---:|
"""
    for k, v in report["context_window_budget"].items():
        content += f"| **{k.replace('_', ' ').upper()}** | {v['fit_percentage']}% | {v['fit_count']} | {v['truncated_count']} |\n"

    content += """
---

## 4. Tool Invocation Frequency

| Tool Name | Invocations | Share |
|:---|---:|---:|
"""
    total_tools = sum(report["tool_usage_distribution"].values())
    for t_name, count in sorted(report["tool_usage_distribution"].items(), key=lambda x: x[1], reverse=True):
        content += f"| `{t_name}` | {count} | {count/total_tools*100:.2f}% |\n"

    content += f"""
---

## 5. Dataset Quality & Anomaly Checks

- **Missing / Empty CoT:** {report['anomalies']['missing_cot_count']}
- **Empty Context:** {report['anomalies']['empty_context_count']}
- **Left-Truncated Contexts (Missing Initial User Prompt):** {report['anomalies']['missing_user_in_raw_context']}
- **Broken JSON in Tool Arguments:** {report['anomalies']['json_parse_errors_in_tools']}
- **Template Rendering Errors:** {report['anomalies']['template_rendering_errors_count']}

> [!NOTE]
> When training with Qwen 2.5 / Qwen 3.8 chat templates, ensure that left-truncated contexts are handled by prepending a fallback user prompt or reconstructing full multi-turn sessions using `convert_sft.py`.
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Inspect Glint-Research/Fable-5-traces dataset.")
    parser.add_argument("--data-path", type=str, default="dataset/data/fable5_cot_merged.jsonl", help="Path to jsonl dataset file")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3.8-27B", help="Hugging Face tokenizer name or local path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records to inspect (for fast preview)")
    parser.add_argument("--view", type=int, default=None, help="Render a specific sample index with syntax highlighting")
    parser.add_argument("--export-dir", type=str, default="dataset/reports", help="Directory to save inspection reports")

    args = parser.parse_args()

    if args.view is not None:
        records = load_records(args.data_path)
        if 0 <= args.view < len(records):
            tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
            view_sample(records[args.view], args.view, tok)
        else:
            console.print(f"[bold red]Index {args.view} out of range (0 to {len(records)-1})[/bold red]")
    else:
        inspect_dataset(
            data_path=args.data_path,
            tokenizer_name=args.tokenizer,
            sample_limit=args.limit,
            export_dir=args.export_dir,
        )


if __name__ == "__main__":
    main()
