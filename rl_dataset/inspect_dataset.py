"""Deep dataset inspection and statistical analysis engine for nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1."""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure UTF-8 console output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Import parser
sys.path.insert(0, str(Path(__file__).parent.parent))
from rl_dataset.parser import NemotronRLRecord, stream_nemotron_dataset


def approximate_token_count(text: str) -> int:
    """Fast approximation of token count (~3.5 chars per token for code/terminal text)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def extract_base_command(keystrokes: str) -> List[str]:
    """Extract primary shell command binaries from a keystroke string."""
    clean = keystrokes.strip()
    if not clean:
        return []

    # Split on command separators (;, &&, ||, |, \n)
    segments = re.split(r"[;&|\n]+", clean)
    base_cmds = []

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        # Strip leading env vars (e.g. VAR=1 cmd) or sudo
        tokens = seg.split()
        for tok in tokens:
            if "=" in tok and not tok.startswith("-"):
                continue
            if tok in ["sudo", "time", "nohup", "exec", "eval", "xargs"]:
                continue
            # Extract basename if path (e.g., /usr/bin/find -> find)
            cmd_name = Path(tok).name
            if cmd_name and not cmd_name.startswith("-") and not cmd_name.startswith("$"):
                base_cmds.append(cmd_name)
                break
    return base_cmds


def classify_task_domain(task_name: str, initial_prompt: str) -> str:
    """Categorize task into domain based on name and task instruction."""
    text = (task_name + " " + initial_prompt[:500]).lower()

    if any(k in text for k in ["siem", "audit", "security", "pii", "auth", "tls", "cert", "firewall", "permission", "access"]):
        return "Security & SIEM Audit"
    elif any(k in text for k in ["pipeline", "ingest", "etl", "telemetry", "reconcil", "stream", "kafka", "spark", "sql"]):
        return "Data Pipelines & Storage"
    elif any(k in text for k in ["crash", "coredump", "debug", "service", "systemd", "diagnostic", "fail", "oom", "gdb"]):
        return "Service Diagnostics & Crash Recovery"
    elif any(k in text for k in ["build", "compile", "ci", "docker", "makefile", "cmake", "test", "pytest", "package"]):
        return "Build, CI/CD & Testing"
    elif any(k in text for k in ["scada", "modbus", "embedded", "firmware", "sensor", "ota", "iot", "canbus"]):
        return "Embedded, SCADA & Industrial"
    else:
        return "System Administration & Automation"


def compute_quantiles(data: List[float]) -> Dict[str, float]:
    """Compute min, mean, p25, median, p75, p90, p95, p99, max."""
    if not data:
        return {k: 0.0 for k in ["min", "mean", "p25", "p50", "p75", "p90", "p95", "p99", "max"]}
    s = sorted(data)
    n = len(s)

    def get_p(p: float) -> float:
        idx = min(int(n * p), n - 1)
        return s[idx]

    return {
        "min": float(s[0]),
        "mean": float(sum(s) / n),
        "p25": float(get_p(0.25)),
        "p50": float(get_p(0.50)),
        "p75": float(get_p(0.75)),
        "p90": float(get_p(0.90)),
        "p95": float(get_p(0.95)),
        "p99": float(get_p(0.99)),
        "max": float(s[-1]),
    }


def inspect_dataset(
    dataset_path: str = "rl_dataset/data/atcb_terminal_pivot_release_final_v2.jsonl",
    output_dir: str = "rl_dataset/reports",
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Inspect dataset and generate comprehensive statistics."""
    path = Path(dataset_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"[*] STARTING DEEP INSPECTION: {path.resolve()}")
    print("=" * 80)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    file_size_bytes = path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    start_time = time.time()

    # Trackers
    total_records = 0
    unique_tasks = collections.Counter()
    unique_trajectories = set()
    teacher_models = collections.Counter()
    harnesses = collections.Counter()

    turn_indices = []
    total_turns_per_traj = []
    prompt_message_counts = []

    prompt_char_lengths = []
    prompt_token_lengths = []
    analysis_char_lengths = []
    plan_char_lengths = []
    total_answer_char_lengths = []

    task_complete_count = 0
    records_with_commands = 0
    commands_per_step = []
    command_frequencies = collections.Counter()
    domain_distribution = collections.Counter()

    sample_records_preview = []

    print("[*] Processing dataset records...")
    for rec in stream_nemotron_dataset(path, max_records=sample_limit):
        total_records += 1

        if total_records % 5000 == 0:
            print(f"    - Processed {total_records} records...", flush=True)

        # Metadata
        unique_tasks[rec.task_name] += 1
        if rec.metadata.source_trajectory_uid:
            unique_trajectories.add(rec.metadata.source_trajectory_uid)
        teacher_models[rec.metadata.teacher_model] += 1
        harnesses[rec.metadata.harness] += 1

        turn_indices.append(rec.metadata.pivot_agent_turn_index)
        total_turns_per_traj.append(rec.metadata.total_source_agent_turns)
        prompt_message_counts.append(len(rec.input_messages))

        # Text length metrics
        all_prompt_text = "".join(m.content for m in rec.input_messages)
        prompt_char_lengths.append(len(all_prompt_text))
        prompt_token_lengths.append(approximate_token_count(all_prompt_text))

        ans = rec.expected_answer
        analysis_char_lengths.append(len(ans.analysis))
        plan_char_lengths.append(len(ans.plan))
        total_ans_len = len(ans.analysis) + len(ans.plan) + sum(len(c.keystrokes) for c in ans.commands)
        total_answer_char_lengths.append(total_ans_len)

        if ans.task_complete:
            task_complete_count += 1

        num_cmds = len(ans.commands)
        commands_per_step.append(num_cmds)
        if num_cmds > 0:
            records_with_commands += 1
            for cmd in ans.commands:
                base_cmds = extract_base_command(cmd.keystrokes)
                for bc in base_cmds:
                    command_frequencies[bc] += 1

        # Domain classification
        first_prompt = rec.input_messages[0].content if rec.input_messages else ""
        domain = classify_task_domain(rec.task_name, first_prompt)
        domain_distribution[domain] += 1

        # Keep sample preview
        if len(sample_records_preview) < 3:
            sample_records_preview.append({
                "uuid": rec.uuid,
                "task_name": rec.task_name,
                "domain": domain,
                "turn_index": rec.metadata.pivot_agent_turn_index,
                "total_turns": rec.metadata.total_source_agent_turns,
                "input_messages_count": len(rec.input_messages),
                "analysis_snippet": ans.analysis[:200],
                "plan_snippet": ans.plan[:200],
                "commands": [c.keystrokes.strip() for c in ans.commands[:3]],
                "task_complete": ans.task_complete,
            })

    elapsed = time.time() - start_time
    print(f"[+] Completed processing in {elapsed:.2f}s! Total records: {total_records}")

    # Build metric report dictionary
    metrics = {
        "dataset_summary": {
            "dataset_filename": path.name,
            "file_size_mb": round(file_size_mb, 2),
            "total_records": total_records,
            "unique_tasks_count": len(unique_tasks),
            "unique_trajectories_count": len(unique_trajectories),
            "teacher_models": dict(teacher_models),
            "harnesses": dict(harnesses),
            "task_completion_rate_pct": round((task_complete_count / max(total_records, 1)) * 100, 2),
            "action_decision_rate_pct": round((records_with_commands / max(total_records, 1)) * 100, 2),
        },
        "trajectory_statistics": {
            "pivot_turn_index": compute_quantiles(turn_indices),
            "total_source_turns": compute_quantiles(total_turns_per_traj),
            "prompt_message_count": compute_quantiles(prompt_message_counts),
        },
        "content_length_statistics": {
            "prompt_char_length": compute_quantiles(prompt_char_lengths),
            "prompt_approx_token_length": compute_quantiles(prompt_token_lengths),
            "analysis_char_length": compute_quantiles(analysis_char_lengths),
            "plan_char_length": compute_quantiles(plan_char_lengths),
            "total_answer_char_length": compute_quantiles(total_answer_char_lengths),
        },
        "command_statistics": {
            "commands_per_decision_step": compute_quantiles(commands_per_step),
            "top_30_commands": dict(command_frequencies.most_common(30)),
        },
        "domain_distribution": dict(domain_distribution.most_common()),
        "top_20_tasks_by_frequency": dict(unique_tasks.most_common(20)),
        "samples": sample_records_preview,
    }

    # Save JSON summary
    json_path = out_dir / "dataset_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Saved metrics JSON to: {json_path.resolve()}")

    # Generate Markdown Report
    md_path = out_dir / "dataset_inspection_report.md"
    generate_markdown_report(metrics, md_path)
    print(f"[+] Saved comprehensive inspection report to: {md_path.resolve()}")

    # Print Terminal Overview
    print_terminal_summary(metrics)

    return metrics


def generate_markdown_report(metrics: Dict[str, Any], output_path: Path):
    """Render comprehensive markdown report with tables."""
    s = metrics["dataset_summary"]
    t = metrics["trajectory_statistics"]
    c = metrics["content_length_statistics"]
    cmd = metrics["command_statistics"]
    d = metrics["domain_distribution"]

    lines = [
        "# 📊 Nemotron-RL-Agentic-Terminal-Pivot-v1 Inspection Report",
        "",
        f"> **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> **Source Dataset:** `nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1`  ",
        f"> **File Size:** `{s['file_size_mb']} MB` | **Total Decision Points:** `{s['total_records']:,}`",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "| Metric | Value | Description |",
        "| :--- | :--- | :--- |",
        f"| **Total Pivot Records** | `{s['total_records']:,}` | Decision points extracted from successful trajectories |",
        f"| **Unique ATCB Tasks** | `{s['unique_tasks_count']:,}` | Distinct environment problem definitions |",
        f"| **Unique Source Trajectories** | `{s['unique_trajectories_count']:,}` | End-to-end task executions |",
        f"| **Teacher Model** | `{list(s['teacher_models'].keys())[0]}` | Model that generated the reference reasoning and actions |",
        f"| **Harness** | `{list(s['harnesses'].keys())[0]}` | Evaluation environment harness |",
        f"| **Command Action Rate** | `{s['action_decision_rate_pct']}%` | Percentage of steps executing terminal commands |",
        f"| **Task Completion Rate** | `{s['task_completion_rate_pct']}%` | Terminal steps marking goal completion |",
        "",
        "---",
        "",
        "## 2. Functional Domain Distribution",
        "",
        "| Domain Category | Samples | Proportion |",
        "| :--- | :--- | :--- |",
    ]

    tot = s["total_records"]
    for dom, count in d.items():
        pct = (count / max(tot, 1)) * 100
        lines.append(f"| **{dom}** | `{count:,}` | `{pct:.1f}%` |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Trajectory & Turn Distributions",
        "",
        "| Distribution | Min | p25 | Median (p50) | p75 | p90 | p95 | p99 | Max | Mean |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for k, row in [
        ("Pivot Turn Index", t["pivot_turn_index"]),
        ("Total Turns in Trajectory", t["total_source_turns"]),
        ("Input Messages in Prompt", t["prompt_message_count"]),
    ]:
        lines.append(
            f"| **{k}** | `{row['min']:.0f}` | `{row['p25']:.0f}` | `{row['p50']:.0f}` | `{row['p75']:.0f}` | `{row['p90']:.0f}` | `{row['p95']:.0f}` | `{row['p99']:.0f}` | `{row['max']:.0f}` | `{row['mean']:.1f}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Context & Token Length Distributions",
        "",
        "| Text Field | Min Tokens | p25 | Median | p75 | p90 | p95 | p99 | Max Tokens | Mean |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    pt = c["prompt_approx_token_length"]
    lines.append(
        f"| **Full Prompt (Context)** | `{pt['min']:.0f}` | `{pt['p25']:.0f}` | `{pt['p50']:.0f}` | `{pt['p75']:.0f}` | `{pt['p90']:.0f}` | `{pt['p95']:.0f}` | `{pt['p99']:.0f}` | `{pt['max']:.0f}` | `{pt['mean']:.1f}` |"
    )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Top 30 Shell Commands in Teacher Decisions",
        "",
        "| Rank | Command | Total Occurrences | Description / Context |",
        "| :--- | :--- | :--- | :--- |",
    ])

    cmd_descriptions = {
        "cat": "Inspecting file contents, configs, source code",
        "grep": "Searching logs, codebase patterns, error traces",
        "python": "Executing repair scripts, data transforms, testing",
        "find": "Locating files, configs, scripts across directory tree",
        "ls": "Directory listings, file permission inspections",
        "sed": "Inline text edits, regex replacements in configs",
        "docker": "Inspecting container state, images, compose logs",
        "pytest": "Running test suites to verify task completion",
        "curl": "Testing HTTP endpoints, webhook telemetry",
        "systemctl": "Checking service status, restarting daemons",
        "tail": "Inspecting live service logs and error streams",
        "head": "Sampling data files and large log headers",
        "awk": "Text processing, column extraction, log filtering",
        "git": "Checking git status, diffs, commits",
        "chmod": "Fixing file execution permissions",
        "echo": "Writing test inputs, config values",
        "rm": "Cleaning corrupted caches, temporary artifacts",
        "mkdir": "Creating required directories",
        "cp": "Backing up and replacing modified files",
        "pip": "Installing or updating Python dependencies",
    }

    for rank, (name, count) in enumerate(cmd["top_30_commands"].items(), 1):
        desc = cmd_descriptions.get(name, "Terminal utility operation")
        lines.append(f"| {rank} | `{name}` | `{count:,}` | {desc} |")

    lines.extend([
        "",
        "---",
        "",
        "## 6. Sample Decision Pivot Previews",
        "",
    ])

    for i, s_rec in enumerate(metrics["samples"], 1):
        lines.extend([
            f"### Sample {i}: Task `{s_rec['task_name']}` ({s_rec['domain']})",
            f"- **UUID:** `{s_rec['uuid']}`",
            f"- **Turn:** Step {s_rec['turn_index']} of {s_rec['total_turns']} (Prompt Messages: {s_rec['input_messages_count']})",
            f"- **Task Complete:** `{s_rec['task_complete']}`",
            "",
            "**Teacher Analysis (`<think>`):**",
            "```markdown",
            s_rec["analysis_snippet"],
            "```",
            "",
            "**Teacher Plan:**",
            "```markdown",
            s_rec["plan_snippet"],
            "```",
            "",
            "**Executed Keystrokes:**",
            "```bash",
            "\n".join(s_rec["commands"]),
            "```",
            "",
            "---",
        ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def print_terminal_summary(metrics: Dict[str, Any]):
    """Print high-level summary to console."""
    s = metrics["dataset_summary"]
    d = metrics["domain_distribution"]
    pt = metrics["content_length_statistics"]["prompt_approx_token_length"]
    top_cmds = list(metrics["command_statistics"]["top_30_commands"].items())[:8]

    print("\n" + "=" * 80)
    print("📈 NEMOTRON RL DATASET SUMMARY")
    print("=" * 80)
    print(f"[*] Total Pivot Records:       {s['total_records']:,}")
    print(f"[*] Unique ATCB Tasks:         {s['unique_tasks_count']:,}")
    print(f"[*] Unique Trajectories:       {s['unique_trajectories_count']:,}")
    print(f"[*] Teacher Model:             {list(s['teacher_models'].keys())[0]}")
    print(f"[*] Action Decision Rate:      {s['action_decision_rate_pct']}%")
    print(f"[*] Median Context Length:     {pt['p50']:.0f} tokens (p95: {pt['p95']:.0f}, Max: {pt['max']:.0f})")
    print("\n[*] Domain Distribution:")
    for dom, count in d.items():
        pct = (count / max(s['total_records'], 1)) * 100
        print(f"    - {dom:<35}: {count:6,d} ({pct:4.1f}%)")
    print("\n[*] Top Shell Commands:")
    cmd_str = ", ".join([f"{name} ({c:,})" for name, c in top_cmds])
    print(f"    {cmd_str}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Deep inspection of nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1 dataset.")
    parser.add_argument("--dataset", type=str, default="rl_dataset/data/atcb_terminal_pivot_release_final_v2.jsonl", help="Path to JSONL dataset")
    parser.add_argument("--output-dir", type=str, default="rl_dataset/reports", help="Directory for output reports")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit records for fast inspection test")

    args = parser.parse_args()
    inspect_dataset(dataset_path=args.dataset, output_dir=args.output_dir, sample_limit=args.sample_limit)


if __name__ == "__main__":
    main()
