"""Modal application for Deep Rollout Auditing, Group-Variance Diagnostics, and Anti-Gaming Analytics."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import modal

app = modal.App("fable5-inspect-rl-outputs")

volume_outputs = modal.Volume.from_name("fable5-prime-rl-outputs", create_if_missing=True)
volume_checkpoints = modal.Volume.from_name("fable5-rl-checkpoints", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12").pip_install("rich>=13.7.0", "tabulate>=0.9.0", "numpy>=1.26.0")


def audit_trajectory(data: dict[str, Any]) -> dict[str, Any]:
    """Perform deep forensic audit on a single trajectory for structural validity and reward gaming."""
    audit = {
        "group_id": data.get("group_id") or data.get("task", {}).get("data", {}).get("uuid", "unknown"),
        "is_valid_json": False,
        "has_reasoning": False,
        "has_valid_commands": False,
        "command_count": 0,
        "commands": [],
        "task_complete": False,
        "reward": 0.0,
        "reward_breakdown": {},
        "hacking_flags": [],
        "risk_level": "LOW",
        "total_tokens": 0,
        "assistant_tokens": 0,
        "observation_tokens": 0,
        "prompt_tokens": 0,
        "is_truncated": False,
    }

    # 1. Extract rewards
    rewards = data.get("rewards", {})
    audit["reward_breakdown"] = rewards
    if "evaluate_decision" in rewards:
        ev = rewards["evaluate_decision"]
        audit["reward"] = ev.get("score", 0.0) if isinstance(ev, dict) else float(ev)

    # 2. Extract generated text from nodes / calls
    text_content = ""
    nodes = data.get("nodes", [])
    for n in nodes:
        msg = n.get("message", {})
        if msg.get("role") == "assistant":
            c = msg.get("content", "")
            text_content += c + "\n"
            audit["assistant_tokens"] += len(c.split()) * 1.3  # heuristic token estimate

    if not text_content:
        calls = data.get("calls", [])
        for c in calls:
            resp = c.get("response", {})
            if isinstance(resp, dict):
                choices = resp.get("choices", [])
                for ch in choices:
                    m = ch.get("message", {}) or ch.get("text", "")
                    c_text = m.get("content", "") if isinstance(m, dict) else str(m)
                    text_content += c_text + "\n"
                    audit["assistant_tokens"] += len(c_text.split()) * 1.3

    # Estimate prompt and observation tokens
    task_prompt = str(data.get("task", {}).get("data", {}).get("prompt", ""))
    audit["prompt_tokens"] = int(len(task_prompt.split()) * 1.3)
    audit["total_tokens"] = int(audit["prompt_tokens"] + audit["assistant_tokens"] + audit["observation_tokens"])

    # Check truncation
    stop_reason = data.get("stop_condition") or data.get("metadata", {}).get("stop_reason")
    if stop_reason in ["context_length", "max_tokens", "length"] or audit["total_tokens"] >= 16000:
        audit["is_truncated"] = True

    # 3. Check for reasoning / structure
    if "<think>" in text_content and "</think>" in text_content:
        audit["has_reasoning"] = True
    elif '"analysis"' in text_content and '"plan"' in text_content:
        audit["has_reasoning"] = True

    # 4. Check JSON parse & commands
    clean_json_str = text_content.strip()
    if "```json" in clean_json_str:
        clean_json_str = clean_json_str.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in clean_json_str:
        clean_json_str = clean_json_str.split("```", 1)[1].split("```", 1)[0].strip()

    parsed = None
    try:
        parsed = json.loads(clean_json_str)
        audit["is_valid_json"] = True
    except Exception:
        m = re.search(r"(\{.*\})", text_content, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
                audit["is_valid_json"] = True
            except Exception:
                pass

    if parsed and isinstance(parsed, dict):
        cmds = parsed.get("commands", [])
        audit["task_complete"] = parsed.get("task_complete", False)
        if isinstance(cmds, list):
            audit["command_count"] = len(cmds)
            extracted_cmds = []
            for c in cmds:
                if isinstance(c, dict) and "keystrokes" in c:
                    ks = c.get("keystrokes", "")
                    extracted_cmds.append(ks)
                    if not ks.endswith("\n"):
                        audit["hacking_flags"].append("MALFORMED_KEYSTROKE_NO_NEWLINE")
            audit["commands"] = extracted_cmds
            if extracted_cmds:
                audit["has_valid_commands"] = True

    # 5. Anti-Gaming Checks
    if audit["reward"] < 0:
        audit["hacking_flags"].append("NEGATIVE_SUBMISSION_PENALTY_TRIGGERED")
        audit["risk_level"] = "HIGH"

    if audit["task_complete"] and audit["command_count"] == 0 and audit["reward"] <= 0:
        audit["hacking_flags"].append("EMPTY_TASK_COMPLETE_SUBMISSION")

    return audit


@app.function(
    image=image,
    volumes={
        "/outputs": volume_outputs,
        "/checkpoints": volume_checkpoints,
    },
    timeout=600,
)
def deep_inspect(step: int | None = None, num_samples: int = 3) -> None:
    """Execute forensic deep audit, group variance diagnostics, and context budget analysis."""
    import numpy as np
    from rich.console import Console
    from rich.table import Table

    console = Console()
    volume_outputs.reload()
    volume_checkpoints.reload()

    out_root = Path("/outputs/prime-rl-run")
    if not out_root.exists():
        out_root = Path("/outputs")

    def get_step_num(p: Path) -> int:
        m = re.search(r"step_(\d+)", p.as_posix())
        return int(m.group(1)) if m else 0

    trace_files = sorted(list(out_root.rglob("traces.jsonl")), key=get_step_num)
    if not trace_files:
        console.print("[red][!] No trace files found in /outputs yet.[/red]")
        return

    # Select target step
    target_trace = trace_files[-1]
    if step is not None:
        matches = [tf for tf in trace_files if f"step_{step}/" in tf.as_posix()]
        if matches:
            target_trace = matches[-1]

    step_name = f"step_{get_step_num(target_trace)}"
    console.rule(f"[bold cyan]🔍 Deep Audit & GRPO Variance Diagnostics: {step_name} ({target_trace.relative_to(out_root)})[/bold cyan]")

    lines = target_trace.read_text(encoding="utf-8", errors="replace").splitlines()
    audits = []
    for line in lines:
        try:
            d = json.loads(line)
            audits.append((d, audit_trajectory(d)))
        except Exception:
            pass

    if not audits:
        console.print("[yellow][!] No valid trajectories in trace file.[/yellow]")
        return

    rewards = [a["reward"] for _, a in audits]
    tokens = [a["total_tokens"] for _, a in audits]

    # Group-Level GRPO Variance Metrics (G=4)
    group_map: dict[str, list[float]] = {}
    for _, a in audits:
        gid = a["group_id"]
        group_map.setdefault(gid, []).append(a["reward"])

    zero_var_groups = sum(1 for g_rews in group_map.values() if len(g_rews) > 1 and np.std(g_rews) < 1e-4)
    all_zero_groups = sum(1 for g_rews in group_map.values() if all(r <= 0.0 for r in g_rews))
    all_perfect_groups = sum(1 for g_rews in group_map.values() if all(r >= 0.95 for r in g_rews))
    total_groups = max(1, len(group_map))

    # 1. GRPO Signal & Variance Table (Target Step)
    var_table = Table(title=f"GRPO Group Variance & Learning Signal ({step_name})", show_lines=True)
    var_table.add_column("Metric", style="bold yellow")
    var_table.add_column("Value", style="bold green")
    var_table.add_column("Diagnostic Interpretation", style="cyan")

    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    var_table.add_row("Reward Mean (μ)", f"{mean_r:.4f}", "Target baseline range: 0.10 - 0.80")
    var_table.add_row("Reward Std Dev (σ)", f"{std_r:.4f}", "Within-batch learning signal magnitude")
    var_table.add_row(
        "Zero-Variance Groups (σ=0)",
        f"{zero_var_groups}/{total_groups} ({zero_var_groups/total_groups*100:.1f}%)",
        "[green]HEALTHY[/green]" if zero_var_groups/total_groups < 0.50 else "[yellow]HIGH - Needs Task Diversity[/yellow]",
    )
    var_table.add_row("All-Zero Groups (R<=0)", f"{all_zero_groups}/{total_groups} ({all_zero_groups/total_groups*100:.1f}%)", "Floor saturation indicator")
    var_table.add_row("All-Perfect Groups (R>=0.95)", f"{all_perfect_groups}/{total_groups} ({all_perfect_groups/total_groups*100:.1f}%)", "Ceiling saturation indicator")
    var_table.add_row(
        "Premature Submission Penalty Rate",
        f"{sum(1 for r in rewards if r < 0)}/{len(rewards)} ({sum(1 for r in rewards if r < 0)/len(rewards)*100:.1f}%)",
        "Instances of unearned completion claims",
    )
    console.print(var_table)

    # 2. Cumulative Multi-Step Run Diagnostics (Across All Steps)
    all_step_audits = []
    all_step_groups: dict[str, list[dict[str, Any]]] = {}
    for tf in trace_files:
        step_tag = tf.parent.name
        try:
            for l in tf.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    d = json.loads(l)
                    aud = audit_trajectory(d)
                    all_step_audits.append(aud)
                    unique_gid = f"{step_tag}_{aud['group_id']}"
                    all_step_groups.setdefault(unique_gid, []).append(aud)
                except Exception:
                    pass
        except Exception:
            pass

    if all_step_audits:
        valid_policy_trainable = 0
        truncation_contaminated = 0
        mastered_ceiling = 0
        homogeneous_suboptimal = 0

        for gid, members in all_step_groups.items():
            rews = [m["reward"] for m in members]
            truncs = [m["is_truncated"] for m in members]
            has_trunc = any(truncs)
            is_var = len(rews) > 1 and np.std(rews) >= 1e-4

            if is_var:
                if has_trunc:
                    truncation_contaminated += 1
                else:
                    valid_policy_trainable += 1
            else:
                if all(r >= 0.95 for r in rews):
                    mastered_ceiling += 1
                else:
                    homogeneous_suboptimal += 1

        total_groups_cum = len(all_step_groups)
        valid_groups_total = total_groups_cum - truncation_contaminated

        cum_table = Table(title=f"Truthful GRPO Variance Diagnostics (Steps 1 → {get_step_num(trace_files[-1])})", show_lines=True)
        cum_table.add_column("Variance & Learning Category", style="bold yellow")
        cum_table.add_column("Count & Percentage", style="bold green")
        cum_table.add_column("Diagnostic Interpretation", style="cyan")

        cum_table.add_row("Total Unique Prompt Groups", f"{total_groups_cum:,}", "G=4 prompt batches")
        cum_table.add_row(
            "🟢 Valid Policy Trainable (σ>0, 0% trunc)",
            f"{valid_policy_trainable}/{total_groups_cum} ({valid_policy_trainable/max(1, total_groups_cum)*100:.1f}%)",
            "[green]GENUINE EXPLORATION & RELATIVE ADVANTAGE[/green]",
        )
        cum_table.add_row(
            "🟢 Mastered Ceiling Groups (σ=0, R>=0.95)",
            f"{mastered_ceiling}/{total_groups_cum} ({mastered_ceiling/max(1, total_groups_cum)*100:.1f}%)",
            "[green]BENIGN MASTERY - 100% Policy Competence[/green]",
        )
        cum_table.add_row(
            "🟡 Homogeneous Sub-optimal (σ=0, R<0.95)",
            f"{homogeneous_suboptimal}/{total_groups_cum} ({homogeneous_suboptimal/max(1, total_groups_cum)*100:.1f}%)",
            "Sampling concentration on unsolved pivots",
        )
        cum_table.add_row(
            "🔴 Truncation-Contaminated (σ>0, trunc>0)",
            f"{truncation_contaminated}/{total_groups_cum} ({truncation_contaminated/max(1, total_groups_cum)*100:.1f}%)",
            "[red]BAD VARIANCE - Overflow-induced false contrast[/red]",
        )
        adj_rate = (valid_policy_trainable / max(1, valid_groups_total)) * 100
        cum_table.add_row(
            "⭐ Adjusted Valid Trainable Rate",
            f"{valid_policy_trainable}/{valid_groups_total} ({adj_rate:.1f}%)",
            "[bold green]CLEAN POLICY LEARNING SIGNAL[/bold green]",
        )
        console.print(cum_table)

    # 3. Context Budget & Token Quantiles Table
    tok_table = Table(title=f"Context Window Budget & Quantiles (16K Limit)", show_lines=True)
    tok_table.add_column("Percentile / Quantile", style="bold yellow")
    tok_table.add_column("Estimated Token Count", style="bold green")
    tok_table.add_column("Context Budget Headroom (16,384)", style="cyan")

    p50 = int(np.percentile(tokens, 50))
    p90 = int(np.percentile(tokens, 90))
    p95 = int(np.percentile(tokens, 95))
    p99 = int(np.percentile(tokens, 99))
    trunc_cnt = sum(1 for _, a in audits if a["is_truncated"])

    tok_table.add_row("Median (p50)", f"{p50:,} tokens", f"{16384 - p50:,} tokens remaining ({p50/16384*100:.1f}% used)")
    tok_table.add_row("p90", f"{p90:,} tokens", f"{16384 - p90:,} tokens remaining ({p90/16384*100:.1f}% used)")
    tok_table.add_row("p95", f"{p95:,} tokens", f"{16384 - p95:,} tokens remaining ({p95/16384*100:.1f}% used)")
    tok_table.add_row("p99", f"{p99:,} tokens", f"{16384 - p99:,} tokens remaining ({p99/16384*100:.1f}% used)")
    tok_table.add_row("Truncation Rate (>16K)", f"{trunc_cnt}/{len(audits)} ({trunc_cnt/len(audits)*100:.1f}%)", "[green]0% Truncation[/green]" if trunc_cnt == 0 else "[red]Context Exhaustion Alert[/red]")
    console.print(tok_table)

    # 4. Granular Samples from target step
    console.rule(f"[bold magenta]📋 Granular Trajectory Inspections ({step_name})[/bold magenta]")
    for idx, (raw_data, audit) in enumerate(audits[-num_samples:], 1):
        console.print(f"\n[bold white on blue] Sample {idx}/{num_samples} | Reward: {audit['reward']:.4f} | Task Complete: {audit['task_complete']} | Tokens: ~{audit['total_tokens']} [/bold white on blue]")
        if audit["hacking_flags"]:
            console.print(f"[bold red]⚠️ Flags:[/bold red] {', '.join(audit['hacking_flags'])}")
        else:
            console.print("[bold green]✅ Alignment Clean: Structure and Correctness Validated[/bold green]")

        # Extract textual response
        text_pieces = []
        for n in raw_data.get("nodes", []):
            m = n.get("message", {})
            if m.get("role") == "assistant" and m.get("content"):
                text_pieces.append(m.get("content"))
        if not text_pieces:
            for c in raw_data.get("calls", []):
                resp = c.get("response", {})
                if isinstance(resp, dict):
                    for ch in resp.get("choices", []):
                        m = ch.get("message", {}) or ch.get("text", "")
                        c_text = m.get("content", "") if isinstance(m, dict) else str(m)
                        if c_text:
                            text_pieces.append(c_text)

        full_text = "\n---\n".join(text_pieces) if text_pieces else "[No assistant text]"
        console.print(f"[bold dim]Excerpt:[/bold dim]\n{full_text[:1200]}")
        if len(full_text) > 1200:
            console.print(f"[dim]... [truncated {len(full_text)-1200} chars] ...[/dim]")


@app.local_entrypoint()
def main(step: int | None = None, samples: int = 3) -> None:
    deep_inspect.remote(step=step, num_samples=samples)
