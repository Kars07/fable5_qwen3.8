"""Modal script to inspect bug-fixing actions, patches, and command integrity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal
from rich.console import Console
from rich.panel import Panel

app = modal.App("fable5-inspect-bug-fixes")
volume_outputs = modal.Volume.from_name("fable5-prime-rl-outputs", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("rich>=13.7.0")


@app.function(
    image=image,
    volumes={"/outputs": volume_outputs},
    timeout=600,
)
def inspect_bug_fixes(limit: int = 4):
    console = Console(force_terminal=True, width=120)
    volume_outputs.reload()
    base_dir = Path("/outputs/prime-rl-run/rollouts")

    if not base_dir.exists():
        console.print("[red]No rollouts directory found.[/red]")
        return

    trace_files = sorted(base_dir.rglob("*.jsonl"))
    console.print(f"[bold cyan]Found {len(trace_files)} trace files across all steps.[/bold cyan]\n")

    fix_actions = []
    for tf in trace_files:
        step_name = tf.parent.name
        for line in tf.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                d = json.loads(line)
                rewards = d.get("rewards", {})
                score = rewards.get("evaluate_decision", {}).get("score", 0.0) if isinstance(rewards.get("evaluate_decision"), dict) else rewards.get("evaluate_decision", 0.0)
                nodes = d.get("nodes", [])
                for n in nodes:
                    msg = n.get("message", {})
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        # Look for late turns or edit actions
                        turn = d.get("task", {}).get("data", {}).get("turn_index", 0)
                        if any(k in content.lower() for k in ["sed", "echo", "python", "patch", "violation", "fix", "writing", "created", "task_complete"]):
                            fix_actions.append({
                                "step": step_name,
                                "task": d.get("task", {}).get("data", {}).get("task_name", "terminal-task"),
                                "turn": turn,
                                "total_turns": d.get("task", {}).get("data", {}).get("total_turns", 1),
                                "score": score,
                                "content": content,
                            })
            except Exception:
                pass

    console.print(f"[bold green]Identified {len(fix_actions)} diagnostic & fix trajectories.[/bold green]\n")

    for i, item in enumerate(fix_actions[-limit:], 1):
        parsed = {}
        try:
            clean = item["content"].strip()
            if "```json" in clean:
                clean = clean.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in clean:
                clean = clean.split("```", 1)[1].split("```", 1)[0].strip()
            parsed = json.loads(clean)
        except Exception:
            pass

        analysis = parsed.get("analysis", item["content"][:300])
        plan = parsed.get("plan", "")
        cmds = parsed.get("commands", [])

        body = f"[bold yellow]Task:[/bold yellow] {item['task']} (Turn {item['turn']}/{item['total_turns']})\n"
        body += f"[bold yellow]Reward Score:[/bold yellow] [bold green]{item['score']:.4f}[/bold green]\n\n"
        body += f"[bold cyan]Analysis & Bug Diagnosis:[/bold cyan]\n{analysis}\n\n"
        if plan:
            body += f"[bold cyan]Fix Plan:[/bold cyan]\n{plan}\n\n"
        if cmds:
            body += f"[bold magenta]Executed Commands:[/bold magenta]\n"
            for c in cmds:
                body += f"  $ {c.get('keystrokes', '').strip()}\n"

        console.print(Panel(body, title=f"Bug Fix Trajectory [{item['step']}] Example {i}/{limit}", border_style="green" if item['score'] > 0.7 else "yellow"))
