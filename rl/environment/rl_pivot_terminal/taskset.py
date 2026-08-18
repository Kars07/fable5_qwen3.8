"""Verifiers v1 Taskset for Nemotron Terminal Pivot RL with Dense Reward Shaping."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field

from verifiers.v1.configs.task import TaskConfig
from verifiers.v1.configs.taskset import TasksetConfig
from verifiers.v1.task import Task, TaskData
from verifiers.v1.taskset import Taskset
from verifiers.v1.trace import Trace
from verifiers.v1.utils.decorators import reward


class TerminalPivotTaskData(TaskData):
    """Immutable data row representing an ATCB terminal state transition pivot."""

    model_config = ConfigDict(frozen=True)

    uuid: str = Field(default="", description="Unique UUID for this decision turn")
    task_name: str = Field(default="", description="Benchmark task / issue identifier")
    trajectory_uid: str = Field(default="", description="Source trajectory UID")
    turn_index: int = Field(default=0, description="0-indexed turn position in trajectory")
    total_turns: int = Field(default=1, description="Total turns in source trajectory")
    domain: str = Field(default="terminal", description="Task classification category")
    expected_answer_raw: dict[str, Any] = Field(
        default_factory=dict,
        description="Parsed ground-truth expected answer containing analysis, plan, and keystrokes",
    )
    reference_completion: str = Field(
        default="", description="Canonical assistant completion containing <think> and <tool_call>"
    )


class TerminalPivotTaskConfig(TaskConfig):
    """Configuration for individual terminal pivot tasks."""

    dense_rewards: bool = True


class TerminalPivotTask(Task[TerminalPivotTaskData, TerminalPivotTaskConfig]):
    """Terminal decision step evaluation task with dense verifiable reward shaping."""

    @reward
    async def evaluate_decision(self, trace: Trace) -> float:
        """Score the agent's generated decision against reference solution and format constraints."""
        response_text = ""
        if hasattr(trace, "nodes") and trace.nodes:
            for node in reversed(trace.nodes):
                if hasattr(node, "message") and hasattr(node.message, "content"):
                    response_text = str(node.message.content or "")
                    if response_text:
                        break

        if not response_text and hasattr(trace, "calls") and trace.calls:
            last_call = trace.calls[-1]
            response_text = str(getattr(last_call, "response", "") or "")

        score = 0.0

        # 1. Base Structure & Reasoning (0.20 max)
        has_think = ("<think>" in response_text and "</think>" in response_text)
        has_json_reasoning = ('"analysis"' in response_text and '"plan"' in response_text)
        if has_think or has_json_reasoning:
            score += 0.10
        elif "<think>" in response_text or '"analysis"' in response_text:
            score += 0.05

        has_json_schema = '"commands"' in response_text and '"task_complete"' in response_text
        has_tool_call = "<tool_call>" in response_text or "<command>" in response_text
        if has_json_schema or has_tool_call:
            score += 0.10
        elif any(kw in response_text for kw in ["```json", "```bash", "keystrokes"]):
            score += 0.05

        # Extract ONLY actual generated commands (no vocabulary bleed from analysis/plan)
        gen_cmds = []
        clean_str = response_text.strip()
        if "```json" in clean_str:
            clean_str = clean_str.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in clean_str:
            clean_str = clean_str.split("```", 1)[1].split("```", 1)[0].strip()

        parsed_json = None
        try:
            parsed_json = json.loads(clean_str)
        except Exception:
            m = re.search(r"(\{.*\})", response_text, re.DOTALL)
            if m:
                try:
                    parsed_json = json.loads(m.group(1))
                except Exception:
                    pass

        if parsed_json and isinstance(parsed_json, dict):
            raw_cmds = parsed_json.get("commands", [])
            if isinstance(raw_cmds, list):
                for c in raw_cmds:
                    if isinstance(c, dict) and "keystrokes" in c:
                        gen_cmds.append(str(c.get("keystrokes", "")).strip())
                    elif isinstance(c, str):
                        gen_cmds.append(c.strip())

        if not gen_cmds:
            tool_matches = re.findall(r"<(?:tool_call|command)>(.*?)</(?:tool_call|command)>", response_text, re.DOTALL)
            for tm in tool_matches:
                gen_cmds.append(tm.strip())

        def canonical_tokens(cmd_str: str) -> set[str]:
            tokens = set()
            for tok in re.findall(r"[\w\.\/-]+", cmd_str):
                clean_tok = tok.strip("'\"")
                if clean_tok.startswith("./"):
                    clean_tok = clean_tok[2:]
                clean_tok = clean_tok.rstrip("/")
                if clean_tok:
                    tokens.add(clean_tok)
                    if "/" in clean_tok:
                        tokens.add(clean_tok.split("/")[-1])
            return tokens

        all_gen_tokens: set[str] = set()
        gen_binaries: set[str] = set()
        for gc in gen_cmds:
            all_gen_tokens.update(canonical_tokens(gc))
            parts = gc.split()
            if parts:
                base_b = parts[0].strip("'\"")
                if base_b.startswith("./"):
                    base_b = base_b[2:]
                gen_binaries.add(base_b)

        # 2. Ground-Truth Command Correctness (0.50 max — Continuous Command-Token Similarity)
        expected_raw = self.data.expected_answer_raw
        expected_complete = expected_raw.get("task_complete", False)
        expected_cmds = expected_raw.get("commands", [])

        cmd_correctness = 0.0
        if expected_cmds:
            total_cmd_score = 0.0
            for cmd in expected_cmds:
                ks = cmd.get("keystrokes", "").strip()
                if not ks:
                    continue
                exp_tokens = canonical_tokens(ks)
                if not exp_tokens:
                    continue

                # Exact command string match against parsed generated commands
                if any(ks == gc or ks.strip() == gc.strip() for gc in gen_cmds):
                    total_cmd_score += 1.0
                    continue

                # Token & argument overlap strictly on generated commands
                overlap = exp_tokens.intersection(all_gen_tokens)
                overlap_ratio = len(overlap) / max(1, len(exp_tokens))

                # Binary / Tool match check
                base_cmd = ks.split()[0] if ks.split() else ""
                if base_cmd.startswith("./"):
                    base_cmd = base_cmd[2:]
                has_base = base_cmd in gen_binaries

                cmd_score = (0.25 if has_base else 0.0) + 0.75 * overlap_ratio
                total_cmd_score += min(1.0, cmd_score)

            cmd_correctness = min(1.0, (total_cmd_score / max(1, len(expected_cmds))))
            score += 0.50 * cmd_correctness

        # 3. Submission Integrity & Completion Gate (0.30 max)
        # Prevents fake edit + premature submission gaming:
        # False / unearned completion claims receive an active NEGATIVE PENALTY (-0.50).
        model_claims_complete = ('"task_complete": true' in response_text or '"task_complete":true' in response_text)

        if expected_complete:
            # Task SHOULD be completed at this step
            if model_claims_complete:
                if not expected_cmds or cmd_correctness >= 0.5:
                    score += 0.30
                else:
                    # Premature claim without proper fix -> Heavy negative penalty
                    score -= 0.50
            elif any(term in response_text.lower() for term in ["complete", "finished", "done", "submitted"]):
                score += 0.15
        else:
            # Task is NOT yet complete (intermediate step)
            if not model_claims_complete:
                # Continuous credit based on exact intermediate action precision
                score += 0.30 * cmd_correctness
            else:
                # False premature completion submission -> Heavy negative penalty (-0.50)
                score -= 0.50

        return max(-1.0, min(1.0, score))


def prune_prompt_messages(messages: list[dict[str, Any]], max_chars: int = 42000) -> list[dict[str, Any]]:
    """Ensure multi-turn prompt messages fit strictly within pre-rollout budget (<=14K tokens)."""
    if not messages or not isinstance(messages, list):
        return messages

    total_chars = sum(len(str(m.get("content", ""))) for m in messages if isinstance(m, dict))
    if total_chars <= max_chars:
        return messages

    kept_first = messages[0:1]
    remainder = messages[1:]

    pruned_remainder: list[dict[str, Any]] = []
    accumulated_chars = sum(len(str(m.get("content", ""))) for m in kept_first if isinstance(m, dict))

    for msg in reversed(remainder):
        if not isinstance(msg, dict):
            continue
        c = str(msg.get("content", ""))
        if len(c) > 8000:
            head = c[:2500]
            tail = c[-2500:]
            c = f"{head}\n\n[... historical terminal output truncated for context budget ...]\n\n{tail}"
            msg = {**msg, "content": c}

        if accumulated_chars + len(c) <= max_chars or not pruned_remainder:
            pruned_remainder.insert(0, msg)
            accumulated_chars += len(c)
        else:
            break

    return kept_first + pruned_remainder


class TerminalPivotConfig(TasksetConfig):
    """Configuration for Nemotron Terminal Pivot taskset loader."""

    id: str = "rl_pivot_terminal"
    split: str = "train"
    num_tasks: int | None = None
    start: int = 0
    task_name_filter: str | None = None
    dataset_path: str | None = None


class TerminalPivotTaskset(Taskset[TerminalPivotTask, TerminalPivotConfig]):
    """Taskset loader streaming ATCB terminal decision pivots."""

    def load(self) -> Iterable[TerminalPivotTask]:
        """Load converted dataset JSONL and yield typed tasks."""
        if self.config.dataset_path:
            file_path = Path(self.config.dataset_path)
        else:
            filename = (
                "nemotron_terminal_rl_train.jsonl"
                if self.config.split == "train"
                else "nemotron_terminal_rl_val.jsonl"
            )
            candidates = [
                Path("rl_dataset/data/converted") / filename,
                Path("/opt/rl_dataset/data/converted") / filename,
                Path("../rl_dataset/data/converted") / filename,
                Path("D:/fable5_qwen3.7/rl_dataset/data/converted") / filename,
            ]
            file_path = next((p for p in candidates if p.exists()), candidates[0])

        if not file_path.exists():
            raise FileNotFoundError(
                f"Converted RL dataset not found at {file_path}. Run rl_dataset/convert_rl.py first."
            )

        task_config = TerminalPivotTaskConfig()
        count = 0
        skipped = 0

        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                if skipped < self.config.start:
                    skipped += 1
                    continue

                if self.config.num_tasks is not None and count >= self.config.num_tasks:
                    break

                row = json.loads(line)
                task_name = row.get("task_name", "")
                if self.config.task_name_filter and self.config.task_name_filter not in task_name:
                    continue

                raw_messages = row.get("messages") or row.get("prompt") or []
                prompt_messages = prune_prompt_messages(raw_messages)
                reference = row.get("reference_completion") or row.get("expected_assistant_response") or ""
                task_data = TerminalPivotTaskData(
                    prompt=prompt_messages,
                    uuid=row.get("uuid", ""),
                    task_name=task_name,
                    trajectory_uid=row.get("trajectory_uid") or row.get("source_trajectory_uid") or "",
                    turn_index=row.get("turn_index", 0),
                    total_turns=row.get("total_turns", 1),
                    domain=row.get("domain", "terminal"),
                    expected_answer_raw=row.get("expected_answer_raw", {}),
                    reference_completion=reference,
                )

                yield TerminalPivotTask(data=task_data, config=task_config)
                count += 1
