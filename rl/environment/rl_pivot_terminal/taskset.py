"""Verifiers v1 Taskset for Nemotron / ATCB Terminal Pivot Reinforcement Learning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pydantic import Field

from verifiers.v1.configs.task import TaskConfig
from verifiers.v1.configs.taskset import TasksetConfig
from verifiers.v1.task import Task, TaskData
from verifiers.v1.taskset import Taskset
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages
from verifiers.v1.utils.decorators import reward


class TerminalPivotTaskData(TaskData):
    """TaskData wire model for a terminal decision pivot."""

    uuid: str = ""
    task_name: str = ""
    trajectory_uid: str = ""
    turn_index: int = 0
    total_turns: int = 1
    expected_answer_raw: Dict[str, Any] = Field(default_factory=dict)
    reference_completion: str = ""


class TerminalPivotTaskConfig(TaskConfig):
    """Configuration for individual terminal pivot tasks."""

    reward_think_format: bool = True
    reward_tool_format: bool = True
    reward_command_match: bool = True


class TerminalPivotTask(Task[TerminalPivotTaskData, TerminalPivotTaskConfig]):
    """Behavior and scoring implementation for a terminal decision pivot."""

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

        if not response_text:
            return 0.0

        score = 0.0

        # 1. Format Reward: Check for <think> and </think> tags
        if self.config.reward_think_format:
            if "<think>" in response_text and "</think>" in response_text:
                score += 0.2

        # 2. Action Reward: Check for valid tool call or task complete
        expected_raw = self.data.expected_answer_raw
        expected_complete = expected_raw.get("task_complete", False)
        expected_cmds = expected_raw.get("commands", [])

        if expected_complete and not expected_cmds:
            # Expected task complete
            if "Task complete" in response_text or "task_complete" in response_text:
                score += 0.8
        elif expected_cmds:
            # Expected bash commands
            if "<tool_call>" in response_text or "bash" in response_text:
                score += 0.4

            # Check if any expected command keywords appear in the agent's action
            matched_cmds = 0
            for cmd in expected_cmds:
                ks = cmd.get("keystrokes", "").strip()
                if ks and ks in response_text:
                    matched_cmds += 1

            if matched_cmds > 0:
                score += 0.4 * (matched_cmds / len(expected_cmds))

        return min(1.0, score)


class TerminalPivotConfig(TasksetConfig):
    """Configuration for loading Nemotron Terminal Pivot tasksets."""

    dataset_path: Path = Field(
        default=Path("rl_dataset/data/converted/nemotron_terminal_rl_train.jsonl"),
        description="Path to converted JSONL dataset file",
    )
    val_dataset_path: Path = Field(
        default=Path("rl_dataset/data/converted/nemotron_terminal_rl_val.jsonl"),
        description="Path to holdout validation JSONL dataset file",
    )
    split: str = Field("train", description="Split to load: 'train' or 'val'")
    num_tasks: Optional[int] = Field(None, description="Maximum number of tasks to load")
    task_filter: Optional[str] = Field(None, description="Substring filter for task names")


class TerminalPivotTaskset(Taskset[TerminalPivotTask, TerminalPivotConfig]):
    """Taskset loader yielding TerminalPivotTask instances for Verifiers v1 / Prime-RL."""

    def load(self) -> Iterable[TerminalPivotTask]:
        """Construct and yield typed tasks from JSONL dataset."""
        file_path = self.config.dataset_path if self.config.split == "train" else self.config.val_dataset_path
        if not file_path.exists():
            # Fallback to local search
            root_path = Path(__file__).resolve().parents[3]
            alt_path = root_path / file_path
            if alt_path.exists():
                file_path = alt_path
            else:
                raise FileNotFoundError(f"Terminal pivot dataset not found: {file_path}")

        count = 0
        task_cfg = TerminalPivotTaskConfig()

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if self.config.num_tasks is not None and count >= self.config.num_tasks:
                    break

                line_str = line.strip()
                if not line_str:
                    continue

                raw = json.loads(line_str)
                task_name = raw.get("task_name", "")

                if self.config.task_filter and self.config.task_filter.lower() not in task_name.lower():
                    continue

                data = TerminalPivotTaskData(
                    idx=count,
                    name=task_name,
                    prompt=raw.get("messages", []),
                    uuid=raw.get("uuid", ""),
                    task_name=task_name,
                    trajectory_uid=raw.get("trajectory_uid", ""),
                    turn_index=int(raw.get("turn_index", 0)),
                    total_turns=int(raw.get("total_turns", 1)),
                    expected_answer_raw=raw.get("expected_answer_raw", {}),
                    reference_completion=raw.get("reference_completion", ""),
                )

                yield TerminalPivotTask(data=data, config=task_cfg)
                count += 1
