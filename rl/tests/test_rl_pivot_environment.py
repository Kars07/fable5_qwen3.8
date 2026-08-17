"""Unit tests for Nemotron Terminal Pivot Taskset and Reward Function."""

from __future__ import annotations

from pathlib import Path
import pytest

from rl.environment.rl_pivot_terminal.taskset import (
    TerminalPivotConfig,
    TerminalPivotTask,
    TerminalPivotTaskConfig,
    TerminalPivotTaskData,
    TerminalPivotTaskset,
)
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.graph import MessageNode
from verifiers.v1.trace import AgentInfo, Trace, TraceTask
from verifiers.v1.types import AssistantMessage


def test_terminal_pivot_taskset_loading() -> None:
    """Verify taskset loader yields typed tasks with correct metadata."""
    train_path = Path("rl_dataset/data/converted/nemotron_terminal_rl_train.jsonl")
    if not train_path.exists():
        pytest.skip("Converted dataset not found locally.")

    config = TerminalPivotConfig(num_tasks=5, split="train")
    taskset = TerminalPivotTaskset(config)
    tasks = list(taskset)

    assert len(tasks) == 5
    for task in tasks:
        assert isinstance(task, TerminalPivotTask)
        assert isinstance(task.data, TerminalPivotTaskData)
        assert task.data.task_name != ""
        assert task.data.uuid != ""
        assert task.data.total_turns > 0
        assert len(task.data.prompt) > 0


@pytest.mark.asyncio
async def test_terminal_pivot_reward_scoring() -> None:
    """Verify format reward and command matching reward calculations."""
    task_data = TerminalPivotTaskData(
        uuid="test-uuid",
        task_name="test-task",
        expected_answer_raw={
            "analysis": "Testing",
            "plan": "Run ls",
            "commands": [{"keystrokes": "ls -la /app\n", "duration": 1.0}],
            "task_complete": False,
        },
        reference_completion="<think>Analysis</think><tool_call>ls -la /app</tool_call>",
    )
    task = TerminalPivotTask(data=task_data, config=TerminalPivotTaskConfig())

    task_info = TraceTask(type="TerminalPivotTask", data=task_data)
    agent_info = AgentInfo(config=AgentConfig())

    # 1. Perfect model response with <think> tags and exact tool call
    perfect_trace = Trace(
        task=task_info,
        agent=agent_info,
        nodes=[
            MessageNode(
                id=0,
                message=AssistantMessage(
                    content="<think>\nI will list files.\n</think>\n<tool_call>\nls -la /app\n</tool_call>"
                ),
            )
        ],
    )
    score_perfect = await task.evaluate_decision(perfect_trace)
    assert score_perfect == 1.0  # 0.2 format + 0.4 tool + 0.4 command match = 1.0

    # 2. Response missing <think> tags
    no_think_trace = Trace(
        task=task_info,
        agent=agent_info,
        nodes=[
            MessageNode(
                id=0,
                message=AssistantMessage(
                    content="<tool_call>\nls -la /app\n</tool_call>"
                ),
            )
        ],
    )
    score_no_think = await task.evaluate_decision(no_think_trace)
    assert score_no_think == 0.8  # 0.4 tool + 0.4 command match = 0.8

    # 3. Completely unrelated response
    bad_trace = Trace(
        task=task_info,
        agent=agent_info,
        nodes=[
            MessageNode(
                id=0,
                message=AssistantMessage(content="Hello World!"),
            )
        ],
    )
    score_bad = await task.evaluate_decision(bad_trace)
    assert score_bad == 0.0
