"""Parser and data structures for nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1 dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator


class TerminalCommand(BaseModel):
    """Represents a single terminal action command with keystrokes and duration."""

    keystrokes: str = Field(..., description="Raw keystrokes or shell command sent to the terminal")
    duration: float = Field(1.0, ge=0.0, description="Duration in seconds for terminal keystroke execution")

    @field_validator("keystrokes", mode="before")
    @classmethod
    def validate_keystrokes(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v)


class TerminusExpectedAnswer(BaseModel):
    """The reference teacher decision in Terminus-2 format."""

    analysis: str = Field(..., description="Teacher chain-of-thought analysis of terminal state")
    plan: str = Field(..., description="Teacher step-by-step plan for next actions")
    commands: List[TerminalCommand] = Field(default_factory=list, description="Sequence of terminal commands")
    task_complete: bool = Field(False, description="Whether the task goal is marked as completed")

    @classmethod
    def from_raw(cls, raw: Union[str, Dict[str, Any]]) -> "TerminusExpectedAnswer":
        """Parse from raw JSON string or dictionary."""
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # Handle edge-case raw string
                return cls(analysis=raw, plan="", commands=[], task_complete=False)
        elif isinstance(raw, dict):
            data = raw
        else:
            raise ValueError(f"Unsupported expected_answer type: {type(raw)}")

        commands_raw = data.get("commands", [])
        commands_list = []
        if isinstance(commands_raw, list):
            for cmd in commands_raw:
                if isinstance(cmd, dict):
                    commands_list.append(
                        TerminalCommand(
                            keystrokes=cmd.get("keystrokes", ""),
                            duration=float(cmd.get("duration", 1.0)),
                        )
                    )
                elif isinstance(cmd, str):
                    commands_list.append(TerminalCommand(keystrokes=cmd, duration=1.0))

        return cls(
            analysis=str(data.get("analysis", "")),
            plan=str(data.get("plan", "")),
            commands=commands_list,
            task_complete=bool(data.get("task_complete", False)),
        )


class ChatMessage(BaseModel):
    """Single message in the prompt trajectory."""

    role: str = Field(..., description="Role of the speaker (user, assistant, tool, system)")
    content: str = Field(..., description="Message text content or terminal buffer")


class PivotMetadata(BaseModel):
    """Environment and trajectory metadata for a pivot point."""

    harness: str = Field("terminus_2", description="Test harness name")
    teacher_model: str = Field("", description="Teacher model that generated reference action")
    source_trajectory_uid: str = Field("", description="Unique ID of source trajectory")
    pivot_agent_turn_index: int = Field(0, ge=0, description="Turn index of this decision point")
    total_source_agent_turns: int = Field(1, ge=1, description="Total turns in full source trajectory")


class NemotronRLRecord(BaseModel):
    """Full typed representation of a Nemotron RL decision point record."""

    schema_version: str = Field("v1.0", description="Schema version")
    uuid: str = Field(..., description="Unique ID for this decision pivot")
    task_name: str = Field(..., description="ATCB task name")
    tool_name: str = Field("terminal", description="Primary tool name")
    input_messages: List[ChatMessage] = Field(..., description="Interaction history up to pivot")
    expected_answer: TerminusExpectedAnswer = Field(..., description="Reference next action")
    agent_ref: Dict[str, Any] = Field(default_factory=dict, description="NeMo Gym judge agent routing")
    metadata: PivotMetadata = Field(..., description="Trajectory and environment metadata")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NemotronRLRecord":
        """Build a typed record from raw JSON dictionary."""
        # 1. Parse input messages
        raw_inputs = data.get("responses_create_params", {}).get("input", [])
        input_messages = []
        if isinstance(raw_inputs, list):
            for msg in raw_inputs:
                if isinstance(msg, dict):
                    input_messages.append(
                        ChatMessage(
                            role=str(msg.get("role", "user")),
                            content=str(msg.get("content", "")),
                        )
                    )

        # 2. Parse expected answer
        expected_raw = data.get("expected_answer", {})
        expected_answer = TerminusExpectedAnswer.from_raw(expected_raw)

        # 3. Parse metadata
        meta_raw = data.get("metadata", {})
        metadata = PivotMetadata(
            harness=str(meta_raw.get("harness", "terminus_2")),
            teacher_model=str(meta_raw.get("teacher_model", "")),
            source_trajectory_uid=str(meta_raw.get("source_trajectory_uid", "")),
            pivot_agent_turn_index=int(meta_raw.get("pivot_agent_turn_index", 0)),
            total_source_agent_turns=int(meta_raw.get("total_source_agent_turns", 1)),
        )

        return cls(
            schema_version=str(data.get("schema_version", "v1.0")),
            uuid=str(data.get("uuid", "")),
            task_name=str(data.get("task_name", "")),
            tool_name=str(data.get("tool_name", "terminal")),
            input_messages=input_messages,
            expected_answer=expected_answer,
            agent_ref=data.get("agent_ref", {}),
            metadata=metadata,
        )


def stream_nemotron_dataset(
    file_path: Union[str, Path],
    max_records: Optional[int] = None,
) -> Iterator[NemotronRLRecord]:
    """Iterate over dataset records from JSONL file with memory efficiency."""
    path = Path(file_path)
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_records is not None and count >= max_records:
                break
            line_str = line.strip()
            if not line_str:
                continue
            raw_dict = json.loads(line_str)
            yield NemotronRLRecord.from_dict(raw_dict)
            count += 1


def extract_initial_task_prompt(record: NemotronRLRecord) -> str:
    """Extract initial user task instruction from prompt messages."""
    if not record.input_messages:
        return ""
    return record.input_messages[0].content


def format_as_qwen_rl_prompt(
    record: NemotronRLRecord,
    include_system_prompt: bool = True,
) -> List[Dict[str, str]]:
    """Format the record prompt history into standard Qwen chat messages."""
    messages = []
    if include_system_prompt:
        system_content = (
            "You are an expert AI terminal agent operating in a Linux environment.\n"
            "Analyze the terminal state step-by-step in <think> tags, plan your action, "
            "and execute bash commands or state task completion."
        )
        messages.append({"role": "system", "content": system_content})

    for msg in record.input_messages:
        messages.append({"role": msg.role, "content": msg.content})

    return messages


def format_expected_as_assistant_response(record: NemotronRLRecord) -> str:
    """Convert expected_answer into native Qwen CoT + tool call response."""
    ans = record.expected_answer
    thought_parts = []
    if ans.analysis:
        thought_parts.append(ans.analysis)
    if ans.plan:
        thought_parts.append(f"Plan: {ans.plan}")

    thought_str = "\n\n".join(thought_parts)
    response = f"<think>\n{thought_str}\n</think>\n"

    if ans.task_complete and not ans.commands:
        response += "Task complete."
    elif ans.commands:
        commands_json = [cmd.keystrokes for cmd in ans.commands]
        response += f'<tool_call>\n{{"name": "bash", "arguments": {{"commands": {json.dumps(commands_json)}}}}}\n</tool_call>'

    return response
