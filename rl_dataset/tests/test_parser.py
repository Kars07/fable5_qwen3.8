"""Unit tests for Nemotron RL dataset parser and data structures."""

import json
import pytest
from rl_dataset.parser import (
    ChatMessage,
    NemotronRLRecord,
    PivotMetadata,
    TerminalCommand,
    TerminusExpectedAnswer,
    extract_initial_task_prompt,
    format_as_qwen_rl_prompt,
    format_expected_as_assistant_response,
)


@pytest.fixture
def sample_raw_record():
    return {
        "schema_version": "v1.0",
        "uuid": "test-uuid-12345",
        "task_name": "telemetry-pipeline-audit-2",
        "tool_name": "terminal",
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": "You are an AI assistant tasked with solving command-line tasks.\nFix the telemetry pipeline in /app.",
                },
                {
                    "role": "assistant",
                    "content": "I will inspect the directory structure.",
                },
                {
                    "role": "user",
                    "content": "ingest.py\nconfig.json\nlogs/",
                },
            ]
        },
        "expected_answer": json.dumps({
            "analysis": "The directory contains ingest.py and config.json. I need to read config.json.",
            "plan": "Check the configuration file for database connection parameters.",
            "commands": [
                {"keystrokes": "cat /app/config.json\n", "duration": 0.5},
                {"keystrokes": "python3 /app/ingest.py --test\n", "duration": 1.5},
            ],
            "task_complete": False,
        }),
        "agent_ref": {
            "type": "responses_api_agents",
            "name": "terminus_judge_string_only_simple_agent",
        },
        "metadata": {
            "harness": "terminus_2",
            "teacher_model": "zai-org/GLM-5.1",
            "source_trajectory_uid": "traj-abc-999",
            "pivot_agent_turn_index": 1,
            "total_source_agent_turns": 15,
        },
    }


def test_parse_terminal_command():
    cmd = TerminalCommand(keystrokes="ls -la /app\n", duration=0.8)
    assert cmd.keystrokes == "ls -la /app\n"
    assert cmd.duration == 0.8


def test_parse_expected_answer_json_string():
    raw_json = json.dumps({
        "analysis": "Need to check permissions.",
        "plan": "Run chmod +x script.sh.",
        "commands": [{"keystrokes": "chmod +x script.sh\n", "duration": 0.5}],
        "task_complete": False,
    })
    ans = TerminusExpectedAnswer.from_raw(raw_json)
    assert ans.analysis == "Need to check permissions."
    assert ans.plan == "Run chmod +x script.sh."
    assert len(ans.commands) == 1
    assert ans.commands[0].keystrokes == "chmod +x script.sh\n"
    assert not ans.task_complete


def test_parse_expected_answer_task_complete():
    raw_dict = {
        "analysis": "All tests pass. Task is complete.",
        "plan": "Mark task completed.",
        "commands": [],
        "task_complete": True,
    }
    ans = TerminusExpectedAnswer.from_raw(raw_dict)
    assert ans.task_complete is True
    assert len(ans.commands) == 0


def test_parse_nemotron_record(sample_raw_record):
    rec = NemotronRLRecord.from_dict(sample_raw_record)
    assert rec.uuid == "test-uuid-12345"
    assert rec.task_name == "telemetry-pipeline-audit-2"
    assert len(rec.input_messages) == 3
    assert rec.input_messages[0].role == "user"
    assert rec.metadata.source_trajectory_uid == "traj-abc-999"
    assert rec.metadata.pivot_agent_turn_index == 1
    assert len(rec.expected_answer.commands) == 2


def test_extract_initial_task_prompt(sample_raw_record):
    rec = NemotronRLRecord.from_dict(sample_raw_record)
    prompt = extract_initial_task_prompt(rec)
    assert "Fix the telemetry pipeline in /app." in prompt


def test_format_as_qwen_rl_prompt(sample_raw_record):
    rec = NemotronRLRecord.from_dict(sample_raw_record)
    qwen_messages = format_as_qwen_rl_prompt(rec, include_system_prompt=True)
    assert len(qwen_messages) == 4
    assert qwen_messages[0]["role"] == "system"
    assert "expert AI terminal agent" in qwen_messages[0]["content"]
    assert qwen_messages[1]["role"] == "user"


def test_format_expected_as_assistant_response(sample_raw_record):
    rec = NemotronRLRecord.from_dict(sample_raw_record)
    resp = format_expected_as_assistant_response(rec)
    assert "<think>" in resp
    assert "</think>" in resp
    assert "<tool_call>" in resp
    assert '"name": "bash"' in resp
    assert "cat /app/config.json" in resp
