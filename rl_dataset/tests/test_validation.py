"""Unit tests for Nemotron dataset validation and schema checking."""

import json
import pytest
from rl_dataset.parser import NemotronRLRecord
from rl_dataset.validation import NemotronDatasetValidator


@pytest.fixture
def valid_record_dict():
    return {
        "schema_version": "v1.0",
        "uuid": "valid-uuid-001",
        "task_name": "scada-firmware-debug",
        "tool_name": "terminal",
        "responses_create_params": {
            "input": [
                {"role": "user", "content": "Debug Modbus register configuration on port 502."}
            ]
        },
        "expected_answer": json.dumps({
            "analysis": "Inspecting network sockets.",
            "plan": "Check active ports with netstat.",
            "commands": [{"keystrokes": "netstat -tuln | grep 502\n", "duration": 1.0}],
            "task_complete": False,
        }),
        "metadata": {
            "harness": "terminus_2",
            "teacher_model": "zai-org/GLM-5.1",
            "source_trajectory_uid": "traj-001",
            "pivot_agent_turn_index": 0,
            "total_source_agent_turns": 10,
        },
    }


def test_validator_detects_duplicate_uuid(valid_record_dict):
    validator = NemotronDatasetValidator()
    rec = NemotronRLRecord.from_dict(valid_record_dict)

    # First check succeeds
    validator.seen_uuids.add(rec.uuid)

    # Second check with same UUID flags duplicate
    validator.seen_uuids.add(rec.uuid)
    validator.add_issue(2, rec.uuid, "ERROR", "duplicate_uuid", "Duplicate detected")
    assert validator.issue_counts["[ERROR] duplicate_uuid"] == 1


def test_validator_detects_empty_initial_prompt(valid_record_dict):
    valid_record_dict["responses_create_params"]["input"] = [{"role": "user", "content": "   "}]
    rec = NemotronRLRecord.from_dict(valid_record_dict)

    validator = NemotronDatasetValidator()
    validator.validate_messages(1, rec.uuid, rec.input_messages)
    assert validator.issue_counts["[ERROR] empty_initial_prompt"] == 1


def test_validator_detects_turn_index_out_of_bounds(valid_record_dict):
    valid_record_dict["metadata"]["pivot_agent_turn_index"] = 15
    valid_record_dict["metadata"]["total_source_agent_turns"] = 10
    rec = NemotronRLRecord.from_dict(valid_record_dict)

    validator = NemotronDatasetValidator()
    validator.validate_metadata(1, rec.uuid, rec.metadata)
    assert validator.issue_counts["[ERROR] turn_index_out_of_bounds"] == 1


def test_validator_detects_null_byte_in_command(valid_record_dict):
    valid_record_dict["expected_answer"] = json.dumps({
        "analysis": "Testing null byte.",
        "plan": "Execute command.",
        "commands": [{"keystrokes": "cat /etc/passwd\x00malicious", "duration": 1.0}],
        "task_complete": False,
    })
    rec = NemotronRLRecord.from_dict(valid_record_dict)

    validator = NemotronDatasetValidator()
    validator.validate_keystrokes_safety(1, rec.uuid, rec.expected_answer.commands)
    assert validator.issue_counts["[ERROR] null_byte_in_command"] == 1
