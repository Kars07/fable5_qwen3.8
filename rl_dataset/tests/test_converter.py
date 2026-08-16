"""Unit tests for dataset format conversion and trajectory splitting."""

import json
from pathlib import Path
import pytest
from rl_dataset.convert_rl import convert_dataset, convert_record_to_qwen_rl
from rl_dataset.parser import NemotronRLRecord


@pytest.fixture
def dummy_record():
    raw = {
        "schema_version": "v1.0",
        "uuid": "test-uuid-split",
        "task_name": "data-pipeline-repair",
        "tool_name": "terminal",
        "responses_create_params": {
            "input": [
                {"role": "user", "content": "Fix the data pipeline."}
            ]
        },
        "expected_answer": json.dumps({
            "analysis": "Inspecting pipeline errors.",
            "plan": "Check logs.",
            "commands": [{"keystrokes": "cat /var/log/pipeline.log\n", "duration": 1.0}],
            "task_complete": False,
        }),
        "agent_ref": {},
        "metadata": {
            "harness": "terminus_2",
            "teacher_model": "zai-org/GLM-5.1",
            "source_trajectory_uid": "traj-split-001",
            "pivot_agent_turn_index": 0,
            "total_source_agent_turns": 5,
        },
    }
    return NemotronRLRecord.from_dict(raw)


def test_convert_single_record(dummy_record):
    converted = convert_record_to_qwen_rl(dummy_record)
    assert converted["uuid"] == "test-uuid-split"
    assert converted["task_name"] == "data-pipeline-repair"
    assert len(converted["messages"]) == 2
    assert converted["messages"][0]["role"] == "system"
    assert "<think>" in converted["reference_completion"]
    assert "<tool_call>" in converted["reference_completion"]


def test_trajectory_split_no_leakage(tmp_path):
    # Create mock dataset file with 2 trajectories (each having 3 turns)
    mock_file = tmp_path / "mock_dataset.jsonl"
    recs = []
    for traj_idx in range(4):
        for turn_idx in range(3):
            recs.append({
                "schema_version": "v1.0",
                "uuid": f"uid-{traj_idx}-{turn_idx}",
                "task_name": f"task-{traj_idx}",
                "tool_name": "terminal",
                "responses_create_params": {"input": [{"role": "user", "content": f"Task {traj_idx}"}]},
                "expected_answer": json.dumps({
                    "analysis": "A", "plan": "P",
                    "commands": [{"keystrokes": "ls\n", "duration": 1.0}],
                    "task_complete": False,
                }),
                "metadata": {
                    "harness": "terminus_2",
                    "teacher_model": "zai-org/GLM-5.1",
                    "source_trajectory_uid": f"traj-{traj_idx}",
                    "pivot_agent_turn_index": turn_idx,
                    "total_source_agent_turns": 3,
                },
            })

    with open(mock_file, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    out_dir = tmp_path / "converted"
    result = convert_dataset(
        input_path=str(mock_file),
        output_dir=str(out_dir),
        val_split_ratio=0.25,  # 1 trajectory in val, 3 in train
        seed=42,
    )

    assert result["total_records"] == 12
    assert result["val_trajectories"] == 1
    assert result["train_trajectories"] == 3
    assert result["val_records"] == 3
    assert result["train_records"] == 9

    # Verify zero leakage
    train_uids = set()
    with open(result["train_path"], "r", encoding="utf-8") as f:
        for line in f:
            train_uids.add(json.loads(line)["trajectory_uid"])

    val_uids = set()
    with open(result["val_path"], "r", encoding="utf-8") as f:
        for line in f:
            val_uids.add(json.loads(line)["trajectory_uid"])

    assert len(train_uids.intersection(val_uids)) == 0
