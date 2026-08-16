"""Deep integration and data integrity tests on downloaded Nemotron RL dataset."""

from pathlib import Path
import pytest
from rl_dataset.parser import stream_nemotron_dataset
from rl_dataset.inspect_dataset import classify_task_domain, extract_base_command

DATASET_PATH = Path("rl_dataset/data/atcb_terminal_pivot_release_final_v2.jsonl")


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="Dataset file not found locally")
def test_stream_first_500_records():
    """Verify first 500 records parse with 100% success rate."""
    count = 0
    uuids = set()

    for rec in stream_nemotron_dataset(DATASET_PATH, max_records=500):
        count += 1
        assert rec.uuid is not None and len(rec.uuid) > 0
        assert rec.task_name is not None and len(rec.task_name) > 0
        assert len(rec.input_messages) > 0
        assert rec.metadata.harness == "terminus_2"
        assert rec.metadata.total_source_agent_turns > 0
        assert rec.metadata.pivot_agent_turn_index < rec.metadata.total_source_agent_turns
        uuids.add(rec.uuid)

    assert count == 500
    assert len(uuids) == 500  # All UUIDs unique


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="Dataset file not found locally")
def test_domain_classification_coverage():
    """Verify that domain classifier covers diverse task types."""
    domains = set()
    for rec in stream_nemotron_dataset(DATASET_PATH, max_records=200):
        dom = classify_task_domain(rec.task_name, rec.input_messages[0].content)
        domains.add(dom)

    # Must find at least 3 distinct domains in 200 samples
    assert len(domains) >= 3


def test_extract_base_command_helper():
    """Verify shell command extraction logic."""
    assert extract_base_command("cat /etc/hosts | grep localhost\n") == ["cat", "grep"]
    assert extract_base_command("sudo python3 -m pytest tests/\n") == ["python3"]
    assert extract_base_command("find /var/log -name '*.log' ; tail -n 20 error.log\n") == ["find", "tail"]
    assert extract_base_command("VAR=123 docker compose up -d\n") == ["docker"]
