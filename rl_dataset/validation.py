"""Comprehensive schema validation, integrity checking, and safety auditing for Nemotron RL dataset."""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure UTF-8 console output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))
from rl_dataset.parser import NemotronRLRecord, stream_nemotron_dataset


@dataclass
class ValidationIssue:
    """Represents a specific validation issue found in a record."""

    record_index: int
    uuid: str
    issue_type: str  # ERROR, WARNING, ANOMALY
    category: str
    message: str


class NemotronDatasetValidator:
    """Validates schema compliance, content sanity, keystroke safety, and trajectory consistency."""

    def __init__(
        self,
        dataset_path: str = "rl_dataset/data/atcb_terminal_pivot_release_final_v2.jsonl",
        max_errors_to_record: int = 100,
    ):
        self.dataset_path = Path(dataset_path)
        self.max_errors_to_record = max_errors_to_record
        self.issues: List[ValidationIssue] = []
        self.issue_counts: collections.Counter = collections.Counter()
        self.seen_uuids: Set[str] = set()
        self.trajectory_turns: Dict[str, Set[int]] = collections.defaultdict(set)
        self.trajectory_totals: Dict[str, int] = {}
        self.total_records_checked = 0

    def add_issue(self, idx: int, uid: str, issue_type: str, category: str, msg: str):
        """Record a validation issue."""
        self.issue_counts[f"[{issue_type}] {category}"] += 1
        if len(self.issues) < self.max_errors_to_record:
            self.issues.append(
                ValidationIssue(
                    record_index=idx,
                    uuid=uid,
                    issue_type=issue_type,
                    category=category,
                    message=msg,
                )
            )

    def validate_keystrokes_safety(self, idx: int, uid: str, commands: List[Any]):
        """Audit terminal keystrokes for malformed characters, null bytes, or dangerous anomalies."""
        for c_idx, cmd in enumerate(commands):
            ks = cmd.keystrokes
            if not ks or not ks.strip():
                self.add_issue(idx, uid, "WARNING", "empty_command", f"Command at index {c_idx} has empty keystrokes.")
                continue

            # Check for null bytes
            if "\x00" in ks:
                self.add_issue(idx, uid, "ERROR", "null_byte_in_command", f"Command {c_idx} contains null byte.")

            # Check for unbalanced quotes
            single_quotes = ks.count("'")
            double_quotes = ks.count('"')
            if single_quotes % 2 != 0:
                self.add_issue(idx, uid, "WARNING", "unbalanced_single_quotes", f"Command {c_idx} has unbalanced single quotes: {ks[:60]}")
            if double_quotes % 2 != 0:
                self.add_issue(idx, uid, "WARNING", "unbalanced_double_quotes", f"Command {c_idx} has unbalanced double quotes: {ks[:60]}")

            # Check duration bounds
            if cmd.duration < 0:
                self.add_issue(idx, uid, "ERROR", "negative_duration", f"Command {c_idx} duration {cmd.duration} is negative.")
            elif cmd.duration > 300.0:
                self.add_issue(idx, uid, "WARNING", "excessive_duration", f"Command {c_idx} duration {cmd.duration}s exceeds 5 minutes.")

    def validate_messages(self, idx: int, uid: str, messages: List[Any]):
        """Validate input message structure and roles."""
        if not messages:
            self.add_issue(idx, uid, "ERROR", "empty_input_messages", "Record has 0 input messages.")
            return

        first_msg = messages[0]
        if first_msg.role not in ["user", "system"]:
            self.add_issue(idx, uid, "WARNING", "unexpected_initial_role", f"Initial message role is '{first_msg.role}' instead of user/system.")

        if not first_msg.content.strip():
            self.add_issue(idx, uid, "ERROR", "empty_initial_prompt", "Initial message text content is completely empty.")

        # Check turn alternations
        for i in range(1, len(messages)):
            prev_role = messages[i - 1].role
            cur_role = messages[i].role
            if prev_role == cur_role and cur_role == "assistant":
                self.add_issue(idx, uid, "WARNING", "consecutive_assistant_turns", f"Consecutive assistant messages at turn {i}.")

    def validate_expected_answer(self, idx: int, uid: str, ans: Any):
        """Validate reasoning content and consistency."""
        if not ans.analysis.strip() and not ans.plan.strip():
            self.add_issue(idx, uid, "WARNING", "empty_reasoning", "Expected answer has neither analysis nor plan.")

        if ans.task_complete and ans.commands:
            self.add_issue(
                idx,
                uid,
                "ANOMALY",
                "task_complete_with_commands",
                f"Record marks task_complete=True while also emitting {len(ans.commands)} commands.",
            )

        if not ans.task_complete and not ans.commands:
            self.add_issue(
                idx,
                uid,
                "ERROR",
                "no_action_and_not_complete",
                "Record marks task_complete=False but provides 0 commands to execute.",
            )

    def validate_metadata(self, idx: int, uid: str, meta: Any):
        """Validate trajectory indexing and metadata consistency."""
        if meta.pivot_agent_turn_index >= meta.total_source_agent_turns:
            self.add_issue(
                idx,
                uid,
                "ERROR",
                "turn_index_out_of_bounds",
                f"Pivot turn {meta.pivot_agent_turn_index} >= total turns {meta.total_source_agent_turns}.",
            )

        if meta.source_trajectory_uid:
            self.trajectory_turns[meta.source_trajectory_uid].add(meta.pivot_agent_turn_index)
            self.trajectory_totals[meta.source_trajectory_uid] = meta.total_source_agent_turns

    def run_validation(self, sample_limit: Optional[int] = None) -> Dict[str, Any]:
        """Execute full dataset validation."""
        print("=" * 80)
        print(f"[*] RUNNING COMPREHENSIVE DATASET VALIDATION: {self.dataset_path.resolve()}")
        print("=" * 80)

        start_time = time.time()
        idx = 0

        for rec in stream_nemotron_dataset(self.dataset_path, max_records=sample_limit):
            idx += 1
            uid = rec.uuid

            # 1. UUID uniqueness
            if not uid:
                self.add_issue(idx, "<missing>", "ERROR", "missing_uuid", "Record is missing uuid.")
            elif uid in self.seen_uuids:
                self.add_issue(idx, uid, "ERROR", "duplicate_uuid", f"Duplicate uuid detected: {uid}")
            else:
                self.seen_uuids.add(uid)

            # 2. Validate input messages
            self.validate_messages(idx, uid, rec.input_messages)

            # 3. Validate expected answer
            self.validate_expected_answer(idx, uid, rec.expected_answer)

            # 4. Validate keystrokes
            self.validate_keystrokes_safety(idx, uid, rec.expected_answer.commands)

            # 5. Validate metadata
            self.validate_metadata(idx, uid, rec.metadata)

            if idx % 5000 == 0:
                print(f"    - Validated {idx} records...", flush=True)

        self.total_records_checked = idx
        elapsed = time.time() - start_time

        # Check trajectory continuity across full dataset
        non_contiguous_trajectories = 0
        for traj_uid, turns in self.trajectory_turns.items():
            total = self.trajectory_totals.get(traj_uid, 0)
            # Check if all turns from 0 to max_turn exist
            max_seen = max(turns)
            if len(turns) < (max_seen + 1):
                non_contiguous_trajectories += 1

        error_count = sum(v for k, v in self.issue_counts.items() if "[ERROR]" in k)
        warning_count = sum(v for k, v in self.issue_counts.items() if "[WARNING]" in k)
        anomaly_count = sum(v for k, v in self.issue_counts.items() if "[ANOMALY]" in k)

        print("\n" + "=" * 80)
        print("🛡️ VALIDATION SUMMARY")
        print("=" * 80)
        print(f"[*] Total Records Checked:    {self.total_records_checked:,}")
        print(f"[*] Total Unique UUIDs:       {len(self.seen_uuids):,}")
        print(f"[*] Total Errors:             {error_count:,}")
        print(f"[*] Total Warnings:           {warning_count:,}")
        print(f"[*] Total Anomalies:          {anomaly_count:,}")
        print(f"[*] Validation Duration:      {elapsed:.2f}s")
        print("\n[*] Issue Breakdown by Category:")
        if not self.issue_counts:
            print("    [+] Zero issues found! Dataset is 100% compliant.")
        else:
            for cat, count in sorted(self.issue_counts.items(), key=lambda x: -x[1]):
                print(f"    - {cat:<40}: {count:6,d}")
        print("=" * 80 + "\n")

        return {
            "total_records_checked": self.total_records_checked,
            "unique_uuids": len(self.seen_uuids),
            "errors": error_count,
            "warnings": warning_count,
            "anomalies": anomaly_count,
            "issue_breakdown": dict(self.issue_counts),
            "sample_issues": [asdict(iss) for iss in self.issues[:20]],
            "elapsed_seconds": round(elapsed, 2),
        }


def main():
    parser = argparse.ArgumentParser(description="Validate Nemotron RL dataset integrity and schema compliance.")
    parser.add_argument("--dataset", type=str, default="rl_dataset/data/atcb_terminal_pivot_release_final_v2.jsonl", help="Path to JSONL dataset")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit validation to N sample records")
    args = parser.parse_args()

    validator = NemotronDatasetValidator(dataset_path=args.dataset)
    results = validator.run_validation(sample_limit=args.sample_limit)
    if results["errors"] > 0:
        print(f"[-] Validation failed with {results['errors']} errors.")
        sys.exit(1)
    else:
        print("[+] Validation passed successfully with 0 fatal errors!")


if __name__ == "__main__":
    main()
