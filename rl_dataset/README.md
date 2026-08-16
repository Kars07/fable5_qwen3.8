# 🚀 Nemotron-RL-Agentic-Terminal-Pivot-v1 Dataset Toolkit

Comprehensive inspection, validation, parsing, visualization, and conversion suite for [`nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1`](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1).

---

## 📊 Dataset Overview

The dataset provides **31,111 decision points** extracted from successful agent trajectories across **630 unique ATCB Linux terminal environments**. Each record represents an agent pivot point:
- **`responses_create_params.input`**: Natural language task description + historical terminal stdout/stderr interactions up to the decision step.
- **`expected_answer`**: Teacher agent action in **Terminus-2 format** containing `<think>`-style `analysis`, step-by-step `plan`, sequence of shell `commands[].keystrokes`, and `task_complete` status.
- **`agent_ref`**: NeMo Gym `terminus_judge` router for verifiable reward computation (RLVR).

| Metric | Value |
| :--- | :--- |
| **Total Decision Points** | `31,111` records |
| **Unique ATCB Tasks** | `630` seed tasks |
| **Unique Trajectories** | `2,716` successful trajectories |
| **Teacher Model** | `zai-org/GLM-5.1` |
| **Harness** | `terminus_2` |
| **Median Context Length** | `7,397` tokens (p95: `26,909`, Max: `73,306`) |
| **Validation Status** | **0 Fatal Errors** (100% Schema Compliant) |

---

## 📁 Directory Structure

```
rl_dataset/
├── data/
│   ├── atcb_terminal_pivot_release_final_v2.jsonl   # Raw HuggingFace dataset (1.31 GB)
│   └── converted/                                   # Qwen RL / GRPO converted datasets
│       ├── nemotron_terminal_rl_train.jsonl         # Trajectory-grouped Train set (95%)
│       ├── nemotron_terminal_rl_val.jsonl           # Trajectory-grouped Holdout Val set (5%)
│       └── nemotron_terminal_rl_all.jsonl           # Full converted dataset
├── reports/
│   ├── dataset_inspection_report.md                 # Detailed statistical inspection report
│   └── dataset_metrics.json                         # Full quantile metrics JSON
├── tests/
│   ├── test_parser.py                               # Unit tests for data models & parsing
│   ├── test_validation.py                           # Unit tests for schema & safety auditing
│   ├── test_data_integrity.py                       # Integration tests on live dataset
│   └── test_converter.py                            # Tests for trajectory-aware splitting
├── download.py                                      # Dataset downloader with SHA256 caching
├── parser.py                                        # Pydantic models & format helpers
├── inspect_dataset.py                               # Deep statistical & quantile profiler
├── validation.py                                    # 100% record validator & safety auditor
├── visualize.py                                     # Interactive CLI trajectory viewer
└── convert_rl.py                                    # Qwen RLVR & GRPO dataset converter
```

---

## 🛠️ Tool Usage & CLI Commands

### 1. Download & Verify Dataset
```powershell
python rl_dataset/download.py
```

### 2. Run Deep Statistical Inspection
```powershell
python rl_dataset/inspect_dataset.py
```
Outputs detailed analysis to `rl_dataset/reports/dataset_inspection_report.md`.

### 3. Run Schema & Integrity Validation
```powershell
python rl_dataset/validation.py
```

### 4. Interactive Terminal Visualization
```powershell
# View 1st record
python rl_dataset/visualize.py --index 1

# View specific task
python rl_dataset/visualize.py --task telemetry-pipeline-audit-2 --full

# View by UUID
python rl_dataset/visualize.py --uuid <uuid>
```

### 5. Convert to Qwen RL / GRPO Format
```powershell
python rl_dataset/convert_rl.py --val-ratio 0.05
```

### 6. Run Test Suite
```powershell
pytest rl_dataset/tests -v
```
