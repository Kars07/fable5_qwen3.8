# 📊 Nemotron-RL-Agentic-Terminal-Pivot-v1 Inspection Report

> **Generated:** 2026-08-16 19:54:17  
> **Source Dataset:** `nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1`  
> **File Size:** `1308.88 MB` | **Total Decision Points:** `31,111`

---

## 1. Executive Summary

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Total Pivot Records** | `31,111` | Decision points extracted from successful trajectories |
| **Unique ATCB Tasks** | `630` | Distinct environment problem definitions |
| **Unique Source Trajectories** | `2,716` | End-to-end task executions |
| **Teacher Model** | `zai-org/GLM-5.1` | Model that generated the reference reasoning and actions |
| **Harness** | `terminus_2` | Evaluation environment harness |
| **Command Action Rate** | `91.16%` | Percentage of steps executing terminal commands |
| **Task Completion Rate** | `8.84%` | Terminal steps marking goal completion |

---

## 2. Functional Domain Distribution

| Domain Category | Samples | Proportion |
| :--- | :--- | :--- |
| **System Administration & Automation** | `13,923` | `44.8%` |
| **Data Pipelines & Storage** | `10,278` | `33.0%` |
| **Security & SIEM Audit** | `4,062` | `13.1%` |
| **Build, CI/CD & Testing** | `1,439` | `4.6%` |
| **Embedded, SCADA & Industrial** | `920` | `3.0%` |
| **Service Diagnostics & Crash Recovery** | `489` | `1.6%` |

---

## 3. Trajectory & Turn Distributions

| Distribution | Min | p25 | Median (p50) | p75 | p90 | p95 | p99 | Max | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pivot Turn Index** | `0` | `3` | `7` | `12` | `20` | `26` | `42` | `77` | `9.1` |
| **Total Turns in Trajectory** | `3` | `13` | `17` | `25` | `35` | `43` | `63` | `78` | `20.3` |
| **Input Messages in Prompt** | `1` | `7` | `15` | `25` | `41` | `53` | `85` | `155` | `19.1` |

---

## 4. Context & Token Length Distributions

| Text Field | Min Tokens | p25 | Median | p75 | p90 | p95 | p99 | Max Tokens | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full Prompt (Context)** | `833` | `3652` | `7397` | `13969` | `21603` | `26909` | `36797` | `73306` | `9967.9` |

---

## 5. Top 30 Shell Commands in Teacher Decisions

| Rank | Command | Total Occurrences | Description / Context |
| :--- | :--- | :--- | :--- |
| 1 | `cat` | `34,362` | Inspecting file contents, configs, source code |
| 2 | `ls` | `11,822` | Directory listings, file permission inspections |
| 3 | `python3` | `9,791` | Terminal utility operation |
| 4 | `head` | `7,411` | Sampling data files and large log headers |
| 5 | `echo` | `7,119` | Writing test inputs, config values |
| 6 | `1` | `6,411` | Terminal utility operation |
| 7 | `"` | `6,166` | Terminal utility operation |
| 8 | `grep` | `5,733` | Searching logs, codebase patterns, error traces |
| 9 | `sqlite3` | `4,440` | Terminal utility operation |
| 10 | `wc` | `3,294` | Terminal utility operation |
| 11 | `for` | `3,115` | Terminal utility operation |
| 12 | `find` | `2,962` | Locating files, configs, scripts across directory tree |
| 13 | `import` | `2,962` | Terminal utility operation |
| 14 | `#` | `2,956` | Terminal utility operation |
| 15 | `cd` | `2,835` | Terminal utility operation |
| 16 | `xxd` | `2,155` | Terminal utility operation |
| 17 | `sed` | `2,094` | Inline text edits, regex replacements in configs |
| 18 | `C-c` | `1,933` | Terminal utility operation |
| 19 | `redis-cli` | `1,892` | Terminal utility operation |
| 20 | `rm` | `1,875` | Cleaning corrupted caches, temporary artifacts |
| 21 | `with` | `1,752` | Terminal utility operation |
| 22 | `su` | `1,686` | Terminal utility operation |
| 23 | `psql` | `1,578` | Terminal utility operation |
| 24 | `'` | `1,443` | Terminal utility operation |
| 25 | `if` | `1,353` | Terminal utility operation |
| 26 | `tail` | `1,274` | Inspecting live service logs and error streams |
| 27 | `bash` | `1,230` | Terminal utility operation |
| 28 | `make` | `1,223` | Terminal utility operation |
| 29 | `EOF` | `1,176` | Terminal utility operation |
| 30 | `curl` | `1,058` | Testing HTTP endpoints, webhook telemetry |

---

## 6. Sample Decision Pivot Previews

### Sample 1: Task `telemetry-pipeline-audit-2` (Security & SIEM Audit)
- **UUID:** `t2_pre_rp_c0378f5f3c4c05a0868c9831e5698d2f`
- **Turn:** Step 0 of 41 (Prompt Messages: 1)
- **Task Complete:** `False`

**Teacher Analysis (`<think>`):**
```markdown
I'm at the project root directory. I need to explore the project structure, understand the existing code, and then implement the missing pipeline logic.
```

**Teacher Plan:**
```markdown
First, I'll explore the directory structure to understand what files exist, then read the key files to understand the current implementation and what needs to be done.
```

**Executed Keystrokes:**
```bash
find /app/ingest_pipeline -type f | head -50
find /data -type f | head -20
ls -la /app/reports/ 2>/dev/null || echo 'reports dir does not exist'
```

---
### Sample 2: Task `telemetry-pipeline-audit-2` (Security & SIEM Audit)
- **UUID:** `t2_pre_rp_22d7dbbdf741367b8e8da0a1c7c49101`
- **Turn:** Step 1 of 41 (Prompt Messages: 3)
- **Task Complete:** `False`

**Teacher Analysis (`<think>`):**
```markdown
I can see the project structure. There are several key files: config/security.json, cli.py, normalizer.py, plugins/auditor.py, plugins/validators.py, plugins/minimizer.py, run.py, and tests/test_smoke
```

**Teacher Plan:**
```markdown
Read all the existing source files to understand the current implementation, then look at some sample data files to understand the envelope format.
```

**Executed Keystrokes:**
```bash
cat /app/ingest_pipeline/config/security.json
cat /app/ingest_pipeline/__init__.py
cat /app/ingest_pipeline/cli.py
```

---
### Sample 3: Task `telemetry-pipeline-audit-2` (Security & SIEM Audit)
- **UUID:** `t2_pre_rp_87567f59afe083f9494c32ca44aafa7c`
- **Turn:** Step 2 of 41 (Prompt Messages: 5)
- **Task Complete:** `False`

**Teacher Analysis (`<think>`):**
```markdown
I've read the config file (contains crypto iterations and gadget memory boundaries), the empty __init__.py, and the stub files for cli.py and normalizer.py. I need to read the plugins and test files n
```

**Teacher Plan:**
```markdown
Continue reading the remaining source files - the plugins and the test suite - to understand what needs to be implemented.
```

**Executed Keystrokes:**
```bash
cat /app/ingest_pipeline/plugins/__init__.py
cat /app/ingest_pipeline/plugins/validators.py
cat /app/ingest_pipeline/plugins/minimizer.py
```

---
