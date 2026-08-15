# Glint-Research/Fable-5-traces Inspection Report

**Dataset Path:** `D:\fable5_qwen3.7\dataset\data\fable5_cot_merged.jsonl`  
**Tokenizer Evaluated:** `Qwen/Qwen3.8-27B`  
**Total Records:** `4665`  
**Unique Sessions:** `60`  

---

## 1. High-Level Summary

- **Tool Use Actions:** 3799 (81.44%)
- **Assistant Text Responses:** 866 (18.56%)
- **Mean Reasoning Tokens per Turn:** 632.0 tokens (Median: 543.0)
- **Mean Total Sequence Length:** 3254.1 tokens (Median: 3074.0)

---

## 2. Token Length Distribution (`Qwen/Qwen3.8-27B`)

| Field | Min | P25 | Median (P50) | P75 | P90 | P95 | P99 | Max | Mean ± Std |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Context (Prompt)** | 3 | 1992 | 2155 | 2322 | 2543 | 2743 | 3257 | 4576 | 2098 ± 527 |
| **CoT Reasoning (`<think>`)** | 73 | 419 | 543 | 718 | 1038 | 1333 | 2055 | 2475 | 632 ± 338 |
| **Target Action / Output** | 3 | 46 | 92 | 216 | 624 | 1195 | 4838 | 20414 | 343 ± 1133 |
| **Total Sequence Length** | 163 | 2774 | 3074 | 3487 | 4148 | 4825 | 8560 | 22348 | 3254 ± 1381 |

---

## 3. Context Window Truncation Budget

| Window Size | Fit Percentage | Samples Fit | Truncated Samples |
|:---|---:|---:|---:|
| **2048 TOKENS** | 6.52% | 304 | 4361 |
| **4096 TOKENS** | 89.45% | 4173 | 492 |
| **8192 TOKENS** | 98.84% | 4611 | 54 |
| **16384 TOKENS** | 99.85% | 4658 | 7 |
| **32768 TOKENS** | 100.0% | 4665 | 0 |
| **65536 TOKENS** | 100.0% | 4665 | 0 |
| **131072 TOKENS** | 100.0% | 4665 | 0 |

---

## 4. Tool Invocation Frequency

| Tool Name | Invocations | Share |
|:---|---:|---:|
| `Bash` | 1544 | 40.64% |
| `Edit` | 960 | 25.27% |
| `Read` | 443 | 11.66% |
| `Write` | 311 | 8.19% |
| `PowerShell` | 136 | 3.58% |
| `WebSearch` | 72 | 1.90% |
| `mcp__Claude_Preview__preview_eval` | 63 | 1.66% |
| `WebFetch` | 44 | 1.16% |
| `TaskUpdate` | 37 | 0.97% |
| `ToolSearch` | 35 | 0.92% |
| `TaskCreate` | 26 | 0.68% |
| `mcp__Claude_Preview__preview_screenshot` | 24 | 0.63% |
| `ScheduleWakeup` | 23 | 0.61% |
| `Monitor` | 13 | 0.34% |
| `mcp__Claude_Preview__preview_start` | 8 | 0.21% |
| `TaskStop` | 7 | 0.18% |
| `Skill` | 7 | 0.18% |
| `mcp__Claude_Preview__preview_console_logs` | 7 | 0.18% |
| `Grep` | 6 | 0.16% |
| `SendUserFile` | 5 | 0.13% |
| `StructuredOutput` | 5 | 0.13% |
| `Glob` | 4 | 0.11% |
| `Agent` | 3 | 0.08% |
| `mcp__Claude_Preview__preview_stop` | 3 | 0.08% |
| `mcp__Claude_Preview__preview_click` | 3 | 0.08% |
| `mcp__Claude_Preview__preview_resize` | 3 | 0.08% |
| `AskUserQuestion` | 2 | 0.05% |
| `Workflow` | 2 | 0.05% |
| `TaskOutput` | 1 | 0.03% |
| `mcp__huggingface__paper_search` | 1 | 0.03% |
| `mcp__Claude_Preview__preview_snapshot` | 1 | 0.03% |

---

## 5. Dataset Quality & Anomaly Checks

- **Missing / Empty CoT:** 0
- **Empty Context:** 0
- **Left-Truncated Contexts (Missing Initial User Prompt):** 2680
- **Broken JSON in Tool Arguments:** 0
- **Template Rendering Errors:** 0

> [!NOTE]
> When training with Qwen 2.5 / Qwen 3.8 chat templates, ensure that left-truncated contexts are handled by prepending a fallback user prompt or reconstructing full multi-turn sessions using `convert_sft.py`.
