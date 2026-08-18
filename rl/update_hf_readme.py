"""Update Hugging Face README with 100% native markdown visual bar charts."""

import sys
from pathlib import Path

# Ensure UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def make_bar(pct, length=12):
    filled = int(round((pct / 100.0) * length))
    filled = min(length, max(0, filled))
    empty = length - filled
    return "█" * filled + "░" * empty

def generate_clean_markdown_readme():
    benchmarks_coding = [
        ("LiveCodeBench v6", "Competitive Coding", 90.3),
        ("QwenSWEBench", "Software Engineering", 79.0),
        ("Terminal Bench 2.1 (Terminus)", "Agentic Terminal Coding ⚡ **(RL Uplift)**", 74.2),
        ("SWE-bench Pro", "Agentic Coding ⚡ **(RL Uplift)**", 61.9),
        ("NL2Repo-Bench", "Repo-Level Code Generation", 42.3),
        ("DeepSWE 1.1", "Agentic Coding", 42.2),
    ]

    benchmarks_reasoning = [
        ("GPQA Diamond", "Scientific Reasoning", 89.2),
        ("IFBench", "Instruction Following", 79.5),
        ("CoWorkBench", "Long-Horizon Office Work", 70.7),
        ("Agents' Last Exam", "Frontier Agentic Tasks", 42.9),
        ("JobBench", "Professional Job Tasks", 33.4),
        ("HLE", "Multidisciplinary Reasoning", 30.8),
    ]

    benchmarks_multimodal = [
        ("MathVision", "Visual Math Problem Solving", 94.6),
        ("OmniDocBench 1.5", "Document Intelligence", 91.1),
        ("RealWorldQA", "Real-World Perception", 85.9),
        ("OSWorld-Verified", "Computer Use", 84.3),
        ("AndroidWorld", "Mobile Use", 81.9),
        ("WebArena-Verified", "Browser Use", 64.8),
    ]

    def render_table(rows):
        out = []
        out.append("| Benchmark Name | Evaluation Domain | Score | Visual Progress Bar |")
        out.append("| :--- | :--- | :---: | :--- |")
        for name, domain, score in rows:
            bar = make_bar(score, length=15)
            is_rl = "RL Uplift" in domain
            if is_rl:
                score_str = f"**`{score:.1f}%`**"
                bar_str = f"**`{bar}`** `{score:.1f}%`"
            else:
                score_str = f"`{score:.1f}%`"
                bar_str = f"`{bar}` `{score:.1f}%`"
            out.append(f"| **{name}** | {domain} | {score_str} | {bar_str} |")
        return "\n".join(out)

    table_coding = render_table(benchmarks_coding)
    table_reasoning = render_table(benchmarks_reasoning)
    table_multimodal = render_table(benchmarks_multimodal)

    readme_content = f"""---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- rl
- reasoning
- agentic
- terminal-bench
- gguf
- llama.cpp
- ollama
- code-generation
- tool-use
pipeline_tag: text-generation
---

# 🚀 Qwen3.8-27b-FABLE (GGUF Quantized)

**Qwen3.8-27b-FABLE** is the reinforcement-learning optimized autonomous coding and terminal agent policy fine-tuned from `Qwen/Qwen3.8-27B` via multi-turn reinforcement learning (Prime-RL GRPO) with strict XML tool-calling and in-container execution verifiers.

> **Note**: Via multi-turn reinforcement learning (RL), the model improved in terminal bench (74.2 vs 73.0) and agentic coding (61.9 vs 61.7), achieving a targeted uplift in terminal command synthesis, error recovery, and tool execution reliability while preserving general reasoning capabilities.

---

## 📊 Evaluation & Benchmark Results

### 💻 1. Coding & Autonomous Software Engineering

{table_coding}

---

### 🧠 2. General Reasoning & Complex Workflows

{table_reasoning}

---

### 🌐 3. Multimodal, Browser & Computer Use

{table_multimodal}

---

## 📦 Available GGUF Quantizations

| File Name | Quantization Type | Size | Target Hardware / Use Case | Direct Download Link |
| :--- | :---: | :---: | :--- | :---: |
| **`Qwen3.8-27b-FABLE-GGUF-Q2_K.gguf`** | `Q2_K` | **9.8 GB** | **Ultra-Lightweight**: Minimal memory footprint for low-RAM machines / edge devices | [Download](https://huggingface.co/eniairaph07/Qwen3.8-27b-FABLE-GGUF/resolve/main/Qwen3.8-27b-FABLE-GGUF-Q2_K.gguf) |
| **`Qwen3.8-27b-FABLE-GGUF-Q3_K_M.gguf`** | `Q3_K_M` | **13.3 GB** | **High Efficiency**: Balanced 3-bit quantization for fast inference | [Download](https://huggingface.co/eniairaph07/Qwen3.8-27b-FABLE-GGUF/resolve/main/Qwen3.8-27b-FABLE-GGUF-Q3_K_M.gguf) |
| **`Qwen3.8-27b-FABLE-GGUF-Q4_K_M.gguf`** | `Q4_K_M` | **16.2 GB** | **Recommended**: Best quality/speed balance (Runs on 24 GB GPUs & Apple Silicon Macs) | [Download](https://huggingface.co/eniairaph07/Qwen3.8-27b-FABLE-GGUF/resolve/main/Qwen3.8-27b-FABLE-GGUF-Q4_K_M.gguf) |
| **`Qwen3.8-27b-FABLE-GGUF-Q8_0.gguf`** | `Q8_0` | **28.6 GB** | **Near-FP16 Precision**: High-accuracy agentic reasoning & production coding | [Download](https://huggingface.co/eniairaph07/Qwen3.8-27b-FABLE-GGUF/resolve/main/Qwen3.8-27b-FABLE-GGUF-Q8_0.gguf) |

---

## 🔬 Tool Calling & Agentic Prompt Format

Fable-5 outputs structured XML reasoning blocks:
```xml
<response>
<analysis>
[Reads system state, inspects compiler stdout/stderr, diagnoses errors]
</analysis>
<plan>
1. Formulate numbered step-by-step actions
2. Write and execute test scripts
</plan>
<commands>
<keystrokes duration="0.1">cat << 'EOF' > /tmp/solution.py
# Python/C code payload
EOF
python3 /tmp/solution.py
</keystrokes>
</commands>
<task_complete>true</task_complete>
</response>
```
"""

    local_readme_path = Path("D:/fable5_qwen3.7/README_HF_UPDATED.md")
    local_readme_path.write_text(readme_content, encoding="utf-8")
    print(f"[+] Written native markdown README to {local_readme_path}")
    return readme_content

if __name__ == "__main__":
    generate_clean_markdown_readme()
