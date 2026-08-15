# Fable-5 Traces Dataset Inspection & SFT Suite

A complete inspection, validation, conversion, and DataLoader suite for **[Glint-Research/Fable-5-traces](https://huggingface.co/datasets/Glint-Research/Fable-5-traces)** tailored for Supervised Fine-Tuning (SFT) of reasoning coding agents using **`Qwen/Qwen3.8-27B`** (and Qwen 2.5 / 3 series models).

---

## 📊 Dataset Key Findings (`Qwen/Qwen3.8-27B`)

| Metric | Measurement / Value |
|:---|:---|
| **Total Examples** | **4,665** interaction turns |
| **Unique Source Sessions** | **60** complete agent trajectories |
| **Tool-Use Actions** | **3,799 (81.44%)** |
| **Final Text Responses** | **866 (18.56%)** |
| **Reasoning Trace Length (`<think>`)** | **Median: 543 tokens**, Mean: 632 tokens, P90: 1,038 tokens, Max: 2,475 tokens |
| **Context (Prompt) Length** | **Median: 2,155 tokens**, Mean: 2,098 tokens, P90: 2,543 tokens, Max: 4,576 tokens |
| **Action / Output Length** | **Median: 92 tokens**, Mean: 343 tokens, P90: 624 tokens, Max: 20,414 tokens (code write) |
| **Total Sequence Length** | **Median: 3,074 tokens**, Mean: 3,254 tokens, P90: 4,148 tokens, Max: 22,348 tokens |
| **4,096 Token Budget Fit** | **89.45%** of records fit without truncation |
| **8,192 Token Budget Fit** | **98.84%** of records fit without truncation |
| **16,384 Token Budget Fit** | **99.85%** of records fit without truncation |

### Top Tools Invoked
1. **`Bash`** (1,544 calls — 40.64%)
2. **`Edit`** (960 calls — 25.27%)
3. **`Read`** (443 calls — 11.66%)
4. **`Write`** (311 calls — 8.19%)
5. **`PowerShell`** (136 calls — 3.58%)
6. **`WebSearch`** (72 calls — 1.90%)
7. **`Claude_Preview` / Web Tools** (100+ calls)

---

## 📁 Repository Structure

```
dataset/
├── download.py              # Download raw dataset from Hugging Face
├── parser.py                # Core context parser, normalizer, and message converter
├── inspect_dataset.py       # Rich CLI dataset inspector, quantile profiler & sample viewer
├── convert_sft.py           # Convert raw traces to Qwen Native & ChatML SFT formats
├── validation.py            # Dataset structure validator & PyTorch SFTDataset class
├── visualize.py             # Generates publication-quality charts & histograms
├── test_dataset_loader.py   # PyTorch DataLoader & loss-masking verification test
├── data/
│   ├── fable5_cot_merged.jsonl      # Local copy of raw dataset
│   └── converted/
│       ├── fable5_sft_qwen_native.jsonl    # Qwen native format with reasoning_content & tool_calls
│       ├── fable5_sft_chatml_inline.jsonl  # ChatML inline <think> format
│       └── fable5_sft_sessions.jsonl       # 60 full reconstructed multi-turn sessions
└── reports/
    ├── inspection_report.json       # Detailed statistical JSON metrics
    ├── inspection_report.md         # Full Markdown inspection documentation
    └── figures/
        ├── cumulative_context_coverage.png
        ├── tool_usage_distribution.png
        ├── output_type_breakdown.png
        └── token_length_profiling.png
```

---

## 🚀 How to Use

### 1. Download Dataset
Download the merged JSONL and sample traces to `dataset/data/`:
```bash
.venv\Scripts\python.exe dataset\download.py
```

### 2. Inspect Dataset & View Quantiles
Run comprehensive profiling using the `Qwen/Qwen3.8-27B` tokenizer:
```bash
.venv\Scripts\python.exe dataset\inspect_dataset.py
```

To render and inspect a single sample with colorized syntax highlighting (`USER`, `<think>`, `<tool_call>`, `TOOL RESULT`):
```bash
.venv\Scripts\python.exe dataset\inspect_dataset.py --view 0
```

### 3. Convert to SFT Formats
Generate clean train-ready SFT datasets:
```bash
.venv\Scripts\python.exe dataset\convert_sft.py --tokenizer Qwen/Qwen3.8-27B
```
This generates:
- **`fable5_sft_qwen_native.jsonl`**: Standard Qwen 3.8 format with `reasoning_content` and structured `tool_calls`.
- **`fable5_sft_chatml_inline.jsonl`**: Inline `<think>...</think>` tags compatible with Unsloth / LLaMA-Factory / TRL.
- **`fable5_sft_sessions.jsonl`**: Full reconstructed multi-turn trajectories across 60 complete coding sessions.

### 4. Validate Dataset Integrity & Sequence Lengths
Run validation checks on any dataset file:
```bash
.venv\Scripts\python.exe dataset\validation.py --file dataset/data/converted/fable5_sft_qwen_native.jsonl --max-len 4096
```

### 5. Generate Visual Charts
Produce charts in `dataset/reports/figures/`:
```bash
.venv\Scripts\python.exe dataset\visualize.py
```

### 6. Test PyTorch Training DataLoader
Verify batch collation, padding, and loss masking:
```bash
.venv\Scripts\python.exe dataset\test_dataset_loader.py
```

---

## 💡 Important SFT Training Nuances for Qwen 3.8 / Qwen 2.5

1. **Left-Truncated Context Handling**:
   In the raw merged dataset, long transcripts were left-truncated to fit maximum context limits, dropping the initial `USER:` prompt on ~57% of rows. The parser in `parser.py` automatically handles this by inserting a coherent fallback user prompt or merging full session trajectories so that chat templates render with 0 errors.

2. **Loss Masking for Reasoning (`<think>`) and Actions**:
   In SFT, **both** the reasoning trace (`<think>...</think>`) and the target action (`<tool_call>` or final response) are supervised (labels != -100). The user prompts and previous tool outputs in context are masked with label `-100`.

3. **Recommended Sequence Length**:
   - **4,096 tokens**: Fits **89.5%** of all turns (fastest training).
   - **8,192 tokens**: Fits **98.8%** of all turns (recommended sweet spot for full code edits).
   - **16,384+ tokens**: Recommended if training on `fable5_sft_sessions.jsonl` (full multi-turn sessions).
