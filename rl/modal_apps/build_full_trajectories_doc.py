"""Parse and compile full raw trajectory outputs for all Step 20 tasks into a comprehensive Markdown file."""

import json
from pathlib import Path

def build_report():
    base_dir = Path("D:/fable5_qwen3.7/eval_trajectories/tb2_eval_h200_step_20_1787045334")
    out_file = Path("D:/fable5_qwen3.7/STEP20_FULL_RAW_OUTPUTS_ALL_TASKS.md")
    artifact_file = Path("C:/Users/eniai/.gemini/antigravity-cli/brain/1fc3d7cd-3d3f-4a55-911c-39bfb7f8b975/step20_full_raw_outputs_all_tasks.md")
    
    tasks = sorted([d for d in base_dir.iterdir() if d.is_dir()], key=lambda d: d.name)
    
    lines = []
    lines.append("# 📑 Fable-5 RL Policy (Step 20): Master Dossier of All Tasks (1 to 36)")
    lines.append("")
    lines.append("**Model**: `Qwen/Qwen3.8-27B` + `step_20` Standalone PEFT Adapter")
    lines.append("**Benchmark**: Terminal-Bench 2.1 (`terminal-bench@2.0`) on Live H200 SXM")
    lines.append("**Total Evaluated Tasks**: 36 Tasks")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 📋 Table of Contents & Score Summary")
    lines.append("")
    lines.append("| # | Task Name | Reward / Verdict | Turns | Prompt Tokens | Output Tokens | Total Tokens |")
    lines.append("| :-: | :--- | :---: | :---: | :---: | :---: | :---: |")

    task_summaries = []
    
    for idx, t_dir in enumerate(tasks, 1):
        res_file = t_dir / "result.json"
        traj_file = t_dir / "trajectory.json"
        
        reward = "N/A"
        n_input_tokens = 0
        n_output_tokens = 0
        turns = 0
        
        if res_file.exists():
            try:
                res_data = json.loads(res_file.read_text(encoding="utf-8"))
                agent_res = res_data.get("agent_result", {})
                reward_val = agent_res.get("reward", None)
                if reward_val is not None:
                    reward = f"`1.0 PASS`" if reward_val == 1.0 else f"`0.0 FAIL`"
                n_input_tokens = agent_res.get("n_input_tokens", 0)
                n_output_tokens = agent_res.get("n_output_tokens", 0)
            except Exception:
                pass
                
        if traj_file.exists():
            try:
                traj_data = json.loads(traj_file.read_text(encoding="utf-8"))
                step_list = traj_data.get("steps", []) if isinstance(traj_data, dict) else traj_data
                turns = len(step_list)
            except Exception:
                pass
                
        tot_tok = n_input_tokens + n_output_tokens
        t_name = t_dir.name
        clean_name = t_name.split("__")[0]
        
        lines.append(f"| {idx} | [{clean_name}](#task-{idx}-{clean_name}) | {reward} | {turns} | {n_input_tokens:,} | {n_output_tokens:,} | {tot_tok:,} |")
        task_summaries.append((idx, t_name, clean_name, reward, turns, n_input_tokens, n_output_tokens, tot_tok, t_dir))
        
    lines.append("")
    lines.append("---")
    lines.append("")
    
    for idx, t_name, clean_name, reward, turns, in_tok, out_tok, tot_tok, t_dir in task_summaries:
        traj_file = t_dir / "trajectory.json"
        lines.append(f"# <a id=\"task-{idx}-{clean_name}\"></a>Task {idx}: `{clean_name}`")
        lines.append(f"* **Full Benchmark ID**: `{t_name}`")
        lines.append(f"* **Official Result**: {reward}")
        lines.append(f"* **Turns**: **{turns} Turns** | **Prompt Tokens**: {in_tok:,} | **Output Tokens**: {out_tok:,} | **Total Tokens**: {tot_tok:,}")
        lines.append("")
        
        if not traj_file.exists():
            lines.append("> No trajectory data available.")
            lines.append("")
            continue
            
        try:
            traj_data = json.loads(traj_file.read_text(encoding="utf-8"))
            step_list = traj_data.get("steps", []) if isinstance(traj_data, dict) else traj_data
            
            for s in step_list:
                src = s.get("source")
                step_id = s.get("step_id")
                
                if src == "agent":
                    msg = s.get("message", "").strip()
                    tool_calls = s.get("tool_calls", [])
                    lines.append(f"### 🤖 Turn {step_id}: Model Generation")
                    if msg:
                        lines.append("```xml")
                        lines.append(msg)
                        lines.append("```")
                    if tool_calls:
                        lines.append("**Executed Tool Calls:**")
                        for tc in tool_calls:
                            fn = tc.get("function_name", "bash_command")
                            args = tc.get("arguments", {})
                            ks = args.get("keystrokes", "")
                            lines.append(f"* **`{fn}`**:")
                            if ks:
                                lines.append("```bash")
                                lines.append(ks.strip())
                                lines.append("```")
                    lines.append("")
                elif src == "user" and step_id > 1:
                    obs = s.get("observation", {})
                    content = ""
                    if isinstance(obs, dict) and "results" in obs:
                        for r in obs["results"]:
                            content += r.get("content", "") + "\n"
                    elif isinstance(obs, str):
                        content = obs
                    if content.strip():
                        # Truncate very long terminal output
                        trunc = content.strip()
                        if len(trunc) > 1200:
                            trunc = trunc[:1200] + f"\n... [Truncated {len(content) - 1200} characters] ..."
                        lines.append(f"<details><summary>💻 Turn {step_id}: Terminal Output Observation</summary>")
                        lines.append("")
                        lines.append("```text")
                        lines.append(trunc)
                        lines.append("```")
                        lines.append("</details>")
                        lines.append("")
        except Exception as e:
            lines.append(f"> Error parsing trajectory: {e}")
            lines.append("")
            
        lines.append("---")
        lines.append("")
        
    doc_content = "\n".join(lines)
    out_file.write_text(doc_content, encoding="utf-8")
    artifact_file.write_text(doc_content, encoding="utf-8")
    print(f"Successfully generated master report with {len(tasks)} tasks at:\n  - {out_file}\n  - {artifact_file}")
    print(f"Total document length: {len(doc_content):,} characters / {len(lines):,} lines.")

if __name__ == "__main__":
    build_report()
