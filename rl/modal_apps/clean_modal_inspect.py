"""Accurate inspection filtering only the 33 tasks that actually executed turns."""

import modal
import json
from pathlib import Path

app = modal.App("accurate-eval-inspector")
image = modal.Image.debian_slim(python_version="3.12")
volume_outputs = modal.Volume.from_name("fable5-terminalbench-outputs-h200")

@app.function(image=image, volumes={"/outputs": volume_outputs})
def inspect_actual_executed_tasks():
    target_dir = Path("/outputs/tb2_eval_h200_step_20_1787045334/harbor_output")
    job_dirs = sorted([d for d in target_dir.iterdir() if d.is_dir()], key=lambda d: d.name)
    if not job_dirs:
        print("[!] No job directories found.")
        return

    latest_job = job_dirs[-1]
    all_task_dirs = sorted([d for d in latest_job.iterdir() if d.is_dir()], key=lambda d: d.name)

    executed_tasks = []
    
    for t_dir in all_task_dirs:
        res_file = t_dir / "result.json"
        traj_file = t_dir / "agent" / "trajectory.json"
        
        step_count = 0
        steps = []
        if traj_file.exists():
            try:
                traj_data = json.loads(traj_file.read_text(encoding="utf-8"))
                steps = traj_data.get("steps", []) if isinstance(traj_data, dict) else traj_data
                step_count = len(steps)
            except Exception:
                pass

        # If 0 steps and no trajectory, this was an unattempted queued folder from when we stopped the run
        if step_count == 0:
            continue

        reward = 0.0
        in_tok = 0
        out_tok = 0
        if res_file.exists():
            try:
                res_data = json.loads(res_file.read_text(encoding="utf-8"))
                ver_res = res_data.get("verifier_result", {})
                ver_rewards = ver_res.get("rewards", {}) if isinstance(ver_res, dict) else {}
                r_val = ver_rewards.get("reward", None)
                if r_val is None:
                    r_val = res_data.get("agent_result", {}).get("reward", 0.0)
                reward = float(r_val) if r_val is not None else 0.0

                agent_res = res_data.get("agent_result", {})
                in_tok = agent_res.get("n_input_tokens", 0)
                out_tok = agent_res.get("n_output_tokens", 0)
            except Exception:
                pass

        clean_name = t_dir.name.split("__")[0]
        executed_tasks.append({
            "dir_name": t_dir.name,
            "clean_name": clean_name,
            "reward": reward,
            "turns": step_count,
            "in_tok": in_tok,
            "out_tok": out_tok,
            "tot_tok": in_tok + out_tok,
            "steps": steps,
        })

    print(f"\n{'='*105}")
    print(f" [*] ACCURATE TERMINAL-BENCH 2.1 EVALUATION INSPECTION (EXACTLY {len(executed_tasks)} EXECUTED TASKS)")
    print(f"{'='*105}")
    print(f" {'#':<3} | {'TASK NAME':<36} | {'STATUS':<10} | {'TURNS':<5} | {'PROMPT TOK':<12} | {'OUTPUT TOK':<11} | {'TOTAL TOK':<12}")
    print(f"{'-'*105}")

    passed = 0
    total_prompt = 0
    total_output = 0
    total_turns = 0

    for idx, t in enumerate(executed_tasks, 1):
        status = "[PASS]" if t["reward"] == 1.0 else "[FAIL]"
        if t["reward"] == 1.0:
            passed += 1
        total_prompt += t["in_tok"]
        total_output += t["out_tok"]
        total_turns += t["turns"]

        in_s = f"{t['in_tok']:,}"
        out_s = f"{t['out_tok']:,}"
        tot_s = f"{t['tot_tok']:,}"
        print(f" {idx:>2}. | {t['clean_name']:<36} | {status:<10} | {t['turns']:>5} | {in_s:>12} | {out_s:>11} | {tot_s:>12}")

    print(f"{'='*105}")
    win_rate = (passed / len(executed_tasks)) * 100
    print(f" [=] ACCURATE BENCHMARK REWARD STATS:")
    print(f"     * Total Attempted Tasks:   {len(executed_tasks)}")
    print(f"     * Total Verified Passed:   {passed} / {len(executed_tasks)} ({win_rate:.1f}% Ground-Truth Pass Rate)")
    print(f"     * Total Interactive Turns: {total_turns:,}")
    print(f"     * Total Prompt Tokens:     {total_prompt:,}")
    print(f"     * Total Output Tokens:     {total_output:,}")
    print(f"     * Cumulative All Tokens:   {(total_prompt + total_output):,}")
    print(f"{'='*105}\n")

@app.local_entrypoint()
def main():
    inspect_actual_executed_tasks.remote()
