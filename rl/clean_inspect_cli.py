"""Clean, beautiful CLI inspector for Terminal-Bench 2.1 Step 20 evaluation results."""

import sys
import json
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    base_dir = Path("D:/fable5_qwen3.7/eval_trajectories/tb2_eval_h200_step_20_1787045334")
    if not base_dir.exists():
        print(f"[!] Directory not found: {base_dir}")
        return

    tasks = sorted([d for d in base_dir.iterdir() if d.is_dir()], key=lambda d: d.name)
    
    passed_count = 0
    total_count = len(tasks)
    total_in_tokens = 0
    total_out_tokens = 0
    total_turns = 0

    rows = []

    for idx, t_dir in enumerate(tasks, 1):
        res_file = t_dir / "result.json"
        traj_file = t_dir / "trajectory.json"

        reward_str = "0.0 FAIL"
        is_pass = False
        in_tok = 0
        out_tok = 0
        turns = 0

        if res_file.exists():
            try:
                res_data = json.loads(res_file.read_text(encoding="utf-8"))
                
                # Check verifier_result rewards (Official Harbor Ground Truth)
                ver_res = res_data.get("verifier_result", {})
                ver_rewards = ver_res.get("rewards", {}) if isinstance(ver_res, dict) else {}
                r_val = ver_rewards.get("reward", None)
                
                # Fallback to agent_result if not in verifier_result
                if r_val is None:
                    r_val = res_data.get("agent_result", {}).get("reward", 0.0)

                if r_val == 1.0:
                    reward_str = "1.0 PASS"
                    is_pass = True
                    passed_count += 1
                    
                agent_res = res_data.get("agent_result", {})
                in_tok = agent_res.get("n_input_tokens", 0)
                out_tok = agent_res.get("n_output_tokens", 0)
            except Exception:
                pass

        if traj_file.exists():
            try:
                traj_data = json.loads(traj_file.read_text(encoding="utf-8"))
                steps = traj_data.get("steps", []) if isinstance(traj_data, dict) else traj_data
                turns = len(steps)
            except Exception:
                pass

        tot_tok = in_tok + out_tok
        total_in_tokens += in_tok
        total_out_tokens += out_tok
        total_turns += turns
        
        clean_name = t_dir.name.split("__")[0]
        rows.append((idx, clean_name, reward_str, is_pass, turns, in_tok, out_tok, tot_tok))

    # Header
    print("\n" + "=" * 105)
    print(" [*] FABLE-5 RL POLICY (STEP 20) -- TERMINAL-BENCH 2.1 EVALUATION INSPECTOR")
    print("=" * 105)
    print(f" {'#':<3} | {'TASK NAME':<36} | {'STATUS':<10} | {'TURNS':<5} | {'PROMPT TOK':<12} | {'OUTPUT TOK':<11} | {'TOTAL TOK':<12}")
    print("-" * 105)

    for idx, name, status, is_pass, turns, in_tok, out_tok, tot_tok in rows:
        symbol = "[PASS]" if is_pass else "[FAIL]"
        in_str = f"{in_tok:,}"
        out_str = f"{out_tok:,}"
        tot_str = f"{tot_tok:,}"
        print(f" {idx:>2}. | {name:<36} | {symbol:<10} | {turns:>5} | {in_str:>12} | {out_str:>11} | {tot_str:>12}")

    print("=" * 105)
    win_rate = (passed_count / max(1, total_count)) * 100
    tot_in_str = f"{total_in_tokens:,}"
    tot_out_str = f"{total_out_tokens:,}"
    tot_all_str = f"{(total_in_tokens + total_out_tokens):,}"
    print(f" [=] OFFICIAL BENCHMARK SUMMARY:")
    print(f"     * Total Tasks Evaluated:   {total_count}")
    print(f"     * Total Passed Tasks:      {passed_count} / {total_count} ({win_rate:.1f}% Verified Pass Rate)")
    print(f"     * Total Interactive Turns: {total_turns:,}")
    print(f"     * Total Prompt Tokens:     {tot_in_str}")
    print(f"     * Total Output Tokens:     {tot_out_str}")
    print(f"     * Cumulative All Tokens:   {tot_all_str}")
    print("=" * 105 + "\n")

if __name__ == "__main__":
    main()
