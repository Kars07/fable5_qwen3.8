"""Side-by-Side 5-Task Benchmark Evaluation: Base Qwen3.8-27B vs. Fable-5 RL Policy (Step 20).

Evaluates the exact same 5 Terminal-Bench 2.0 tasks using Harbor & Terminus 2:
1. Run Phase 1: Base Model (Qwen/Qwen3.8-27B) on 5 tasks
2. Run Phase 2: Fable-5 Policy (Step 20 LoRA) on 5 tasks
3. Generates complete side-by-side scorecard & behavioral diffs.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any
import urllib.request
import modal

app = modal.App("fable5-tb2-side-by-side-eval")

# Persistent Cloud Volumes (kars07 profile)
volume_hf = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
volume_eval_outputs = modal.Volume.from_name("fable5-terminalbench-outputs-h200", create_if_missing=True)
volume_prime_outputs = modal.Volume.from_name("fable5-prime-rl-outputs", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "tmux", "sqlite3", "xxd", "wget", "psmisc")
    .pip_install(
        "vllm>=0.8.0",
        "huggingface_hub>=0.24.0",
        "peft>=0.11.0",
        "transformers>=4.48.0",
        "rich>=13.7.0",
        "tabulate>=0.9.0",
        "numpy>=1.26.0",
        "e2b>=2.35.0",
        "harbor==0.20.0",
        "hf_transfer>=0.1.8",
    )
    .add_local_dir("rl", remote_path="/opt/rl", copy=True, ignore=lambda p: "__pycache__" in str(p) or str(p).endswith(".pyc"))
    .add_local_dir("verifiers", remote_path="/opt/verifiers", copy=True, ignore=lambda p: "__pycache__" in str(p) or str(p).endswith(".pyc"))
    .run_commands(
        "python3 -c \"import harbor.environments.e2b as m, pathlib; p = pathlib.Path(m.__file__); p.write_text(p.read_text().replace('timeout=86_400', 'timeout=3600'))\""
    )
    .env({
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/opt/rl:/opt/verifiers",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)


def _wait_for_vllm_server(url: str = "http://127.0.0.1:8000/v1/models", proc: subprocess.Popen | None = None, log_path: Path | None = None, timeout: int = 1200) -> None:
    """Poll vLLM server endpoint until ready, streaming logs periodically and failing fast on exit."""
    start = time.time()
    last_print = 0
    while time.time() - start < timeout:
        if proc and proc.poll() is not None:
            log_content = log_path.read_text(encoding="utf-8", errors="replace") if log_path and log_path.exists() else ""
            raise RuntimeError(f"vLLM server exited prematurely with code {proc.returncode}:\n{log_content}")

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print("\n[+] vLLM OpenAI-compatible server is READY and responding on port 8000!", flush=True)
                    return
        except Exception:
            pass

        now = time.time()
        if log_path and log_path.exists() and (now - last_print > 15):
            last_print = now
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = lines[-5:] if len(lines) >= 5 else lines
            print(f"[*] Waiting for vLLM ({int(now - start)}s elapsed)... Latest log:\n" + "\n".join(f"    | {l}" for l in tail), flush=True)

        time.sleep(5)

    if log_path and log_path.exists():
        print("[!] FULL vLLM SERVER LOG ON TIMEOUT:\n" + log_path.read_text(encoding="utf-8", errors="replace"), flush=True)
    raise TimeoutError(f"vLLM server on {url} failed to start within {timeout}s")


def _run_single_eval_phase(
    model_mode: str, # "base" or "step_20"
    num_tasks: int,
    dataset: str,
    harness: str,
    out_base: Path,
    token: str,
    e2b_key: str,
    console: Any,
) -> dict[str, Any]:
    """Boot vLLM for the given mode, run Harbor on N tasks, and extract results."""
    phase_dir = out_base / f"eval_{model_mode}_{int(time.time())}"
    phase_dir.mkdir(parents=True, exist_ok=True)
    vllm_log = phase_dir / "vllm_server.log"

    model_name = "Qwen/Qwen3.8-27B"
    vllm_lora_args = []

    if model_mode != "base":
        model_name = "eval_policy"
        lora_dir = Path(f"/cache/huggingface/checkpoints/{model_mode}")
        lora_dir.mkdir(parents=True, exist_ok=True)

        # Check local volume first, then HF
        local_step_dir = Path(f"/workspace/prime_outputs/prime-rl-run/checkpoints/{model_mode}")
        if not local_step_dir.exists():
            local_step_dir = Path(f"/workspace/prime_outputs/checkpoints/{model_mode}")

        if local_step_dir.exists() and not (lora_dir / "adapter_model.safetensors").exists():
            console.print(f"[*] Copying local checkpoint from {local_step_dir}...")
            shutil.copytree(str(local_step_dir), str(lora_dir), dirs_exist_ok=True)
        elif not (lora_dir / "adapter_model.safetensors").exists():
            from huggingface_hub import snapshot_download
            console.print(f"[*] Syncing standalone adapter for {model_mode} from HF...")
            snapshot_download(
                repo_id="eniairaph07/qwen3.8-27b-fable5-rl-sft-steps",
                allow_patterns=[f"rl_checkpoints/{model_mode}/*"],
                local_dir="/tmp/hf_download",
                token=token,
            )
            src_dir = Path(f"/tmp/hf_download/rl_checkpoints/{model_mode}")
            if src_dir.exists():
                shutil.copytree(str(src_dir), str(lora_dir), dirs_exist_ok=True)

        # Standardize & sanitize PEFT keys if needed
        adapter_path = lora_dir / "adapter_model.safetensors"
        if adapter_path.exists():
            from safetensors.torch import load_file, save_file
            import torch
            raw_weights = load_file(str(adapter_path))
            needs_clean = any("app.model" in k or k.endswith(".0") or "language_model" in k for k in raw_weights.keys())
            if needs_clean:
                clean_state_dict = {}
                for k, v in raw_weights.items():
                    clean_k = k
                    if any(opt in clean_k.lower() for opt in ["optimizer", "optimizers", "exp_avg", "step", "state."]):
                        continue
                    for prefix in [
                        "base_model.model.app.model.model.language_model.",
                        "base_model.model.app.model.language_model.",
                        "app.model.model.language_model.",
                        "app.model.language_model.",
                        "app.model.",
                        "language_model.",
                    ]:
                        if clean_k.startswith(prefix):
                            clean_k = clean_k[len(prefix):]
                            break
                    if clean_k.endswith(".lora_A.0") or clean_k.endswith(".lora_A.default.0"):
                        clean_k = clean_k.split(".lora_A")[0] + ".lora_A.weight"
                    elif clean_k.endswith(".lora_B.0") or clean_k.endswith(".lora_B.default.0"):
                        clean_k = clean_k.split(".lora_B")[0] + ".lora_B.weight"
                    elif clean_k.endswith(".0"):
                        clean_k = clean_k[:-2] + ".weight"

                    if not clean_k.startswith("base_model.model."):
                        if clean_k.startswith("layers."):
                            clean_k = f"base_model.model.model.{clean_k}"
                        elif clean_k.startswith("model.layers."):
                            clean_k = f"base_model.model.{clean_k}"
                        else:
                            clean_k = f"base_model.model.model.{clean_k}"

                    clean_state_dict[clean_k] = v.contiguous().to(torch.bfloat16)

                save_file(clean_state_dict, str(adapter_path))
                lora_cfg = {
                    "base_model_name_or_path": "Qwen/Qwen3.8-27B",
                    "bias": "none",
                    "inference_mode": True,
                    "peft_type": "LORA",
                    "r": 64,
                    "lora_alpha": 128.0,
                    "lora_dropout": 0.05,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    "task_type": "CAUSAL_LM",
                }
                (lora_dir / "adapter_config.json").write_text(json.dumps(lora_cfg, indent=2), encoding="utf-8")

        vllm_lora_args = [
            "--enable-lora",
            "--lora-modules", f"{model_name}={str(lora_dir)}",
            "--max-lora-rank", "64",
        ]

    console.print(f"\n[bold cyan]{'='*30} STARTING vLLM ({model_mode.upper()}) {'='*30}[/bold cyan]")
    vllm_cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "Qwen/Qwen3.8-27B",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.95",
        "--max-model-len", "131072",
        "--max-num-seqs", "64",
        "--enforce-eager",
        "--dtype", "bfloat16",
        "--port", "8000",
        "--trust-remote-code",
        "--download-dir", "/cache/huggingface",
    ] + vllm_lora_args

    with vllm_log.open("w", encoding="utf-8") as handle:
        vllm_proc = subprocess.Popen(vllm_cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)

    try:
        _wait_for_vllm_server("http://127.0.0.1:8000/v1/models", proc=vllm_proc, log_path=vllm_log, timeout=1200)

        harbor_out = phase_dir / "harbor_output"
        harbor_cmd = [
            "harbor", "run",
            "-d", dataset,
            "-m", f"openai/{model_name}",
            "-a", harness,
            "--env", "e2b",
            "--ak", "parser_name=xml",
            "--ak", "enable_summarize=false",
            "--ak", "proactive_summarization_threshold=0",
            "--jobs-dir", str(harbor_out),
            "-n", str(num_tasks),
        ]

        console.print(f"[*] Running Harbor evaluation ({num_tasks} tasks) for {model_mode.upper()}...")
        env_vars = os.environ.copy()
        env_vars["OPENAI_API_BASE"] = "http://127.0.0.1:8000/v1"
        env_vars["OPENAI_API_KEY"] = "EMPTY"
        env_vars["E2B_API_KEY"] = e2b_key
        env_vars["TERM"] = "xterm-256color"

        subprocess.run(harbor_cmd, check=True, env=env_vars)

        # Parse results
        job_dirs = sorted([d for d in harbor_out.iterdir() if d.is_dir()], key=lambda d: d.name)
        task_results = {}
        if job_dirs:
            latest_job = job_dirs[-1]
            for td in latest_job.iterdir():
                if td.is_dir():
                    task_name = td.name.split("__")[0]
                    res_file = td / "result.json"
                    traj_file = td / "agent" / "trajectory.json"
                    reward = 0.0
                    steps = 0
                    tokens = 0
                    if res_file.exists():
                        try:
                            res_data = json.loads(res_file.read_text(encoding="utf-8"))
                            reward = float(res_data.get("agent_result", {}).get("reward") or 0.0)
                        except Exception:
                            pass
                    if traj_file.exists():
                        try:
                            traj_data = json.loads(traj_file.read_text(encoding="utf-8"))
                            step_list = traj_data.get("steps", []) if isinstance(traj_data, dict) else traj_data
                            steps = len(step_list)
                            last_step = step_list[-1] if step_list else {}
                            m = last_step.get("metrics") or {}
                            tokens = (m.get("prompt_tokens") or 0) + (m.get("completion_tokens") or 0)
                        except Exception:
                            pass
                    task_results[task_name] = {
                        "reward": reward,
                        "steps": steps,
                        "tokens": tokens,
                    }

        return task_results
    finally:
        console.print(f"[*] Stopping vLLM ({model_mode})...")
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()
        time.sleep(5)


@app.function(
    image=image,
    gpu="H100",
    cpu=16,
    memory=65536,
    volumes={
        "/cache/huggingface": volume_hf,
        "/outputs": volume_eval_outputs,
        "/workspace/prime_outputs": volume_prime_outputs,
    },
    timeout=7200,
)
def run_head_to_head_tb2(
    tasks: int = 5,
    checkpoint: str = "step_20",
    dataset: str = "terminal-bench@2.0",
    harness: str = "terminus-2",
):
    """Execute side-by-side 5-task Terminal-Bench 2.0 evaluation on Base Model vs. Step 20."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console(force_terminal=True, width=120)
    console.print(Panel.fit("[bold cyan]⚔️ 5-TASK TERMINAL-BENCH 2.0: BASE MODEL vs. FABLE-5 RL POLICY[/bold cyan]"))

    token = os.environ.get("HF_TOKEN")
    e2b_key = os.environ.get("E2B_API_KEY")
    out_base = Path(f"/outputs/tb2_side_by_side_{int(time.time())}")
    out_base.mkdir(parents=True, exist_ok=True)

    # 1. Evaluate Base Model on 5 tasks
    console.print(f"\n[bold yellow]>>> PHASE 1: EVALUATING BASE MODEL (Qwen3.8-27B) ON {tasks} TASKS <<<[/bold yellow]")
    base_results = _run_single_eval_phase(
        model_mode="base",
        num_tasks=tasks,
        dataset=dataset,
        harness=harness,
        out_base=out_base,
        token=token,
        e2b_key=e2b_key,
        console=console,
    )

    # 2. Evaluate Fable-5 Policy (Step 20) on the same 5 tasks
    console.print(f"\n[bold green]>>> PHASE 2: EVALUATING FABLE-5 RL POLICY ({checkpoint.upper()}) ON {tasks} TASKS <<<[/bold green]")
    rl_results = _run_single_eval_phase(
        model_mode=checkpoint,
        num_tasks=tasks,
        dataset=dataset,
        harness=harness,
        out_base=out_base,
        token=token,
        e2b_key=e2b_key,
        console=console,
    )

    # 3. Print Side-by-Side Comparison Scorecard
    console.print("\n" + "="*80)
    console.print("[bold cyan]=== 🏆 FINAL SIDE-BY-SIDE 5-TASK COMPARISON SCORECARD ===[/bold cyan]")
    console.print("="*80)

    table = Table(title=f"Terminal-Bench 2.0: Base Model vs. Fable-5 RL Policy ({checkpoint})")
    table.add_column("Task Name", style="bold")
    table.add_column("Base Model Result", style="yellow")
    table.add_column(f"Fable-5 ({checkpoint}) Result", style="green")
    table.add_column("Behavioral Difference / Analysis", style="white")

    all_tasks = sorted(set(list(base_results.keys()) + list(rl_results.keys())))
    base_solved = 0
    rl_solved = 0

    for t in all_tasks:
        b_info = base_results.get(t, {"reward": 0.0, "steps": 0, "tokens": 0})
        r_info = rl_results.get(t, {"reward": 0.0, "steps": 0, "tokens": 0})

        b_reward = b_info["reward"]
        r_reward = r_info["reward"]
        if b_reward >= 1.0: base_solved += 1
        if r_reward >= 1.0: rl_solved += 1

        b_str = f"[bold green]PASS (1.0)[/bold green]" if b_reward >= 1.0 else f"[bold red]FAIL (0.0)[/bold red]"
        b_str += f"\n({b_info['steps']} steps, {b_info['tokens']} tok)"

        r_str = f"[bold green]PASS (1.0)[/bold green]" if r_reward >= 1.0 else f"[bold red]FAIL (0.0)[/bold red]"
        r_str += f"\n({r_info['steps']} steps, {r_info['tokens']} tok)"

        diff = ""
        if r_reward > b_reward:
            diff = "[bold green]+ RL Policy solved task (Base failed)[/bold green]"
        elif r_reward == b_reward and r_reward >= 1.0:
            diff = f"[cyan]Both passed (RL: {r_info['steps']} steps vs Base: {b_info['steps']} steps)[/cyan]"
        elif r_reward == b_reward:
            diff = "[dim]Both failed test suite[/dim]"
        else:
            diff = "[yellow]Base passed, RL failed[/yellow]"

        table.add_row(t, b_str, r_str, diff)

    table.add_section()
    table.add_row(
        "[bold]TOTAL SOLVE RATE[/bold]",
        f"[bold yellow]{base_solved}/{len(all_tasks)} ({(base_solved/max(1, len(all_tasks)))*100:.1f}%)[/bold yellow]",
        f"[bold green]{rl_solved}/{len(all_tasks)} ({(rl_solved/max(1, len(all_tasks)))*100:.1f}%)[/bold green]",
        f"[bold green]+{((rl_solved - base_solved)/max(1, len(all_tasks)))*100:.1f}% RL Gain[/bold green]"
    )

    console.print(table)
    volume_eval_outputs.commit()


@app.local_entrypoint()
def main(tasks: int = 5, checkpoint: str = "step_20"):
    """Local entrypoint for side-by-side comparison."""
    run_head_to_head_tb2.remote(tasks=tasks, checkpoint=checkpoint)
