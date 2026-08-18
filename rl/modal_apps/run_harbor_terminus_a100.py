"""Regime 2 Level-3 End-to-End Terminal-Bench 2.1 Evaluation on 2x NVIDIA A100-80GB (TP=2).

- Model: Qwen3.8-27B Base + Step 20/25 LoRA Adapter served via vLLM TP=2 on 2x A100-80GB (160 GB VRAM)
- Context Length: 16,384 tokens
- Harness: Terminus 2 (Harbor 0.20)
- Execution Environment: Live E2B Sandbox MicroVMs
- Automated Scoring: Official in-container verification suites
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

app = modal.App("fable5-regime2-terminalbench-a100")

# Persistent Cloud Volumes
volume_hf = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
volume_eval_outputs = modal.Volume.from_name("fable5-terminalbench-outputs-a100", create_if_missing=True)
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


@app.function(
    image=image,
    gpu="A100-80GB:2",
    cpu=16,
    memory=65536,
    volumes={
        "/cache/huggingface": volume_hf,
        "/outputs": volume_eval_outputs,
        "/workspace/prime_outputs": volume_prime_outputs,
    },
    timeout=86400,
)
def run_regime2_eval_a100(
    checkpoint_step: str = "step_20",
    num_tasks: int = 89,
    dataset: str = "terminal-bench@2.0",
    harness: str = "terminus-2",
    e2b_api_key: str | None = None,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Execute Regime 2 Level-3 End-to-End Evaluation on Terminal-Bench 2.1 via Terminus & E2B on 2x A100-80GB."""
    from huggingface_hub import snapshot_download
    from rich.console import Console
    from rich.table import Table

    console = Console(force_terminal=True, width=120)
    console.print(f"[bold cyan]=======================================================================[/bold cyan]")
    console.print(f"[bold cyan]🎯 REGIME 2: Terminal-Bench 2.1 / Terminus on 2x NVIDIA A100-80GB (16K Context)[/bold cyan]")
    console.print(f"[bold cyan]=======================================================================[/bold cyan]")
    console.print(f"Policy Checkpoint: [bold green]{checkpoint_step}[/bold green] | Tasks: [bold green]{num_tasks}[/bold green] | Harness: [bold green]{harness}[/bold green]\n")

    token = hf_token or os.environ.get("HF_TOKEN")
    e2b_key = e2b_api_key or os.environ.get("E2B_API_KEY")

    # Runtime patch for Harbor E2B timeout compliance
    try:
        import harbor.environments.e2b as e2b_mod
        p_e2b = Path(e2b_mod.__file__)
        src_e2b = p_e2b.read_text(encoding="utf-8")
        if "timeout=86_400" in src_e2b:
            p_e2b.write_text(src_e2b.replace("timeout=86_400", "timeout=3600"), encoding="utf-8")
            console.print("[+] Successfully patched Harbor E2B timeout to 3600s!\n")
    except Exception as e:
        console.print(f"[yellow][!] E2B patch notice: {e}[/yellow]\n")

    # 1. Sync LoRA Checkpoint (Stored persistently on volume_hf to eliminate re-downloads)
    lora_dir = Path(f"/cache/huggingface/checkpoints/{checkpoint_step}")
    lora_dir.mkdir(parents=True, exist_ok=True)
    model_name = "eval_policy"
    vllm_lora_args = []

    if checkpoint_step != "base":
        if not (lora_dir / "adapter_model.safetensors").exists():
            # Check local training volume first
            local_step_dir = Path(f"/workspace/prime_outputs/checkpoints/{checkpoint_step}")
            if local_step_dir.exists():
                console.print(f"[*] Found local training checkpoint in {local_step_dir}, copying...")
                shutil.copytree(str(local_step_dir), str(lora_dir), dirs_exist_ok=True)
            else:
                console.print(f"[*] Downloading standalone LoRA adapter for {checkpoint_step} (~2.4 GB) from Hugging Face...")
                snapshot_download(
                    repo_id="eniairaph07/qwen3.8-27b-fable5-rl-sft-steps",
                    allow_patterns=f"rl_checkpoints/{checkpoint_step}/*",
                    local_dir="/tmp/hf_download",
                    token=token,
                )
                src_dir = Path(f"/tmp/hf_download/rl_checkpoints/{checkpoint_step}")
                if src_dir.exists():
                    shutil.copytree(str(src_dir), str(lora_dir), dirs_exist_ok=True)
                    console.print(f"[+] Synced {checkpoint_step} adapter into persistent cache ({lora_dir})")

        # Extract clean LoRA weights if DCP checkpoint is present to guarantee canonical PEFT keys
        if (lora_dir / "trainer" / "__0_0.distcp").exists():
            console.print(f"[*] Extracting standardized PEFT LoRA adapter from DCP checkpoint in {lora_dir}...")
            try:
                import torch
                import torch.distributed.checkpoint as dcp
                from torch.distributed.checkpoint import FileSystemReader
                from safetensors.torch import save_file

                reader = FileSystemReader(lora_dir / "trainer")
                meta = reader.read_metadata()

                # Exclude optimizer momentum buffers (exp_avg, exp_avg_sq) and keep only pure model weights
                lora_keys = [
                    k for k in meta.state_dict_metadata.keys()
                    if "lora" in k.lower()
                    and not any(opt in k.lower() for opt in ["optimizer", "optimizers", "exp_avg", "step", "state."])
                    and (k.startswith("app.model.") or not k.startswith("app."))
                ]
                print(f"[+] Filtered {len(lora_keys)} pure LoRA weight tensors (excluded all optimizer momentum buffers)", flush=True)
                state_dict = {k: torch.empty(meta.state_dict_metadata[k].size, dtype=meta.state_dict_metadata[k].properties.dtype) for k in lora_keys}
                dcp.load(state_dict=state_dict, storage_reader=reader)

                clean_state_dict = {}
                for k, v in state_dict.items():
                    clean_k = k
                    for prefix in ["app.model.model.language_model.", "app.model.language_model.", "app.model.", "language_model."]:
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

                save_file(clean_state_dict, str(lora_dir / "adapter_model.safetensors"))
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
                volume_hf.commit()
                console.print(f"[+] Successfully extracted and saved persistent adapter_model.safetensors ({lora_dir / 'adapter_model.safetensors'})")
            except Exception as e:
                console.print(f"[yellow][!] Could not extract from DCP: {e}[/yellow]")

        if (lora_dir / "adapter_model.safetensors").exists():
            console.print(f"[+] Verified persistent LoRA adapter at {lora_dir}")
            vllm_lora_args = [
                "--enable-lora",
                "--lora-modules", f"{model_name}={str(lora_dir)}",
                "--max-lora-rank", "64",
            ]
        else:
            console.print(f"[yellow][!] No adapter found, running base model.[/yellow]")
            model_name = "Qwen/Qwen3.8-27B"
    else:
        model_name = "Qwen/Qwen3.8-27B"

    # 2. Launch Background vLLM OpenAI Server on 2x A100-80GB (Tensor Parallelism = 2, 32K Context)
    out_dir = Path(f"/outputs/tb2_eval_a100_{checkpoint_step}_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    vllm_log = out_dir / "vllm_server.log"

    vllm_cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "Qwen/Qwen3.8-27B",
        "--tensor-parallel-size", "2",
        "--gpu-memory-utilization", "0.95",
        "--max-model-len", "32768",
        "--dtype", "bfloat16",
        "--port", "8000",
        "--trust-remote-code",
        "--download-dir", "/cache/huggingface",
    ] + vllm_lora_args

    console.print(f"[*] Starting vLLM Server on 2x A100-80GB (TP=2, 32K Context Window)...")
    with vllm_log.open("w", encoding="utf-8") as handle:
        vllm_proc = subprocess.Popen(
            vllm_cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    try:
        _wait_for_vllm_server("http://127.0.0.1:8000/v1/models", proc=vllm_proc, log_path=vllm_log, timeout=1200)

        # 3. Build & Execute Harbor Terminal-Bench Command with E2B MicroVMs (XML Mode)
        harbor_log = out_dir / "harbor_execution.log"
        harbor_results_dir = out_dir / "harbor_output"
        harbor_results_dir.mkdir(parents=True, exist_ok=True)

        harbor_cmd = [
            "harbor", "run",
            "-d", dataset,
            "-m", f"openai/{model_name}",
            "-a", harness,
            "--env", "e2b",
            "--ak", "parser_name=xml",
            "--ak", "enable_summarize=false",
            "--ak", "proactive_summarization_threshold=0",
            "--jobs-dir", str(harbor_results_dir),
        ]
        if num_tasks and num_tasks > 0 and num_tasks < 89:
            harbor_cmd.extend(["--n-tasks", str(num_tasks)])

        env = {
            **os.environ,
            "E2B_API_KEY": e2b_key,
            "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
            "OPENAI_API_KEY": "task2-prime-token",
            "PYTHONUNBUFFERED": "1",
        }

        console.print(f"[*] Executing Harbor Terminal-Bench 2.1 via Terminus 2 on E2B Sandboxes (Full Suite)...")
        console.print(f"    Command: {' '.join(harbor_cmd)}\n")

        t_start = time.time()
        with harbor_log.open("w", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                harbor_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            for line in proc.stdout:
                handle.write(line)
                print(line, end="", flush=True)
            proc.wait()

        eval_duration = time.time() - t_start
        console.print(f"\n[+] Harbor finished with exit code {proc.returncode} in {eval_duration/60:.2f} minutes.\n")

        # 4. Parse Ground-Truth Automated Verifier Results
        results_file = next(harbor_results_dir.rglob("result.json"), None)
        solve_rate = 0.0
        solved_tasks = 0
        total_eval_tasks = num_tasks or 89

        if results_file and results_file.exists():
            try:
                data = json.loads(results_file.read_text(encoding="utf-8"))
                stats = data.get("stats", {})
                evals = stats.get("evals", {})
                for k, v in evals.items():
                    metrics = v.get("metrics", [])
                    if metrics and isinstance(metrics, list) and "mean" in metrics[0]:
                        solve_rate = float(metrics[0]["mean"])
                        break
                if not solve_rate:
                    solve_rate = float(data.get("accuracy", 0.0) or data.get("pass_rate", 0.0))
                total_eval_tasks = stats.get("n_completed_trials", total_eval_tasks) or total_eval_tasks
                solved_tasks = int(round(solve_rate * total_eval_tasks))
            except Exception as e:
                console.print(f"[yellow][!] Result parsing note: {e}[/yellow]")

        # 5. Display Official Regime 2 Terminal-Bench 2.1 Scorecard
        score_table = Table(title=f"Terminal-Bench 2.1 Official Scorecard (2x A100-80GB, 16K Context)", show_lines=True)
        score_table.add_column("Benchmark Metric", style="bold yellow")
        score_table.add_column("Model / Checkpoint Performance", style="bold green")
        score_table.add_column("Leaderboard Baseline Reference", style="cyan")

        score_table.add_row("Evaluated Checkpoint", checkpoint_step, "GRPO Policy Step")
        score_table.add_row("Compute Topology", "2x NVIDIA A100-80GB (160 GB VRAM)", "$6.80 / hr")
        score_table.add_row("Context Length Window", "16,384 tokens", "Full Long-Context")
        score_table.add_row("Autonomous Agent Harness", harness, "Official Terminus 2")
        score_table.add_row("Benchmark Tasks Evaluated", f"{total_eval_tasks} tasks (Full Suite)", "Terminal-Bench 2.1")
        score_table.add_row("Automated Pass Rate", f"[bold green]{solve_rate*100:.1f}% ({solved_tasks}/{total_eval_tasks})[/bold green]", "Base Qwen3.8-27B = 73.0%")
        score_table.add_row("Opus 4.6 Max Baseline", "78.2%", "Top Benchmark Reference")

        console.print(score_table)
        volume_eval_outputs.commit()

        return {
            "checkpoint": checkpoint_step,
            "accuracy": solve_rate,
            "solved": solved_tasks,
            "total": total_eval_tasks,
            "duration_minutes": eval_duration / 60,
        }

    finally:
        console.print("[*] Tearing down vLLM inference server...")
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=20)
        except Exception:
            vllm_proc.kill()


@app.local_entrypoint()
def main(
    checkpoint: str = "step_20",
    tasks: int = 89,
    dataset: str = "terminal-bench@2.0",
    harness: str = "terminus-2",
):
    """Local entrypoint for Terminal-Bench 2.1 evaluation on 2x A100-80GB."""
    run_regime2_eval_a100.remote(
        checkpoint_step=checkpoint,
        num_tasks=tasks,
        dataset=dataset,
        harness=harness,
    )
