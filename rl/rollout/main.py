"""Main Modal entrypoints for Self-Hosted Rollouts, Evaluations, and RL Training."""

from __future__ import annotations

import argparse
from typing import Optional

import modal
from rl.rollout.app import app, hf_cache, image, rl_checkpoints, rl_outputs, sft_checkpoints, vllm_cache
from rl.rollout.evaluators import run_harbor_e2b_eval, run_verifiers_evaluation
from rl.rollout.inference import get_local_e2b_key
from rl.rollout.trainer import run_prime_rl_training


@app.function(
    image=image,
    gpu="A10G",
    cpu=8,
    memory=32768,
    timeout=3600,
    volumes={
        "/cache/huggingface": hf_cache,
        "/cache/vllm": vllm_cache,
        "/opt/artifacts": sft_checkpoints,
        "/outputs": rl_outputs,
    },
)
def evaluate_harbor_modal(
    e2b_api_key: str,
    dataset: str = "terminal-bench/terminal-bench-2",
    harness: str = "terminus_2",
    num_tasks: int = 5,
) -> dict:
    """Modal GPU function running Harbor Terminal-Bench evaluation on E2B microVMs."""
    return run_harbor_e2b_eval(
        e2b_api_key=e2b_api_key,
        dataset=dataset,
        harness=harness,
        num_tasks=num_tasks,
    )


@app.function(
    image=image,
    gpu="A10G",
    cpu=8,
    memory=32768,
    timeout=7200,
    volumes={
        "/cache/huggingface": hf_cache,
        "/cache/vllm": vllm_cache,
        "/opt/artifacts": sft_checkpoints,
        "/outputs": rl_outputs,
        "/checkpoints": rl_checkpoints,
    },
)
def train_rl_modal(
    config_path: str = "/opt/rl/configs/rl/repo_repair_smoke.toml",
    dry_run: bool = False,
) -> dict:
    """Modal GPU function running Prime-RL training loop."""
    return run_prime_rl_training(
        config_path=config_path,
        dry_run=dry_run,
    )


@app.local_entrypoint()
def main(
    mode: str = "eval-verifiers",
    dataset: str = "terminal-bench/terminal-bench-2",
    tasks: int = 5,
    config: str = "rl/configs/rl/repo_repair_smoke.toml",
    dry_run: bool = False,
) -> None:
    """Local entrypoint for launching Modal rollouts."""
    print("=" * 80)
    print(f"🚀 FABLE-5 / PRIME-RL ROLLOUT RUNNER (Mode: {mode})")
    print("=" * 80)

    if mode == "eval-harbor":
        key = get_local_e2b_key()
        res = evaluate_harbor_modal.remote(
            e2b_api_key=key,
            dataset=dataset,
            num_tasks=tasks,
        )
        print("\n[+] Harbor Eval Completed:")
        print(res.get("log", ""))
    elif mode == "train-rl":
        res = train_rl_modal.remote(
            config_path=config,
            dry_run=dry_run,
        )
        print("\n[+] RL Training Finished:")
        print(res.get("stdout", ""))
        if res.get("stderr"):
            print("Errors:", res.get("stderr"))
    else:
        print(f"Unknown mode: {mode}. Supported modes: eval-harbor, train-rl")
