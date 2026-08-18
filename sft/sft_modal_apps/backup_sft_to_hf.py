"""Modal Application to Backup Fine-Tuned SFT LoRA Checkpoint to Hugging Face Hub."""

from __future__ import annotations

import os
from pathlib import Path

import modal

app = modal.App("fable5-sft-hf-backup")

volume_checkpoints = modal.Volume.from_name("fable5-sft-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub>=0.24.0")
)


@app.function(
    image=image,
    volumes={"/opt/artifacts": volume_checkpoints},
    secrets=[modal.Secret.from_name("huggingface-secret")] if "huggingface-secret" in os.environ.get("MODAL_SECRETS", "") else [],
    timeout=3600,
)
def backup_to_hf(
    repo_id: str,
    hf_token: str | None = None,
    checkpoint_subdir: str = "checkpoints/qwen_4bit_lora/best_checkpoint",
    private: bool = True,
) -> dict[str, str]:
    """Upload fine-tuned SFT LoRA weights to Hugging Face Hub."""
    from huggingface_hub import HfApi, create_repo

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN or hf_token parameter must be provided to upload to Hugging Face.")

    volume_checkpoints.reload()

    artifacts_root = Path("/opt/artifacts")
    print(f"[*] Inspecting /opt/artifacts content:\n", flush=True)
    for p in artifacts_root.rglob("*"):
        if p.is_file():
            print(f"    - {p.relative_to(artifacts_root)} ({p.stat().st_size / (1024*1024):.2f} MB)", flush=True)

    ckpt_path = artifacts_root / checkpoint_subdir
    if not ckpt_path.exists():
        # Fallback to finding best_checkpoint anywhere in /opt/artifacts
        candidates = list(artifacts_root.rglob("best_checkpoint")) + list(artifacts_root.rglob("adapter_model.safetensors"))
        if candidates:
            ckpt_path = candidates[0].parent if candidates[0].is_file() else candidates[0]
        else:
            ckpt_path = artifacts_root

    print(f"\n[*] Target upload directory: {ckpt_path}", flush=True)

    import shutil
    staging_dir = Path("/tmp/upload_staging")
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    shutil.copytree(ckpt_path, staging_dir)

    # Sanitize README.md and adapter_config.json
    readme_file = staging_dir / "README.md"
    if readme_file.exists():
        text = readme_file.read_text(encoding="utf-8")
        text = text.replace("/cache/models/Qwen3.8-27B", "Qwen/Qwen3.8-27B")
        text = text.replace("/cache/models/", "")
        readme_file.write_text(text, encoding="utf-8")

    adapter_config_file = staging_dir / "adapter_config.json"
    if adapter_config_file.exists():
        text = adapter_config_file.read_text(encoding="utf-8")
        text = text.replace("/cache/models/Qwen3.8-27B", "Qwen/Qwen3.8-27B")
        text = text.replace("/cache/models/", "")
        adapter_config_file.write_text(text, encoding="utf-8")

    api = HfApi(token=token)

    print(f"[*] Creating/verifying Hugging Face repository: {repo_id} (private={private})...", flush=True)
    create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)

    print(f"[*] Uploading sanitized checkpoint folder from {staging_dir} to {repo_id}...", flush=True)
    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Backup fine-tuned SFT LoRA checkpoint (100 steps)",
    )

    repo_url = f"https://huggingface.co/{repo_id}"
    print(f"\n[+] SUCCESS! Checkpoint successfully uploaded to Hugging Face: {repo_url}", flush=True)

    return {
        "status": "success",
        "repo_id": repo_id,
        "repo_url": repo_url,
        "uploaded_path": str(ckpt_path),
    }


@app.local_entrypoint()
def main(
    repo_id: str = "eniairaph07/qwen3.8-27b-fable5",
    token: str = "",
    checkpoint_dir: str = "checkpoints/qwen_4bit_lora/best_checkpoint",
    private: bool = True,
) -> None:
    """Entrypoint to trigger the backup."""
    if not token:
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            env_file = Path(".env")
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("HF_TOKEN"):
                        token = line.split("=", 1)[1].strip().strip("'\"")

    if not repo_id:
        print("[!] Error: --repo-id is required (e.g. --repo-id 'username/qwen3.8-27b-fable5')")
        return

    print("=" * 80)
    print(f"[*] Backing up fine-tuned SFT Checkpoint to Hugging Face: {repo_id}")
    print("=" * 80)

    res = backup_to_hf.remote(
        repo_id=repo_id,
        hf_token=token,
        checkpoint_subdir=checkpoint_dir,
        private=private,
    )
    print("\n[+] Result:", res)
