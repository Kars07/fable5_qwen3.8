"""Dataset download and verification utility for nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1."""

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

# Ensure UTF-8 console output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ID = "nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1"
FILENAME = "atcb_terminal_pivot_release_final_v2.jsonl"


def compute_file_hash(filepath: Path, chunk_size: int = 8192 * 1024) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_nemotron_rl_dataset(
    dest_dir: str = "rl_dataset/data",
    force: bool = False,
) -> Path:
    """Download atcb_terminal_pivot_release_final_v2.jsonl from Hugging Face."""
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    dest_file = dest_path / FILENAME

    print(f"[*] Checking local dataset file: {dest_file.resolve()}")
    if dest_file.exists() and not force:
        size_mb = dest_file.stat().st_size / (1024 * 1024)
        print(f"[+] Local copy already exists ({size_mb:.2f} MB). Skipping download.")
        return dest_file

    print(f"[*] Downloading {FILENAME} from {REPO_ID} on Hugging Face...")
    cached_file = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset",
        force_download=force,
    )

    shutil.copy2(cached_file, dest_file)
    size_mb = dest_file.stat().st_size / (1024 * 1024)
    print(f"[+] Successfully saved to: {dest_file.resolve()} ({size_mb:.2f} MB)")
    return dest_file


def verify_dataset(filepath: Path) -> bool:
    """Quick integrity check on downloaded jsonl file."""
    print(f"[*] Verifying dataset integrity: {filepath.resolve()}")
    if not filepath.exists():
        print(f"[-] Error: File {filepath} does not exist.")
        return False

    line_count = 0
    valid_json_count = 0
    import json

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            if line.strip():
                try:
                    json.loads(line)
                    valid_json_count += 1
                except Exception:
                    pass

    print(f"[+] Total Lines: {line_count}")
    print(f"[+] Valid JSON Records: {valid_json_count}")
    return line_count > 0 and line_count == valid_json_count


def main():
    parser = argparse.ArgumentParser(description="Download and verify nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1 dataset.")
    parser.add_argument("--dest", type=str, default="rl_dataset/data", help="Destination folder for downloaded data")
    parser.add_argument("--force", action="store_true", help="Force re-download even if local copy exists")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify JSON integrity after download")

    args = parser.parse_args()
    dest_file = download_nemotron_rl_dataset(dest_dir=args.dest, force=args.force)

    if args.verify:
        is_valid = verify_dataset(dest_file)
        if not is_valid:
            print("[-] Dataset verification failed!")
            sys.exit(1)
        print("[+] Dataset verification passed successfully!")


if __name__ == "__main__":
    main()
