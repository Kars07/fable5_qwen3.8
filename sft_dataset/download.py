"""Dataset download utility for Glint-Research/Fable-5-traces."""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

# Ensure UTF-8 console output
sys.stdout.reconfigure(encoding="utf-8")


def download_fable_dataset(
    dest_dir: str = "dataset/data",
    include_pi_traces: bool = False,
    max_pi_traces: int = 20,
    force: bool = False,
) -> Path:
    """Download fable5_cot_merged.jsonl and optionally sample pi-traces from HuggingFace."""
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    merged_dest = dest_path / "fable5_cot_merged.jsonl"
    print(f"[*] Downloading fable5_cot_merged.jsonl from Glint-Research/Fable-5-traces...")

    if merged_dest.exists() and not force:
        print(f"[+] Local copy already exists at: {merged_dest.resolve()} ({merged_dest.stat().st_size / (1024*1024):.2f} MB)")
    else:
        cached_file = hf_hub_download(
            repo_id="Glint-Research/Fable-5-traces",
            filename="fable5_cot_merged.jsonl",
            repo_type="dataset",
            force_download=force,
        )
        # Copy or symlink to local dataset directory for direct access
        import shutil
        shutil.copy2(cached_file, merged_dest)
        print(f"[+] Successfully saved to: {merged_dest.resolve()} ({merged_dest.stat().st_size / (1024*1024):.2f} MB)")

    if include_pi_traces:
        pi_dest_dir = dest_path / "pi-traces"
        pi_dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"[*] Fetching pi-traces file list from Hugging Face...")
        api = HfApi()
        files = api.list_repo_files(repo_id="Glint-Research/Fable-5-traces", repo_type="dataset")
        pi_files = [f for f in files if f.startswith("pi-traces/")][:max_pi_traces]
        print(f"[*] Downloading {len(pi_files)} sample pi-trace files to {pi_dest_dir.resolve()}...")
        for pfile in pi_files:
            cached_pi = hf_hub_download(
                repo_id="Glint-Research/Fable-5-traces",
                filename=pfile,
                repo_type="dataset",
            )
            shutil.copy2(cached_pi, pi_dest_dir / Path(pfile).name)
        print(f"[+] Downloaded {len(pi_files)} pi-trace files.")

    return merged_dest


def main():
    parser = argparse.ArgumentParser(description="Download Glint-Research/Fable-5-traces dataset.")
    parser.add_argument("--dest", type=str, default="dataset/data", help="Destination folder for downloaded data")
    parser.add_argument("--include-pi-traces", action="store_true", help="Download sample pi-traces files")
    parser.add_argument("--max-pi-traces", type=int, default=20, help="Maximum number of pi-traces to download")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")

    args = parser.parse_args()
    download_fable_dataset(
        dest_dir=args.dest,
        include_pi_traces=args.include_pi_traces,
        max_pi_traces=args.max_pi_traces,
        force=args.force,
    )


if __name__ == "__main__":
    main()
