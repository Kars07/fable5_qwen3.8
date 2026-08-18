"""Upload 100% native markdown visual bar chart README.md to Hugging Face."""

import modal
from pathlib import Path

app = modal.App("upload-native-markdown-readme")
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub")

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def upload_readme(content: str, repo_id: str = "eniairaph07/Qwen3.8-27b-FABLE-GGUF"):
    import os
    from huggingface_hub import HfApi
    
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN secret not found!")
        
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"[+] Successfully updated README.md in {repo_id}")

@app.local_entrypoint()
def main():
    from rl.update_hf_readme import generate_clean_markdown_readme
    content = generate_clean_markdown_readme()
    upload_readme.remote(content=content)
