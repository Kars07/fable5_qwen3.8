import modal

app = modal.App("gpu-probe")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch")

@app.function(image=image, gpu="A10G:4", timeout=120)
def probe():
    import torch
    return {
        "cuda": torch.cuda.is_available(),
        "count": torch.cuda.device_count(),
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

@app.local_entrypoint()
def main():
    print("[*] Probing 4x A10G on Modal...", flush=True)
    res = probe.remote()
    print("[+] Probe result:", res, flush=True)
