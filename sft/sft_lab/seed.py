"""Reproducibility and environment diagnostics."""

import os
import platform
import random
import sys
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def generate_environment_report() -> Dict[str, Any]:
    """Generate diagnostic environment report for reproducibility tracking."""
    report = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
            }
            for i in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [],
    }
    return report
