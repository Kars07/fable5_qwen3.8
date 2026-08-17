"""Diagnostic, probing, and sample inspection utilities for RL rollouts and trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def inspect_training_sample_from_trace(
    trace_path: str,
    algo_type: str = "grpo",
) -> Dict[str, Any]:
    """Inspect how a Verifiers trace converts to training tokens, loss masks, and advantages."""
    path = Path(trace_path)
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    traces = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            traces.append(json.loads(line))

    return {
        "trace_count": len(traces),
        "sample_trace_id": traces[0].get("id") if traces else None,
        "algo_type": algo_type,
        "has_rewards": any("reward" in t for t in traces),
    }


def inspect_algorithm_configs(config_paths: Dict[str, str]) -> Dict[str, Any]:
    """Validate TOML/YAML configuration integrity across algorithms."""
    results = {}
    import tomllib

    for name, path_str in config_paths.items():
        p = Path(path_str)
        if not p.exists():
            results[name] = {"exists": False}
            continue

        try:
            cfg = tomllib.loads(p.read_text(encoding="utf-8"))
            results[name] = {
                "exists": True,
                "orchestrator_algo": cfg.get("orchestrator", {}).get("algo", {}).get("type"),
                "has_sources": bool(cfg.get("orchestrator", {}).get("train", {}).get("source")),
            }
        except Exception as e:
            results[name] = {"exists": True, "error": str(e)}

    return results
