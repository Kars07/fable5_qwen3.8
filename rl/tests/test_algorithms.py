"""Unit tests for RL algorithms, loss functions, importance sampling, and GRPO advantages."""

from __future__ import annotations

import json
import math
import tomllib
from pathlib import Path
import pytest

from rl.probes.loss_reference import default_loss, grpo_advantages


def test_prime_grpo_has_no_std_normalization() -> None:
    """Verify Prime GRPO advantage calculation: reward minus group mean without std scaling."""
    assert grpo_advantages([0.0, 1.0, 1.0, 0.0]) == [-0.5, 0.5, 0.5, -0.5]
    assert grpo_advantages([1.0, 1.0]) == [0.0, 0.0]
    res = grpo_advantages([0.2, 0.4, 0.6, 0.8])
    expected = [-0.3, -0.1, 0.1, 0.3]
    for r, e in zip(res, expected):
        assert math.isclose(r, e, rel_tol=1e-9)


def test_default_loss_importance_ratio_mask_and_kl() -> None:
    """Verify importance ratio clipping, loss mask, and KL divergence penalty."""
    result = default_loss(
        trainer_logprobs=[math.log(0.4), math.log(0.1), math.log(0.3)],
        inference_logprobs=[math.log(0.2), math.log(0.3), math.log(0.3)],
        advantages=[1.0, -1.0, 2.0],
        loss_mask=[True, True, False],
        dppo_mask_high=0.1,
        dppo_mask_low=0.1,
        adv_tau=1.0,
        kl_tau=0.2,
    )
    assert result.kept == [False, False, False]
    expected = 0.2 * (math.log(2.0) ** 2 + math.log(1 / 3) ** 2)
    assert math.isclose(result.loss, expected, rel_tol=1e-12)


def test_default_loss_weights_apply_after_components() -> None:
    """Verify loss component scaling with custom per-token loss weights."""
    result = default_loss(
        trainer_logprobs=[0.0],
        inference_logprobs=[0.0],
        advantages=[2.0],
        loss_mask=[True],
        dppo_mask_high=1.0,
        dppo_mask_low=1.0,
        adv_tau=0.5,
        kl_tau=1.0,
        loss_weights=[3.0],
    )
    assert result.loss == -3.0


def test_echo_configs_preserve_ce_samples_and_dispatch_per_environment() -> None:
    """Verify TOML algorithm configurations for dedicated and mixed training sources."""
    dedicated_path = Path("rl/configs/rl/echo_terminal.toml")
    mixed_path = Path("rl/configs/rl/per_environment_grpo_echo.toml")

    if dedicated_path.exists():
        dedicated = tomllib.loads(dedicated_path.read_text(encoding="utf-8"))
        assert dedicated["orchestrator"]["post_batch_filters"] == []

    if mixed_path.exists():
        mixed = tomllib.loads(mixed_path.read_text(encoding="utf-8"))
        assert mixed["orchestrator"]["post_batch_filters"] == []
        assert mixed["orchestrator"]["algo"]["type"] == "grpo"
        sources = {source["name"]: source for source in mixed["orchestrator"]["train"]["source"]}
        assert "algo" not in sources["browser-grpo"]
        assert sources["terminal-echo"]["algo"]["type"] == "echo"
