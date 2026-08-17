"""Unit tests for Terminus 2 Harness and LocalEnvironment execution."""

from __future__ import annotations

import asyncio
import pytest
from pydantic import ValidationError

from rl.harnesses.terminus_2.harness import (
    PROGRAM_SOURCE,
    Terminus2Harness,
    Terminus2HarnessConfig,
)
from rl.harnesses.terminus_2.program import LocalEnvironment, parse_args


def test_terminus2_harness_config_defaults() -> None:
    """Verify default pinned version and configuration structure."""
    config = Terminus2HarnessConfig()
    assert config.version == "0.21.0"
    assert config.disabled_tools is None or config.disabled_tools == []


def test_terminus2_harness_config_validation() -> None:
    """Verify version pattern validation."""
    valid = Terminus2HarnessConfig(version="0.22.0-rc1")
    assert valid.version == "0.22.0-rc1"

    with pytest.raises(ValidationError):
        Terminus2HarnessConfig(version="bad version / with / slashes")


def test_terminus2_harness_capabilities() -> None:
    """Verify harness class capabilities and protocols."""
    harness = Terminus2Harness(Terminus2HarnessConfig())
    assert harness.APPENDS_SYSTEM_PROMPT is True
    assert harness.SUPPORTS_MCP is False


def test_terminus2_program_source_templating() -> None:
    """Verify script templating with version parameter."""
    source = PROGRAM_SOURCE.replace("{version}", "0.21.0")
    assert 'dependencies = ["harbor==0.21.0"]' in source


@pytest.mark.asyncio
async def test_local_environment_exec() -> None:
    """Verify LocalEnvironment subprocess execution."""
    env = LocalEnvironment()
    res = await env.exec("echo hello")
    assert res.return_code == 0
    assert "hello" in res.stdout.strip()
