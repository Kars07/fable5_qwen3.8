"""Unit tests for Claude Code ACP Harness in Verifiers v1."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rl.harnesses.claude_harness.harness import (
    ACP_VERSION,
    CLAUDE_ACP_DIR,
    ClaudeCodeHarness,
    ClaudeCodeHarnessConfig,
)


def test_claude_harness_config_defaults() -> None:
    """Verify default pinned version and configuration structure."""
    config = ClaudeCodeHarnessConfig()
    assert config.version == "2.1.232"
    assert config.disabled_tools is None or config.disabled_tools == []


def test_claude_harness_config_validation() -> None:
    """Verify version pattern validation and custom parameters."""
    valid = ClaudeCodeHarnessConfig(version="2.1.233-beta.1", disabled_tools=["bash"])
    assert valid.version == "2.1.233-beta.1"
    assert valid.disabled_tools == ["bash"]

    with pytest.raises(ValidationError):
        ClaudeCodeHarnessConfig(version="invalid version with spaces")


def test_claude_harness_capabilities() -> None:
    """Verify harness class capabilities and protocols."""
    harness = ClaudeCodeHarness(ClaudeCodeHarnessConfig())
    assert harness.APPENDS_SYSTEM_PROMPT is True
    assert harness.SUPPORTS_MCP is True
    assert harness.SUPPORTS_SKILLS is True


def test_claude_harness_acp_paths() -> None:
    """Verify ACP path formatting."""
    versions = {"version": "2.1.232", "acp_version": ACP_VERSION}
    path = CLAUDE_ACP_DIR.format(**versions)
    assert "2.1.232" in path
    assert ACP_VERSION in path
