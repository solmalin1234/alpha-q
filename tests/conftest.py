"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def configs_dir() -> Path:
    """Path to the configs/ directory."""
    return Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture
def default_config(configs_dir: Path) -> dict:
    """Load the default config dict."""
    from alpha_q.utils.config import load_yaml

    return load_yaml(configs_dir / "default.yaml")
