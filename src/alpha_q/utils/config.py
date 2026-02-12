"""YAML config loading with layered merging and CLI overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (returns a new dict)."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply dot-notation CLI overrides like ``key.subkey=value``.

    Values are parsed as YAML scalars so ``"true"`` becomes ``True``,
    ``"42"`` becomes ``42``, etc.
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override!r}")
        key_path, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        keys = key_path.split(".")
        node = config
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return config


def load_config(
    default_path: str | Path = "configs/default.yaml",
    agent_path: str | Path | None = None,
    env_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Build a config by merging: default → agent → env → CLI overrides."""
    config = load_yaml(default_path)
    if agent_path:
        config = deep_merge(config, load_yaml(agent_path))
    if env_path:
        config = deep_merge(config, load_yaml(env_path))
    if overrides:
        config = apply_overrides(config, overrides)
    return config
