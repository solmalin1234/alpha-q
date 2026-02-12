"""Tests for the YAML config system."""

from __future__ import annotations

from pathlib import Path

from alpha_q.utils.config import apply_overrides, deep_merge, load_config, load_yaml


def test_load_default_config(configs_dir: Path) -> None:
    config = load_yaml(configs_dir / "default.yaml")
    assert "env" in config
    assert "agent" in config
    assert "replay" in config
    assert "training" in config
    assert config["env"]["id"] == "ALE/Pong-v5"


def test_deep_merge_overrides_leaf() -> None:
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 99}}
    result = deep_merge(base, override)
    assert result["a"] == 1
    assert result["b"]["c"] == 99
    assert result["b"]["d"] == 3


def test_deep_merge_adds_new_keys() -> None:
    base = {"a": 1}
    override = {"b": {"c": 2}}
    result = deep_merge(base, override)
    assert result["a"] == 1
    assert result["b"]["c"] == 2


def test_apply_overrides() -> None:
    config = {"agent": {"lr": 0.001}, "seed": 42}
    apply_overrides(config, ["agent.lr=0.01", "seed=123"])
    assert config["agent"]["lr"] == 0.01
    assert config["seed"] == 123


def test_apply_overrides_creates_nested() -> None:
    config: dict = {}
    apply_overrides(config, ["a.b.c=hello"])
    assert config["a"]["b"]["c"] == "hello"


def test_load_config_with_env_override(configs_dir: Path) -> None:
    config = load_config(
        default_path=configs_dir / "default.yaml",
        env_path=configs_dir / "envs" / "breakout.yaml",
    )
    assert config["env"]["id"] == "ALE/Breakout-v5"
    # Other defaults should be preserved
    assert config["agent"]["type"] == "dqn"


def test_load_config_with_cli_overrides(configs_dir: Path) -> None:
    config = load_config(
        default_path=configs_dir / "default.yaml",
        overrides=["seed=99", "training.total_steps=1000"],
    )
    assert config["seed"] == 99
    assert config["training"]["total_steps"] == 1000
