"""Atari environment factory with standard preprocessing."""

from __future__ import annotations

import gymnasium as gym


def make_atari_env(
    env_id: str = "ALE/Pong-v5",
    seed: int = 42,
    frameskip: int = 4,
    frame_stack: int = 4,
    noop_max: int = 30,
    terminal_on_life_loss: bool = True,
    screen_size: int = 84,
    grayscale: bool = True,
    render_mode: str | None = None,
) -> gym.Env:
    """Create an Atari env with standard DQN preprocessing.

    Pipeline: raw Atari → AtariPreprocessing (skip, resize, grayscale,
    noop) → FrameStack(4).
    """
    env = gym.make(env_id, frameskip=1, render_mode=render_mode)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=noop_max,
        frame_skip=frameskip,
        screen_size=screen_size,
        terminal_on_life_loss=terminal_on_life_loss,
        grayscale_obs=grayscale,
        scale_obs=False,  # keep uint8 — scale in network forward pass
    )
    env = gym.wrappers.FrameStackObservation(env, frame_stack)
    env.reset(seed=seed)
    return env


def make_atari_env_from_config(config: dict, render_mode: str | None = None) -> gym.Env:
    """Create an Atari env from a config dict."""
    env_cfg = config["env"]
    return make_atari_env(
        env_id=env_cfg["id"],
        seed=config.get("seed", 42),
        frameskip=env_cfg.get("frameskip", 4),
        frame_stack=env_cfg.get("frame_stack", 4),
        noop_max=env_cfg.get("noop_max", 30),
        terminal_on_life_loss=env_cfg.get("terminal_on_life_loss", True),
        screen_size=env_cfg.get("screen_size", 84),
        grayscale=env_cfg.get("grayscale", True),
        render_mode=render_mode,
    )
