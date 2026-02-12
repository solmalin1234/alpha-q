"""Smoke tests for the Atari environment factory."""

from __future__ import annotations

import pytest


def test_make_atari_env_from_config(default_config: dict) -> None:
    """Test that the env factory creates a working environment."""
    pytest.importorskip("ale_py")

    from alpha_q.envs.atari import make_atari_env_from_config

    env = make_atari_env_from_config(default_config)
    obs, info = env.reset()

    import numpy as np

    obs = np.asarray(obs)
    # FrameStack(4) on 84x84 grayscale → (4, 84, 84)
    assert obs.shape == (4, 84, 84), f"Unexpected shape: {obs.shape}"
    assert obs.dtype == np.uint8

    # Take one step
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)
    obs2 = np.asarray(obs2)
    assert obs2.shape == (4, 84, 84)

    env.close()
