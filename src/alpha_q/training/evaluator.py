"""Evaluation loop for measuring agent performance."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from alpha_q.agents.base import BaseAgent


def evaluate(
    agent: BaseAgent,
    env: gym.Env,
    n_episodes: int = 10,
) -> dict[str, float]:
    """Run *n_episodes* with greedy actions and return summary stats.

    Returns:
        Dict with ``reward_mean``, ``reward_std``, ``length_mean``.
    """
    rewards: list[float] = []
    lengths: list[int] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done:
            action = agent.select_action(np.asarray(obs), eval_mode=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)
            episode_length += 1
            done = terminated or truncated

        rewards.append(episode_reward)
        lengths.append(episode_length)

    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "length_mean": float(np.mean(lengths)),
    }
