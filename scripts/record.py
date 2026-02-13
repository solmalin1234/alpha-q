#!/usr/bin/env python3
"""Record gameplay videos of a trained agent."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

from alpha_q.agents import create_agent
from alpha_q.envs.atari import make_atari_env_from_config
from alpha_q.utils.config import load_config
from alpha_q.utils.seeding import get_device, seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record agent gameplay videos")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="configs/default.yaml", help="Base config")
    parser.add_argument("--agent", default=None, help="Agent config override")
    parser.add_argument("--env", default=None, help="Env config override")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to record")
    parser.add_argument("--output-dir", default=None, help="Video output directory")
    parser.add_argument(
        "--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    config = load_config(
        default_path=args.config,
        agent_path=args.agent,
        env_path=args.env,
        overrides=args.overrides,
    )

    seed_everything(config["seed"])
    device = get_device(config.get("device", "auto"))

    tmp_env = make_atari_env_from_config(config)
    n_actions = tmp_env.action_space.n
    tmp_env.close()

    agent = create_agent(
        config["agent"]["type"], config=config, device=device, n_actions=n_actions
    )
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    video_dir = args.output_dir or config["paths"]["video_dir"]
    Path(video_dir).mkdir(parents=True, exist_ok=True)

    env = make_atari_env_from_config(config, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda ep: ep < args.episodes,
        name_prefix="agent",
    )

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.select_action(np.asarray(obs), eval_mode=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)
            done = terminated or truncated

        print(f"Episode {ep + 1}: reward = {episode_reward:.0f}")

    env.close()
    print(f"Videos saved to: {video_dir}")


if __name__ == "__main__":
    main()
