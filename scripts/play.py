#!/usr/bin/env python3
"""Watch a trained agent play an Atari game live."""

from __future__ import annotations

import argparse
import time

import numpy as np

from alpha_q.agents import create_agent
from alpha_q.envs.atari import make_atari_env_from_config
from alpha_q.utils.config import load_config
from alpha_q.utils.seeding import get_device, seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a trained agent play")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="configs/default.yaml", help="Base config")
    parser.add_argument("--agent", default=None, help="Agent config override")
    parser.add_argument("--env", default=None, help="Env config override")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play")
    parser.add_argument("--delay", type=float, default=0.02, help="Delay between frames (s)")
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

    env = make_atari_env_from_config(config, render_mode="human")

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.select_action(np.asarray(obs), eval_mode=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)
            done = terminated or truncated
            time.sleep(args.delay)

        print(f"Episode {ep}: reward = {episode_reward:.0f}")

    env.close()


if __name__ == "__main__":
    main()
