#!/usr/bin/env python3
"""Training entry point for Alpha-Q agents."""

from __future__ import annotations

import argparse

from alpha_q.agents import create_agent
from alpha_q.envs.atari import make_atari_env_from_config
from alpha_q.utils.config import load_config
from alpha_q.utils.seeding import get_device, seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DQN agent on an Atari environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python scripts/train.py
  python scripts/train.py --env configs/envs/pong.yaml
  python scripts/train.py --agent configs/agents/dqn.yaml --env configs/envs/breakout.yaml
  python scripts/train.py --set training.total_steps=500000 --set agent.lr=1e-4
""",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to base config (default: configs/default.yaml)",
    )
    parser.add_argument("--agent", default=None, help="Path to agent config override")
    parser.add_argument("--env", default=None, help="Path to env config override")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="Override config values (e.g. --set training.total_steps=500000)",
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
    print(f"Device: {device}")
    print(f"Agent: {config['agent']['type']}")
    print(f"Environment: {config['env']['id']}")

    tmp_env = make_atari_env_from_config(config)
    n_actions = tmp_env.action_space.n
    tmp_env.close()

    agent = create_agent(
        config["agent"]["type"], config=config, device=device, n_actions=n_actions
    )

    from alpha_q.training.trainer import train

    train(agent, config)


if __name__ == "__main__":
    main()
