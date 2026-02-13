"""Main training loop for DQN agents."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from tqdm import tqdm

from alpha_q.agents.base import BaseAgent
from alpha_q.envs.atari import make_atari_env_from_config
from alpha_q.memory.prioritized_replay_buffer import PrioritizedReplayBuffer
from alpha_q.memory.replay_buffer import ReplayBuffer
from alpha_q.training.evaluator import evaluate
from alpha_q.utils.logging import ExperimentLogger


def get_epsilon(step: int, cfg: dict) -> float:
    """Linear epsilon decay."""
    agent_cfg = cfg["agent"]
    frac = min(1.0, step / agent_cfg["epsilon_decay_steps"])
    return agent_cfg["epsilon_start"] + frac * (
        agent_cfg["epsilon_end"] - agent_cfg["epsilon_start"]
    )


def get_beta(step: int, cfg: dict) -> float:
    """Linear beta annealing from beta_start to beta_end."""
    replay_cfg = cfg["replay"]
    total = cfg["training"]["total_steps"]
    frac = min(1.0, step / total)
    return replay_cfg["beta_start"] + frac * (replay_cfg["beta_end"] - replay_cfg["beta_start"])


def train(agent: BaseAgent, config: dict) -> None:
    """Run the full training loop.

    Steps: fill buffer → train → eval → checkpoint → log.
    """
    train_cfg = config["training"]
    replay_cfg = config["replay"]
    # ── environments ──────────────────────────────────────────────────────
    train_env = make_atari_env_from_config(config)
    eval_env = make_atari_env_from_config(config)

    # Infer observation shape from the environment
    obs, _ = train_env.reset()
    obs = np.asarray(obs)
    obs_shape = obs.shape

    # ── replay buffer ─────────────────────────────────────────────────────
    use_per = replay_cfg.get("type", "uniform") == "prioritized"
    if use_per:
        buffer = PrioritizedReplayBuffer(
            replay_cfg["capacity"], obs_shape, alpha=replay_cfg.get("alpha", 0.6)
        )
    else:
        buffer = ReplayBuffer(replay_cfg["capacity"], obs_shape)

    # ── logger ────────────────────────────────────────────────────────────
    logger = ExperimentLogger(
        experiment_name=config["mlflow"]["experiment_name"],
        tracking_uri=config["mlflow"]["tracking_uri"],
    )
    logger.log_params(config)

    # ── training ──────────────────────────────────────────────────────────
    episode_rewards: deque[float] = deque(maxlen=100)
    episode_reward = 0.0
    episode_count = 0

    pbar = tqdm(range(1, train_cfg["total_steps"] + 1), desc="Training")

    for step in pbar:
        epsilon = get_epsilon(step, config)

        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = train_env.action_space.sample()
        else:
            action = agent.select_action(obs, eval_mode=False)

        next_obs, reward, terminated, truncated, _ = train_env.step(action)
        next_obs = np.asarray(next_obs)
        done = terminated or truncated

        buffer.push(obs, action, float(reward), next_obs, done)
        obs = next_obs
        episode_reward += float(reward)

        if done:
            obs, _ = train_env.reset()
            obs = np.asarray(obs)
            episode_rewards.append(episode_reward)
            episode_count += 1
            logger.log_metric("episode/reward", episode_reward, step=step)
            if len(episode_rewards) > 0:
                logger.log_metric(
                    "episode/reward_avg100",
                    float(np.mean(episode_rewards)),
                    step=step,
                )
            episode_reward = 0.0

        # ── train step ────────────────────────────────────────────────────
        if len(buffer) >= replay_cfg["min_size"] and step % train_cfg["train_freq"] == 0:
            from alpha_q.utils.seeding import get_device

            device = get_device(config.get("device", "auto"))
            if use_per:
                beta = get_beta(step, config)
                batch = buffer.sample(config["agent"]["batch_size"], device=device, beta=beta)
            else:
                batch = buffer.sample(config["agent"]["batch_size"], device=device)

            metrics = agent.learn(batch)

            # Update priorities for PER
            td_errors = metrics.pop("td_errors", None)
            if use_per and td_errors is not None:
                buffer.update_priorities(batch["indices"], td_errors.cpu().numpy())

            if step % train_cfg["log_freq"] == 0:
                metrics["train/epsilon"] = epsilon
                if use_per:
                    metrics["train/beta"] = beta
                logger.log_metrics(metrics, step=step)

        # ── target sync ───────────────────────────────────────────────────
        if step % config["agent"]["target_update_freq"] == 0:
            agent.sync_target()

        # ── evaluation ────────────────────────────────────────────────────
        if step % train_cfg["eval_freq"] == 0:
            eval_metrics = evaluate(agent, eval_env, train_cfg["eval_episodes"])
            logger.log_metrics(
                {f"eval/{k}": v for k, v in eval_metrics.items()},
                step=step,
            )
            pbar.set_postfix(
                eval_reward=f"{eval_metrics['reward_mean']:.1f}",
                epsilon=f"{epsilon:.3f}",
            )

        # ── checkpoint ────────────────────────────────────────────────────
        if step % train_cfg["checkpoint_freq"] == 0:
            ckpt_path = Path(config["paths"]["checkpoint_dir"]) / f"agent_step_{step}.pt"
            agent.save(ckpt_path)
            logger.log_artifact(ckpt_path)

    # ── cleanup ───────────────────────────────────────────────────────────
    final_path = Path(config["paths"]["checkpoint_dir"]) / "agent_final.pt"
    agent.save(final_path)
    logger.log_artifact(final_path)
    logger.end()
    train_env.close()
    eval_env.close()
