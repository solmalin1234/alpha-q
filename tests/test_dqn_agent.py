"""Tests for the vanilla DQN agent."""

from __future__ import annotations

import numpy as np
import torch

from alpha_q.agents import available_agents, create_agent
from alpha_q.agents.dqn import DQNAgent

# Minimal config matching the structure expected by DQNAgent.
_CFG = {
    "env": {"frame_stack": 4},
    "agent": {
        "type": "dqn",
        "gamma": 0.99,
        "lr": 1e-3,
        "grad_clip": 10.0,
        "batch_size": 4,
    },
}

_N_ACTIONS = 4
_DEVICE = torch.device("cpu")


def _make_agent() -> DQNAgent:
    return create_agent("dqn", config=_CFG, device=_DEVICE, n_actions=_N_ACTIONS)


def _random_batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "states": torch.randint(0, 255, (batch_size, 4, 84, 84), dtype=torch.uint8),
        "actions": torch.randint(0, _N_ACTIONS, (batch_size,)),
        "rewards": torch.randn(batch_size),
        "next_states": torch.randint(0, 255, (batch_size, 4, 84, 84), dtype=torch.uint8),
        "dones": torch.zeros(batch_size),
    }


# ── registry ──────────────────────────────────────────────────────────────


def test_dqn_registered() -> None:
    assert "dqn" in available_agents()


def test_create_via_registry() -> None:
    agent = _make_agent()
    assert isinstance(agent, DQNAgent)


# ── select_action ─────────────────────────────────────────────────────────


def test_select_action_returns_valid_action() -> None:
    agent = _make_agent()
    state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
    action = agent.select_action(state)
    assert 0 <= action < _N_ACTIONS


def test_select_action_deterministic() -> None:
    agent = _make_agent()
    state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
    a1 = agent.select_action(state, eval_mode=True)
    a2 = agent.select_action(state, eval_mode=True)
    assert a1 == a2


# ── learn ─────────────────────────────────────────────────────────────────


def test_learn_returns_metrics() -> None:
    agent = _make_agent()
    metrics = agent.learn(_random_batch())
    assert "train/loss" in metrics
    assert "train/q_mean" in metrics
    assert isinstance(metrics["train/loss"], float)


def test_learn_decreases_loss() -> None:
    agent = _make_agent()
    batch = _random_batch()
    # Override rewards so there's a clear signal
    batch["rewards"] = torch.ones(4) * 10.0

    first_loss = agent.learn(batch)["train/loss"]
    for _ in range(50):
        agent.learn(batch)
    later_loss = agent.learn(batch)["train/loss"]
    assert later_loss < first_loss


# ── sync_target ───────────────────────────────────────────────────────────


def test_sync_target_copies_weights() -> None:
    agent = _make_agent()
    # Perturb online weights so they differ from target
    for p in agent.online_net.parameters():
        p.data.add_(torch.randn_like(p))

    # They should differ now
    for p_on, p_tgt in zip(agent.online_net.parameters(), agent.target_net.parameters()):
        if not torch.equal(p_on, p_tgt):
            break
    else:
        raise AssertionError("Weights should differ before sync")

    agent.sync_target()

    for p_on, p_tgt in zip(agent.online_net.parameters(), agent.target_net.parameters()):
        assert torch.equal(p_on, p_tgt)


# ── state_dict / load_state_dict ──────────────────────────────────────────


def test_state_dict_roundtrip() -> None:
    agent = _make_agent()
    # Do a learning step to change optimizer state
    agent.learn(_random_batch())

    state = agent.state_dict()
    assert "online_net" in state
    assert "target_net" in state
    assert "optimizer" in state

    # Create a fresh agent and load the state
    agent2 = _make_agent()
    agent2.load_state_dict(state)

    for p1, p2 in zip(agent.online_net.parameters(), agent2.online_net.parameters()):
        assert torch.equal(p1, p2)

    for p1, p2 in zip(agent.target_net.parameters(), agent2.target_net.parameters()):
        assert torch.equal(p1, p2)
