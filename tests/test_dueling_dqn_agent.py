"""Tests for the Dueling DQN agent."""

from __future__ import annotations

import numpy as np
import torch

from alpha_q.agents import available_agents, create_agent
from alpha_q.agents.dueling_dqn import DuelingDQNAgent
from alpha_q.networks.dueling_cnn import DuelingCNN

_CFG = {
    "env": {"frame_stack": 4},
    "agent": {
        "type": "dueling_dqn",
        "gamma": 0.99,
        "lr": 1e-3,
        "grad_clip": 10.0,
        "batch_size": 4,
    },
}

_N_ACTIONS = 4
_DEVICE = torch.device("cpu")


def _make_agent() -> DuelingDQNAgent:
    return create_agent("dueling_dqn", config=_CFG, device=_DEVICE, n_actions=_N_ACTIONS)


def _random_batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "states": torch.randint(0, 255, (batch_size, 4, 84, 84), dtype=torch.uint8),
        "actions": torch.randint(0, _N_ACTIONS, (batch_size,)),
        "rewards": torch.randn(batch_size),
        "next_states": torch.randint(0, 255, (batch_size, 4, 84, 84), dtype=torch.uint8),
        "dones": torch.zeros(batch_size),
    }


# ── registry ──────────────────────────────────────────────────────────────


def test_dueling_dqn_registered() -> None:
    assert "dueling_dqn" in available_agents()


def test_create_via_registry() -> None:
    agent = _make_agent()
    assert isinstance(agent, DuelingDQNAgent)


# ── network architecture ─────────────────────────────────────────────────


def test_uses_dueling_network() -> None:
    agent = _make_agent()
    assert isinstance(agent.online_net, DuelingCNN)
    assert isinstance(agent.target_net, DuelingCNN)


def test_dueling_output_shape() -> None:
    net = DuelingCNN(in_channels=4, n_actions=_N_ACTIONS)
    x = torch.randint(0, 255, (2, 4, 84, 84), dtype=torch.uint8)
    out = net(x)
    assert out.shape == (2, _N_ACTIONS)


def test_advantage_mean_centering() -> None:
    """Q values should satisfy: mean_a(Q(s,a)) == V(s) for each state,
    since mean_a(A - mean(A)) == 0."""
    net = DuelingCNN(in_channels=4, n_actions=_N_ACTIONS)
    x = torch.randint(0, 255, (3, 4, 84, 84), dtype=torch.uint8)
    with torch.no_grad():
        q = net(x)
        # The mean of Q across actions should equal V(s)
        features = net.conv(x.float() / 255.0).reshape(3, -1)
        v = net.value_stream(features)
        assert torch.allclose(q.mean(dim=1, keepdim=True), v, atol=1e-5)


# ── select_action ─────────────────────────────────────────────────────────


def test_select_action_returns_valid_action() -> None:
    agent = _make_agent()
    state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
    action = agent.select_action(state)
    assert 0 <= action < _N_ACTIONS


# ── learn ─────────────────────────────────────────────────────────────────


def test_learn_returns_metrics() -> None:
    agent = _make_agent()
    metrics = agent.learn(_random_batch())
    assert "train/loss" in metrics
    assert "train/q_mean" in metrics
    assert "td_errors" in metrics


def test_learn_decreases_loss() -> None:
    agent = _make_agent()
    batch = _random_batch()
    batch["rewards"] = torch.ones(4) * 10.0

    first_loss = agent.learn(batch)["train/loss"]
    for _ in range(50):
        agent.learn(batch)
    later_loss = agent.learn(batch)["train/loss"]
    assert later_loss < first_loss


# ── checkpointing ─────────────────────────────────────────────────────────


def test_state_dict_roundtrip() -> None:
    agent = _make_agent()
    agent.learn(_random_batch())

    state = agent.state_dict()
    agent2 = _make_agent()
    agent2.load_state_dict(state)

    for p1, p2 in zip(agent.online_net.parameters(), agent2.online_net.parameters()):
        assert torch.equal(p1, p2)
