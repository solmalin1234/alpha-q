"""Tests for the Noisy DQN agent and NoisyLinear layer."""

from __future__ import annotations

import numpy as np
import torch

from alpha_q.agents import available_agents, create_agent
from alpha_q.agents.noisy_dqn import NoisyDQNAgent
from alpha_q.networks.noisy_cnn import NoisyCNN
from alpha_q.networks.noisy_linear import NoisyLinear

_CFG = {
    "env": {"frame_stack": 4},
    "agent": {
        "type": "noisy_dqn",
        "gamma": 0.99,
        "lr": 1e-3,
        "grad_clip": 10.0,
        "batch_size": 4,
    },
}

_N_ACTIONS = 4
_DEVICE = torch.device("cpu")


def _make_agent() -> NoisyDQNAgent:
    return create_agent("noisy_dqn", config=_CFG, device=_DEVICE, n_actions=_N_ACTIONS)


def _random_batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "states": torch.randint(0, 255, (batch_size, 4, 84, 84), dtype=torch.uint8),
        "actions": torch.randint(0, _N_ACTIONS, (batch_size,)),
        "rewards": torch.randn(batch_size),
        "next_states": torch.randint(0, 255, (batch_size, 4, 84, 84), dtype=torch.uint8),
        "dones": torch.zeros(batch_size),
    }


# ── NoisyLinear ───────────────────────────────────────────────────────────


def test_noisy_linear_output_shape() -> None:
    layer = NoisyLinear(32, 16)
    x = torch.randn(4, 32)
    assert layer(x).shape == (4, 16)


def test_noisy_linear_different_noise_per_reset() -> None:
    layer = NoisyLinear(32, 16)
    x = torch.randn(1, 32)
    layer.train()
    out1 = layer(x).clone()
    layer.reset_noise()
    out2 = layer(x)
    # With different noise samples, outputs should differ
    assert not torch.equal(out1, out2)


def test_noisy_linear_eval_is_deterministic() -> None:
    layer = NoisyLinear(32, 16)
    layer.eval()
    x = torch.randn(1, 32)
    out1 = layer(x).clone()
    layer.reset_noise()  # should not affect eval output
    out2 = layer(x)
    assert torch.equal(out1, out2)


# ── NoisyCNN ──────────────────────────────────────────────────────────────


def test_noisy_cnn_output_shape() -> None:
    net = NoisyCNN(in_channels=4, n_actions=_N_ACTIONS)
    x = torch.randint(0, 255, (2, 4, 84, 84), dtype=torch.uint8)
    assert net(x).shape == (2, _N_ACTIONS)


def test_noisy_cnn_reset_noise_changes_output() -> None:
    net = NoisyCNN(in_channels=4, n_actions=_N_ACTIONS)
    net.train()
    x = torch.randint(0, 255, (1, 4, 84, 84), dtype=torch.uint8)
    out1 = net(x).clone()
    net.reset_noise()
    out2 = net(x)
    assert not torch.equal(out1, out2)


# ── registry ──────────────────────────────────────────────────────────────


def test_noisy_dqn_registered() -> None:
    assert "noisy_dqn" in available_agents()


def test_create_via_registry() -> None:
    agent = _make_agent()
    assert isinstance(agent, NoisyDQNAgent)


def test_noisy_flag() -> None:
    agent = _make_agent()
    assert agent.noisy is True


def test_uses_noisy_network() -> None:
    agent = _make_agent()
    assert isinstance(agent.online_net, NoisyCNN)
    assert isinstance(agent.target_net, NoisyCNN)


# ── select_action ─────────────────────────────────────────────────────────


def test_select_action_returns_valid_action() -> None:
    agent = _make_agent()
    state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
    action = agent.select_action(state)
    assert 0 <= action < _N_ACTIONS


def test_select_action_eval_deterministic() -> None:
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
