"""Tests for the Double DQN agent."""

from __future__ import annotations

import numpy as np
import torch

from alpha_q.agents import available_agents, create_agent
from alpha_q.agents.double_dqn import DoubleDQNAgent

_CFG = {
    "env": {"frame_stack": 4},
    "agent": {
        "type": "double_dqn",
        "gamma": 0.99,
        "lr": 1e-3,
        "grad_clip": 10.0,
        "batch_size": 4,
    },
}

_N_ACTIONS = 4
_DEVICE = torch.device("cpu")


def _make_agent() -> DoubleDQNAgent:
    return create_agent("double_dqn", config=_CFG, device=_DEVICE, n_actions=_N_ACTIONS)


def _random_batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "states": torch.randint(0, 255, (batch_size, 4, 84, 84), dtype=torch.uint8),
        "actions": torch.randint(0, _N_ACTIONS, (batch_size,)),
        "rewards": torch.randn(batch_size),
        "next_states": torch.randint(0, 255, (batch_size, 4, 84, 84), dtype=torch.uint8),
        "dones": torch.zeros(batch_size),
    }


# ── registry ──────────────────────────────────────────────────────────────


def test_double_dqn_registered() -> None:
    assert "double_dqn" in available_agents()


def test_create_via_registry() -> None:
    agent = _make_agent()
    assert isinstance(agent, DoubleDQNAgent)


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
    assert isinstance(metrics["train/loss"], float)


def test_learn_decreases_loss() -> None:
    agent = _make_agent()
    batch = _random_batch()
    batch["rewards"] = torch.ones(4) * 10.0

    first_loss = agent.learn(batch)["train/loss"]
    for _ in range(50):
        agent.learn(batch)
    later_loss = agent.learn(batch)["train/loss"]
    assert later_loss < first_loss


def test_learn_uses_online_net_for_action_selection() -> None:
    """Verify Double DQN target differs from vanilla DQN target.

    After perturbing the target net, Double DQN uses the online net to
    pick the best next action (then evaluates with target), while vanilla
    DQN would just take the max over the target net.  With different
    online/target weights the two approaches should produce different
    Q-targets and therefore different losses.
    """
    agent = _make_agent()
    batch = _random_batch()

    # Make online and target nets diverge
    for p in agent.target_net.parameters():
        p.data.add_(torch.randn_like(p) * 0.5)

    # Compute Double DQN loss
    double_metrics = agent.learn(batch)

    # Compute what vanilla DQN loss would be on the same batch
    with torch.no_grad():
        q_values = (
            agent.online_net(batch["states"]).gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
        )
        # Vanilla target: max over target net
        vanilla_next_q = agent.target_net(batch["next_states"]).max(dim=1).values
        vanilla_target = batch["rewards"] + agent.gamma * vanilla_next_q * (1.0 - batch["dones"])
        vanilla_loss = torch.nn.functional.smooth_l1_loss(q_values, vanilla_target).item()

    # The losses should differ because the action selection method differs
    assert double_metrics["train/loss"] != vanilla_loss


# ── sync_target ───────────────────────────────────────────────────────────


def test_sync_target_copies_weights() -> None:
    agent = _make_agent()
    for p in agent.online_net.parameters():
        p.data.add_(torch.randn_like(p))

    agent.sync_target()

    for p_on, p_tgt in zip(agent.online_net.parameters(), agent.target_net.parameters()):
        assert torch.equal(p_on, p_tgt)


# ── state_dict / load_state_dict ──────────────────────────────────────────


def test_state_dict_roundtrip() -> None:
    agent = _make_agent()
    agent.learn(_random_batch())

    state = agent.state_dict()
    assert "online_net" in state
    assert "target_net" in state
    assert "optimizer" in state

    agent2 = _make_agent()
    agent2.load_state_dict(state)

    for p1, p2 in zip(agent.online_net.parameters(), agent2.online_net.parameters()):
        assert torch.equal(p1, p2)

    for p1, p2 in zip(agent.target_net.parameters(), agent2.target_net.parameters()):
        assert torch.equal(p1, p2)
