"""Tests for the Rainbow agent."""

from __future__ import annotations

import numpy as np
import torch

from alpha_q.agents import available_agents, create_agent
from alpha_q.agents.rainbow import RainbowAgent
from alpha_q.networks.noisy_linear import NoisyLinear
from alpha_q.networks.rainbow_cnn import RainbowCNN

_N_ACTIONS = 4
_N_ATOMS = 11  # small for fast tests
_V_MIN = -5.0
_V_MAX = 5.0
_DEVICE = torch.device("cpu")

_CFG = {
    "env": {"frame_stack": 4},
    "agent": {
        "type": "rainbow",
        "gamma": 0.99,
        "lr": 1e-3,
        "grad_clip": 10.0,
        "batch_size": 4,
        "n_atoms": _N_ATOMS,
        "v_min": _V_MIN,
        "v_max": _V_MAX,
    },
    "replay": {"n_step": 3},
}


def _make_agent() -> RainbowAgent:
    return create_agent("rainbow", config=_CFG, device=_DEVICE, n_actions=_N_ACTIONS)


def _random_batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "states": torch.randint(0, 255, (batch_size, 4, 84, 84), dtype=torch.uint8),
        "actions": torch.randint(0, _N_ACTIONS, (batch_size,)),
        "rewards": torch.randn(batch_size),
        "next_states": torch.randint(0, 255, (batch_size, 4, 84, 84), dtype=torch.uint8),
        "dones": torch.zeros(batch_size),
    }


# -- RainbowCNN ---------------------------------------------------------------


def test_rainbow_cnn_log_probs_shape() -> None:
    net = RainbowCNN(4, _N_ACTIONS, _N_ATOMS, _V_MIN, _V_MAX)
    x = torch.randint(0, 255, (2, 4, 84, 84), dtype=torch.uint8)
    log_probs = net(x)
    assert log_probs.shape == (2, _N_ACTIONS, _N_ATOMS)


def test_rainbow_cnn_probs_sum_to_one() -> None:
    net = RainbowCNN(4, _N_ACTIONS, _N_ATOMS, _V_MIN, _V_MAX)
    x = torch.randint(0, 255, (3, 4, 84, 84), dtype=torch.uint8)
    probs = net(x).exp()
    sums = probs.sum(dim=2)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_rainbow_cnn_q_values_shape() -> None:
    net = RainbowCNN(4, _N_ACTIONS, _N_ATOMS, _V_MIN, _V_MAX)
    x = torch.randint(0, 255, (2, 4, 84, 84), dtype=torch.uint8)
    q = net.q_values(x)
    assert q.shape == (2, _N_ACTIONS)


def test_rainbow_cnn_has_noisy_layers() -> None:
    net = RainbowCNN(4, _N_ACTIONS, _N_ATOMS, _V_MIN, _V_MAX)
    noisy_count = sum(1 for m in net.modules() if isinstance(m, NoisyLinear))
    assert noisy_count == 4  # 2 per stream (fc + out)


def test_rainbow_cnn_has_dueling_streams() -> None:
    net = RainbowCNN(4, _N_ACTIONS, _N_ATOMS, _V_MIN, _V_MAX)
    assert hasattr(net, "value_fc")
    assert hasattr(net, "value_out")
    assert hasattr(net, "advantage_fc")
    assert hasattr(net, "advantage_out")


def test_rainbow_cnn_support_range() -> None:
    net = RainbowCNN(4, _N_ACTIONS, _N_ATOMS, _V_MIN, _V_MAX)
    assert net.support[0].item() == _V_MIN
    assert net.support[-1].item() == _V_MAX
    assert net.support.shape == (_N_ATOMS,)


def test_rainbow_cnn_reset_noise_changes_output() -> None:
    net = RainbowCNN(4, _N_ACTIONS, _N_ATOMS, _V_MIN, _V_MAX)
    net.train()
    x = torch.randint(0, 255, (1, 4, 84, 84), dtype=torch.uint8)
    out1 = net(x).clone()
    net.reset_noise()
    out2 = net(x)
    # Very unlikely to be identical after noise resample
    assert not torch.equal(out1, out2)


def test_rainbow_cnn_eval_deterministic() -> None:
    net = RainbowCNN(4, _N_ACTIONS, _N_ATOMS, _V_MIN, _V_MAX)
    net.eval()
    x = torch.randint(0, 255, (1, 4, 84, 84), dtype=torch.uint8)
    out1 = net(x)
    out2 = net(x)
    assert torch.equal(out1, out2)


# -- registry ------------------------------------------------------------------


def test_rainbow_registered() -> None:
    assert "rainbow" in available_agents()


def test_create_via_registry() -> None:
    agent = _make_agent()
    assert isinstance(agent, RainbowAgent)


def test_uses_rainbow_network() -> None:
    agent = _make_agent()
    assert isinstance(agent.online_net, RainbowCNN)
    assert isinstance(agent.target_net, RainbowCNN)


def test_rainbow_is_noisy() -> None:
    agent = _make_agent()
    assert agent.noisy is True


# -- select_action -------------------------------------------------------------


def test_select_action_returns_valid_action() -> None:
    agent = _make_agent()
    state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
    action = agent.select_action(state)
    assert 0 <= action < _N_ACTIONS


def test_select_action_deterministic_in_eval() -> None:
    agent = _make_agent()
    state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
    a1 = agent.select_action(state, eval_mode=True)
    a2 = agent.select_action(state, eval_mode=True)
    assert a1 == a2


# -- learn ---------------------------------------------------------------------


def test_learn_returns_metrics() -> None:
    agent = _make_agent()
    metrics = agent.learn(_random_batch())
    assert "train/loss" in metrics
    assert "train/q_mean" in metrics
    assert "td_errors" in metrics
    assert isinstance(metrics["train/loss"], float)


def test_learn_decreases_loss() -> None:
    agent = _make_agent()
    batch = _random_batch()

    first_loss = agent.learn(batch)["train/loss"]
    for _ in range(50):
        agent.learn(batch)
    later_loss = agent.learn(batch)["train/loss"]
    assert later_loss < first_loss


def test_learn_with_per_weights() -> None:
    agent = _make_agent()
    batch = _random_batch()
    batch["weights"] = torch.ones(4)
    batch["indices"] = np.arange(4)
    metrics = agent.learn(batch)
    assert "train/loss" in metrics


def test_learn_uses_n_step() -> None:
    agent = _make_agent()
    assert agent.n_step == 3


# -- sync_target ---------------------------------------------------------------


def test_sync_target_copies_weights() -> None:
    agent = _make_agent()
    for p in agent.online_net.parameters():
        p.data.add_(torch.randn_like(p))

    agent.sync_target()

    for p_on, p_tgt in zip(agent.online_net.parameters(), agent.target_net.parameters()):
        assert torch.equal(p_on, p_tgt)


# -- state_dict / load_state_dict ---------------------------------------------


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
