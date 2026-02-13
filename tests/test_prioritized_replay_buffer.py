"""Tests for the prioritized replay buffer."""

from __future__ import annotations

import numpy as np
import torch

from alpha_q.memory.prioritized_replay_buffer import PrioritizedReplayBuffer


def _make_buffer(capacity: int = 100) -> PrioritizedReplayBuffer:
    return PrioritizedReplayBuffer(capacity=capacity, obs_shape=(2,), alpha=0.6)


def _push_n(buf: PrioritizedReplayBuffer, n: int) -> None:
    for i in range(n):
        state = np.array([i, i], dtype=np.uint8)
        buf.push(state, action=0, reward=1.0, next_state=state, done=False)


def test_push_and_len() -> None:
    buf = _make_buffer()
    assert len(buf) == 0
    _push_n(buf, 5)
    assert len(buf) == 5


def test_circular_overwrite() -> None:
    buf = _make_buffer(capacity=5)
    _push_n(buf, 10)
    assert len(buf) == 5
    assert buf.states[0, 0] == 5  # oldest after wrap


def test_sample_returns_correct_keys_and_shapes() -> None:
    buf = _make_buffer()
    _push_n(buf, 20)
    batch = buf.sample(8)
    assert batch["states"].shape == (8, 2)
    assert batch["actions"].shape == (8,)
    assert batch["rewards"].shape == (8,)
    assert batch["next_states"].shape == (8, 2)
    assert batch["dones"].shape == (8,)
    assert batch["weights"].shape == (8,)
    assert batch["indices"].shape == (8,)


def test_sample_dtypes() -> None:
    buf = _make_buffer()
    _push_n(buf, 5)
    batch = buf.sample(2)
    assert batch["states"].dtype == torch.uint8
    assert batch["actions"].dtype == torch.long
    assert batch["rewards"].dtype == torch.float32
    assert batch["weights"].dtype == torch.float32
    assert batch["indices"].dtype == np.int64


def test_weights_are_normalized() -> None:
    buf = _make_buffer()
    _push_n(buf, 20)
    batch = buf.sample(8, beta=0.4)
    # Max weight should be 1.0 after normalization
    assert abs(batch["weights"].max().item() - 1.0) < 1e-6
    # All weights should be positive
    assert (batch["weights"] > 0).all()


def test_update_priorities_changes_sampling() -> None:
    """After giving one transition a very high priority, it should
    dominate sampling."""
    buf = _make_buffer(capacity=10)
    _push_n(buf, 10)

    # Give index 0 a massive priority
    buf.update_priorities(np.array([0]), np.array([1000.0]))

    counts = np.zeros(10)
    for _ in range(500):
        batch = buf.sample(1)
        counts[batch["indices"][0]] += 1

    # Index 0 should be sampled overwhelmingly
    assert counts[0] > 400


def test_beta_zero_gives_uniform_weights() -> None:
    """With beta=0 there is no IS correction, so all weights equal 1."""
    buf = _make_buffer(capacity=10)
    _push_n(buf, 10)
    buf.update_priorities(np.array([0]), np.array([100.0]))

    batch = buf.sample(10, beta=0.0)
    assert torch.allclose(batch["weights"], torch.ones(10))


def test_higher_beta_downweights_high_priority() -> None:
    """Higher beta should more aggressively correct for oversampled items,
    giving the high-priority transition a lower weight."""
    buf = _make_buffer(capacity=10)
    _push_n(buf, 10)
    buf.update_priorities(np.array([0]), np.array([100.0]))

    batch_low = buf.sample(10, beta=0.1)
    batch_high = buf.sample(10, beta=1.0)

    # Find weight assigned to index 0 in each batch
    mask_low = batch_low["indices"] == 0
    mask_high = batch_high["indices"] == 0
    if mask_low.any() and mask_high.any():
        w_low = batch_low["weights"][mask_low][0].item()
        w_high = batch_high["weights"][mask_high][0].item()
        assert w_high < w_low
