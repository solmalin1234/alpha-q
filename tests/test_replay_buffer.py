"""Tests for the replay buffer."""

from __future__ import annotations

import numpy as np
import torch

from alpha_q.memory.replay_buffer import ReplayBuffer


def test_push_and_len() -> None:
    buf = ReplayBuffer(capacity=100, obs_shape=(4, 84, 84))
    assert len(buf) == 0

    state = np.zeros((4, 84, 84), dtype=np.uint8)
    buf.push(state, action=0, reward=1.0, next_state=state, done=False)
    assert len(buf) == 1


def test_circular_overwrite() -> None:
    buf = ReplayBuffer(capacity=5, obs_shape=(2,))
    for i in range(10):
        state = np.array([i, i], dtype=np.uint8)
        buf.push(state, action=0, reward=0.0, next_state=state, done=False)

    assert len(buf) == 5
    # Buffer should contain the last 5 entries (indices 5-9)
    assert buf.states[0, 0] == 5  # oldest after wrap


def test_sample_returns_correct_shapes() -> None:
    obs_shape = (4, 84, 84)
    buf = ReplayBuffer(capacity=100, obs_shape=obs_shape)

    state = np.random.randint(0, 255, obs_shape, dtype=np.uint8)
    for _ in range(20):
        buf.push(state, action=1, reward=0.5, next_state=state, done=False)

    batch = buf.sample(8)
    assert batch["states"].shape == (8, *obs_shape)
    assert batch["actions"].shape == (8,)
    assert batch["rewards"].shape == (8,)
    assert batch["next_states"].shape == (8, *obs_shape)
    assert batch["dones"].shape == (8,)


def test_sample_dtypes() -> None:
    buf = ReplayBuffer(capacity=10, obs_shape=(2,))
    state = np.array([1, 2], dtype=np.uint8)
    buf.push(state, action=0, reward=1.0, next_state=state, done=True)

    batch = buf.sample(1)
    assert batch["states"].dtype == torch.uint8
    assert batch["actions"].dtype == torch.long
    assert batch["rewards"].dtype == torch.float32
    assert batch["dones"].dtype == torch.float32


def test_sample_values_preserved() -> None:
    buf = ReplayBuffer(capacity=10, obs_shape=(2,))
    state = np.array([42, 99], dtype=np.uint8)
    next_state = np.array([10, 20], dtype=np.uint8)
    buf.push(state, action=3, reward=-1.0, next_state=next_state, done=True)

    batch = buf.sample(1)
    assert torch.equal(batch["states"][0], torch.tensor([42, 99], dtype=torch.uint8))
    assert batch["actions"][0].item() == 3
    assert batch["rewards"][0].item() == -1.0
    assert torch.equal(batch["next_states"][0], torch.tensor([10, 20], dtype=torch.uint8))
    assert batch["dones"][0].item() == 1.0
