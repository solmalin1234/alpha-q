"""Tests for the n-step return buffer wrapper."""

from __future__ import annotations

import numpy as np
import torch

from alpha_q.memory.n_step_buffer import NStepBuffer
from alpha_q.memory.replay_buffer import ReplayBuffer

GAMMA = 0.99


def _make_buffer(n: int = 3, capacity: int = 100) -> NStepBuffer:
    inner = ReplayBuffer(capacity=capacity, obs_shape=(2,))
    return NStepBuffer(inner, n=n, gamma=GAMMA)


def _state(i: int) -> np.ndarray:
    return np.array([i, i], dtype=np.uint8)


# ── basic n-step return ───────────────────────────────────────────────────


def test_nstep_return_computed_correctly() -> None:
    """After n pushes (no done), the inner buffer should hold one
    transition with the correct discounted return."""
    buf = _make_buffer(n=3)

    buf.push(_state(0), 0, 1.0, _state(1), False)
    buf.push(_state(1), 0, 2.0, _state(2), False)
    buf.push(_state(2), 0, 3.0, _state(3), False)

    assert len(buf) == 1
    batch = buf.sample(1)
    expected_R = 1.0 + GAMMA * 2.0 + GAMMA**2 * 3.0
    assert abs(batch["rewards"][0].item() - expected_R) < 1e-5

    # State should be s_0, next_state should be s_3
    assert batch["states"][0, 0].item() == 0
    assert batch["next_states"][0, 0].item() == 3


def test_nstep_done_false_flag() -> None:
    """Non-terminal n-step transition should have done=0."""
    buf = _make_buffer(n=2)
    buf.push(_state(0), 0, 1.0, _state(1), False)
    buf.push(_state(1), 0, 1.0, _state(2), False)

    batch = buf.sample(1)
    assert batch["dones"][0].item() == 0.0


# ── episode boundary flushing ─────────────────────────────────────────────


def test_flush_on_done_before_n() -> None:
    """If the episode ends before n steps, remaining entries are flushed
    with shorter returns."""
    buf = _make_buffer(n=3)

    buf.push(_state(0), 0, 1.0, _state(1), False)
    buf.push(_state(1), 0, 2.0, _state(2), True)

    # Both transitions should have been flushed
    assert len(buf) == 2

    batch = buf.sample(2)
    # All flushed entries should be terminal
    assert (batch["dones"] == 1.0).all()


def test_flush_on_done_at_n() -> None:
    """When done arrives exactly at step n, we get n committed entries."""
    buf = _make_buffer(n=3)

    buf.push(_state(0), 0, 1.0, _state(1), False)
    buf.push(_state(1), 0, 1.0, _state(2), False)
    buf.push(_state(2), 0, 1.0, _state(3), True)

    # 1 from commit (when deque hit n) + 2 from flush
    assert len(buf) == 3


def test_flush_rewards_correct() -> None:
    """Check that flushed (shorter) returns have correct discounting."""
    buf = _make_buffer(n=3)

    buf.push(_state(0), 0, 1.0, _state(1), False)
    buf.push(_state(1), 0, 2.0, _state(2), True)

    # Entry 0: R = 1.0 + γ*2.0
    # Entry 1: R = 2.0
    inner = buf.buffer
    assert abs(inner.rewards[0] - (1.0 + GAMMA * 2.0)) < 1e-5
    assert abs(inner.rewards[1] - 2.0) < 1e-5


# ── n=1 degenerates to standard buffer ────────────────────────────────────


def test_n_step_one_is_identity() -> None:
    """With n=1, every push goes straight through unchanged."""
    buf = _make_buffer(n=1)

    for i in range(5):
        buf.push(_state(i), i, float(i), _state(i + 1), i == 4)

    assert len(buf) == 5
    # Rewards should be raw (no discounting)
    inner = buf.buffer
    for i in range(5):
        assert inner.rewards[i] == float(i)


# ── multiple episodes ─────────────────────────────────────────────────────


def test_multiple_episodes() -> None:
    """Buffer should reset its deque between episodes."""
    buf = _make_buffer(n=3)

    # Episode 1: 2 steps
    buf.push(_state(0), 0, 1.0, _state(1), False)
    buf.push(_state(1), 0, 1.0, _state(2), True)
    count_after_ep1 = len(buf)

    # Episode 2: 4 steps
    buf.push(_state(10), 0, 1.0, _state(11), False)
    buf.push(_state(11), 0, 1.0, _state(12), False)
    buf.push(_state(12), 0, 1.0, _state(13), False)
    buf.push(_state(13), 0, 1.0, _state(14), True)

    # Ep1: 2 flushed entries. Ep2: 1 commit + 3 flushed = 4
    assert len(buf) == count_after_ep1 + 4


# ── delegation ────────────────────────────────────────────────────────────


def test_sample_delegates() -> None:
    """sample() should return valid batches from the inner buffer."""
    buf = _make_buffer(n=2)
    for i in range(10):
        buf.push(_state(i), 0, 1.0, _state(i + 1), i == 9)

    batch = buf.sample(4, device=torch.device("cpu"))
    assert batch["states"].shape == (4, 2)
    assert batch["rewards"].shape == (4,)
