"""Prioritized experience replay buffer (Schaul et al., 2016)."""

from __future__ import annotations

import numpy as np
import torch

from alpha_q.memory.sum_tree import SumTree


class PrioritizedReplayBuffer:
    """Proportional prioritized replay with importance-sampling weights.

    New transitions are inserted with the current maximum priority so they
    are guaranteed to be replayed at least once.  After each learning step
    the caller should invoke :meth:`update_priorities` with the absolute
    TD errors returned by the agent.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        alpha: float = 0.6,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.idx = 0
        self.size = 0

        self._tree = SumTree(capacity)
        self._max_priority = 1.0
        self._epsilon = 1e-6

        self.states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    # ── storage ───────────────────────────────────────────────────────────

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition with maximum priority."""
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done

        self._tree.update(self.idx, self._max_priority**self.alpha)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ── sampling ──────────────────────────────────────────────────────────

    def sample(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        beta: float = 0.4,
    ) -> dict[str, torch.Tensor | np.ndarray]:
        """Sample a prioritized mini-batch with IS weights.

        Returns the usual transition tensors plus ``"weights"`` (IS
        correction, shape ``(B,)``) and ``"indices"`` (numpy int64 array
        for :meth:`update_priorities`).
        """
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)

        segment = self._tree.total / batch_size
        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            cumsum = np.random.uniform(lo, hi)
            data_idx, priority = self._tree.get(cumsum)
            indices[i] = data_idx
            priorities[i] = priority

        # Importance-sampling weights
        probs = priorities / self._tree.total
        weights = (self.size * probs) ** (-beta)
        weights /= weights.max()  # normalise so max weight = 1

        return {
            "states": torch.as_tensor(self.states[indices], device=device),
            "actions": torch.as_tensor(self.actions[indices], dtype=torch.long, device=device),
            "rewards": torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device),
            "next_states": torch.as_tensor(self.next_states[indices], device=device),
            "dones": torch.as_tensor(self.dones[indices], dtype=torch.float32, device=device),
            "weights": torch.as_tensor(weights, dtype=torch.float32, device=device),
            "indices": indices,
        }

    # ── priority updates ──────────────────────────────────────────────────

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities using absolute TD errors."""
        clipped = np.abs(td_errors) + self._epsilon
        for idx, prio_raw in zip(indices, clipped):
            self._tree.update(int(idx), float(prio_raw**self.alpha))
        self._max_priority = max(self._max_priority, float(clipped.max()))

    def __len__(self) -> int:
        return self.size
