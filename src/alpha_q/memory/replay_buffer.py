"""Uniform experience replay buffer with uint8 observation storage."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size circular replay buffer storing observations as uint8.

    Stores transitions ``(state, action, reward, next_state, done)`` and
    returns random mini-batches as PyTorch tensors on a given device.
    """

    def __init__(self, capacity: int, obs_shape: tuple[int, ...]) -> None:
        self.capacity = capacity
        self.idx = 0
        self.size = 0

        self.states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition."""
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: torch.device = torch.device("cpu")
    ) -> dict[str, torch.Tensor]:
        """Sample a random mini-batch and return as tensors on *device*."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "states": torch.as_tensor(self.states[indices], device=device),
            "actions": torch.as_tensor(self.actions[indices], dtype=torch.long, device=device),
            "rewards": torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device),
            "next_states": torch.as_tensor(self.next_states[indices], device=device),
            "dones": torch.as_tensor(self.dones[indices], dtype=torch.float32, device=device),
        }

    def __len__(self) -> int:
        return self.size
