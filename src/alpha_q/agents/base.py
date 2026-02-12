"""Abstract base agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch


class BaseAgent(ABC):
    """Interface that all DQN agent variants must implement."""

    @abstractmethod
    def select_action(self, state: np.ndarray, *, eval_mode: bool = False) -> int:
        """Choose an action given a state.

        When *eval_mode* is True, the agent should act greedily
        (no exploration).
        """

    @abstractmethod
    def learn(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Perform one gradient step on a batch from the replay buffer.

        Returns a dict of metrics (e.g. ``{"loss": 0.5, "q_mean": 3.2}``).
        """

    @abstractmethod
    def sync_target(self) -> None:
        """Copy online network weights to the target network."""

    def save(self, path: str | Path) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        """Load model checkpoint."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state)

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return serializable state for checkpointing."""

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore agent state from a checkpoint dict."""
