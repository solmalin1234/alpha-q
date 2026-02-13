"""Categorical (C51) network architecture (Bellemare et al., 2017)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalCNN(nn.Module):
    """NatureCNN backbone that outputs a categorical return distribution.

    Instead of scalar Q-values, outputs a probability distribution over
    ``n_atoms`` fixed support atoms for each action.  Q-values are
    recovered as the expectation: ``Q(s, a) = sum_i z_i * p_i(s, a)``.

    The forward pass returns **log-probabilities** (shape
    ``(B, n_actions, n_atoms)``) for numerical stability during
    cross-entropy loss computation.
    """

    def __init__(
        self,
        in_channels: int,
        n_actions: int,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions * n_atoms),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities of shape ``(B, n_actions, n_atoms)``."""
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        logits = self.fc(x).view(-1, self.n_actions, self.n_atoms)
        return F.log_softmax(logits, dim=2)

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Return expected Q-values of shape ``(B, n_actions)``."""
        log_probs = self.forward(x)
        probs = log_probs.exp()
        return (probs * self.support).sum(dim=2)
