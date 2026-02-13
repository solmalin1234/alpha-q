"""Dueling network architecture (Wang et al., 2016)."""

from __future__ import annotations

import torch
import torch.nn as nn


class DuelingCNN(nn.Module):
    """Nature CNN backbone with a dueling head (value + advantage streams).

    Shares the same convolutional trunk as :class:`NatureCNN` but splits
    the fully-connected layers into a state-value stream *V(s)* and an
    action-advantage stream *A(s, a)*.  The streams are combined as:

        Q(s, a) = V(s) + A(s, a) - mean_a'(A(s, a'))

    Expects uint8 input of shape ``(B, C, 84, 84)`` and normalises to
    ``[0, 1]`` in the forward pass.
    """

    def __init__(self, in_channels: int, n_actions: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # 64 * 7 * 7 = 3136 for 84x84 input
        self.value_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # Combine: Q = V + (A - mean(A))
        return value + advantage - advantage.mean(dim=1, keepdim=True)
