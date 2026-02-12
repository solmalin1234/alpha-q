"""Nature CNN architecture (Mnih et al. 2015)."""

from __future__ import annotations

import torch
import torch.nn as nn


class NatureCNN(nn.Module):
    """The convolutional network from *Human-level control through deep
    reinforcement learning* (Mnih et al., 2015).

    Expects uint8 input of shape ``(B, C, 84, 84)`` and normalises to
    ``[0, 1]`` in the forward pass to save memory in the replay buffer.

    Architecture:
        conv(8x8, stride 4, 32) → ReLU →
        conv(4x4, stride 2, 64) → ReLU →
        conv(3x3, stride 1, 64) → ReLU →
        flatten → linear(3136, 512) → ReLU →
        linear(512, n_actions)
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
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise uint8 → float32 in [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)
