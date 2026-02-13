"""Nature CNN with NoisyLinear FC layers (Fortunato et al., 2018)."""

from __future__ import annotations

import torch
import torch.nn as nn

from alpha_q.networks.noisy_linear import NoisyLinear


class NoisyCNN(nn.Module):
    """NatureCNN backbone with noisy fully-connected layers.

    Same convolutional trunk as :class:`NatureCNN` but the two FC
    layers are replaced with :class:`NoisyLinear` for parameter-space
    exploration.

    Call :meth:`reset_noise` to resample noise before each forward pass.
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
        self.fc = nn.Sequential(
            NoisyLinear(3136, 512),
            nn.ReLU(),
            NoisyLinear(512, n_actions),
        )

    def reset_noise(self) -> None:
        """Resample noise in all NoisyLinear layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)
