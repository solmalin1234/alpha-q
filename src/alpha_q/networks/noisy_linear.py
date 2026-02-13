"""Noisy linear layer (Fortunato et al., 2018)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _factorized_noise(size: int) -> torch.Tensor:
    """Factorized Gaussian noise: f(x) = sign(x) * sqrt(|x|)."""
    x = torch.randn(size)
    return x.sign() * x.abs().sqrt()


class NoisyLinear(nn.Module):
    """Linear layer with learnable factorized Gaussian noise.

    During training the effective weight is ``mu + sigma * epsilon``
    where ``epsilon`` is resampled via :meth:`reset_noise`.  In eval
    mode the noise is dropped and only ``mu`` is used.
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("epsilon_weight", torch.empty(out_features, in_features))

        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))
        self.register_buffer("epsilon_bias", torch.empty(out_features))

        self.sigma_init = sigma_init
        self._init_parameters()
        self.reset_noise()

    def _init_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        self.mu_weight.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_weight.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma_init / math.sqrt(self.in_features))

    def reset_noise(self) -> None:
        """Resample factorized Gaussian noise."""
        epsilon_in = _factorized_noise(self.in_features)
        epsilon_out = _factorized_noise(self.out_features)
        self.epsilon_weight.copy_(epsilon_out.outer(epsilon_in))
        self.epsilon_bias.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.mu_weight + self.sigma_weight * self.epsilon_weight
            bias = self.mu_bias + self.sigma_bias * self.epsilon_bias
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        return F.linear(x, weight, bias)
