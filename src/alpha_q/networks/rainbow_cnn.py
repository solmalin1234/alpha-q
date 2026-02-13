"""Rainbow network: Dueling + NoisyNets + Categorical (Hessel et al., 2018)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from alpha_q.networks.noisy_linear import NoisyLinear


class RainbowCNN(nn.Module):
    """NatureCNN with dueling streams, noisy layers, and categorical output.

    Combines three architectural improvements:
    - **Dueling** (Wang et al., 2016): separate value and advantage streams
    - **NoisyNets** (Fortunato et al., 2018): NoisyLinear replaces nn.Linear
    - **Categorical** (Bellemare et al., 2017): distributional output over atoms

    Each stream outputs ``n_atoms`` values, combined with the dueling formula
    applied per-atom, then normalised via log-softmax.

    Forward pass returns **log-probabilities** of shape
    ``(B, n_actions, n_atoms)``.
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

        # Value stream: outputs n_atoms (one distribution for V(s))
        self.value_fc = NoisyLinear(3136, 512)
        self.value_out = NoisyLinear(512, n_atoms)

        # Advantage stream: outputs n_actions * n_atoms
        self.advantage_fc = NoisyLinear(3136, 512)
        self.advantage_out = NoisyLinear(512, n_actions * n_atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities of shape ``(B, n_actions, n_atoms)``."""
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)

        # Value stream → (B, 1, n_atoms)
        value = F.relu(self.value_fc(x))
        value = self.value_out(value).view(-1, 1, self.n_atoms)

        # Advantage stream → (B, n_actions, n_atoms)
        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage_out(advantage).view(-1, self.n_actions, self.n_atoms)

        # Dueling combination per atom: Q = V + A - mean(A)
        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.log_softmax(logits, dim=2)

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Return expected Q-values of shape ``(B, n_actions)``."""
        log_probs = self.forward(x)
        probs = log_probs.exp()
        return (probs * self.support).sum(dim=2)

    def reset_noise(self) -> None:
        """Resample noise in all NoisyLinear layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
