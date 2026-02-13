"""Noisy DQN agent (Fortunato et al., 2018)."""

from __future__ import annotations

import numpy as np
import torch

from alpha_q.agents import register
from alpha_q.agents.dqn import DQNAgent
from alpha_q.networks.noisy_cnn import NoisyCNN


@register("noisy_dqn")
class NoisyDQNAgent(DQNAgent):
    """DQN with NoisyNet exploration — no epsilon-greedy needed.

    Uses :class:`NoisyCNN` (factorized Gaussian noise in FC layers)
    for state-dependent exploration.  Noise is resampled before each
    action selection and each learning step.
    """

    noisy = True

    def __init__(self, config: dict, device: torch.device, n_actions: int) -> None:
        agent_cfg = config["agent"]
        in_channels = config["env"].get("frame_stack", 4)

        self.device = device
        self.gamma = agent_cfg["gamma"]
        self.n_step = config.get("replay", {}).get("n_step", 1)
        self.grad_clip = agent_cfg["grad_clip"]

        self.online_net = NoisyCNN(in_channels, n_actions).to(device)
        self.target_net = NoisyCNN(in_channels, n_actions).to(device)
        self.sync_target()
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=agent_cfg["lr"])

    @torch.no_grad()
    def select_action(self, state: np.ndarray, *, eval_mode: bool = False) -> int:
        if eval_mode:
            self.online_net.eval()
            t = torch.as_tensor(state, device=self.device).unsqueeze(0)
            q_values = self.online_net(t)
            self.online_net.train()
        else:
            self.online_net.reset_noise()
            t = torch.as_tensor(state, device=self.device).unsqueeze(0)
            q_values = self.online_net(t)
        return int(q_values.argmax(dim=1).item())

    def learn(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.online_net.reset_noise()
        self.target_net.reset_noise()
        return super().learn(batch)
