"""Dueling DQN agent (Wang et al., 2016)."""

from __future__ import annotations

import torch

from alpha_q.agents import register
from alpha_q.agents.dqn import DQNAgent
from alpha_q.networks.dueling_cnn import DuelingCNN


@register("dueling_dqn")
class DuelingDQNAgent(DQNAgent):
    """Dueling DQN — separate value and advantage streams.

    Identical to vanilla DQN except the network uses a dueling
    architecture that decomposes Q(s, a) into V(s) + A(s, a).
    """

    def __init__(self, config: dict, device: torch.device, n_actions: int) -> None:
        # Skip DQNAgent.__init__ and set up with DuelingCNN instead
        agent_cfg = config["agent"]
        in_channels = config["env"].get("frame_stack", 4)

        self.device = device
        self.gamma = agent_cfg["gamma"]
        self.n_step = config.get("replay", {}).get("n_step", 1)
        self.grad_clip = agent_cfg["grad_clip"]

        self.online_net = DuelingCNN(in_channels, n_actions).to(device)
        self.target_net = DuelingCNN(in_channels, n_actions).to(device)
        self.sync_target()
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=agent_cfg["lr"])
