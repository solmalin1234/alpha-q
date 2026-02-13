"""Rainbow agent (Hessel et al., 2018).

Combines six DQN improvements:
- Double DQN (action selection via online net)
- Prioritized Experience Replay (IS weights + td_errors)
- Dueling architecture (value + advantage streams)
- Multi-step returns (gamma^n)
- Noisy Nets (parameter-space exploration)
- Categorical DQN / C51 (distributional RL)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from alpha_q.agents import register
from alpha_q.agents.base import BaseAgent
from alpha_q.networks.rainbow_cnn import RainbowCNN


@register("rainbow")
class RainbowAgent(BaseAgent):
    """Rainbow: all six DQN improvements in a single agent."""

    noisy = True  # tells trainer to skip epsilon-greedy

    def __init__(self, config: dict, device: torch.device, n_actions: int) -> None:
        agent_cfg = config["agent"]
        in_channels = config["env"].get("frame_stack", 4)

        self.device = device
        self.gamma = agent_cfg["gamma"]
        self.n_step = config.get("replay", {}).get("n_step", 3)
        self.grad_clip = agent_cfg["grad_clip"]

        self.n_atoms = agent_cfg.get("n_atoms", 51)
        self.v_min = agent_cfg.get("v_min", -10.0)
        self.v_max = agent_cfg.get("v_max", 10.0)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

        self.online_net = RainbowCNN(
            in_channels, n_actions, self.n_atoms, self.v_min, self.v_max
        ).to(device)
        self.target_net = RainbowCNN(
            in_channels, n_actions, self.n_atoms, self.v_min, self.v_max
        ).to(device)
        self.sync_target()
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=agent_cfg["lr"])

    # -- action selection ----------------------------------------------------

    @torch.no_grad()
    def select_action(self, state: np.ndarray, *, eval_mode: bool = False) -> int:
        if eval_mode:
            self.online_net.eval()
        else:
            self.online_net.train()
            self.online_net.reset_noise()

        t = torch.as_tensor(state, device=self.device).unsqueeze(0)
        q_values = self.online_net.q_values(t)
        action = int(q_values.argmax(dim=1).item())

        self.online_net.train()
        return action

    # -- learning ------------------------------------------------------------

    def learn(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        batch_size = states.size(0)
        support = self.online_net.support  # (n_atoms,)

        # Resample noise for both nets
        self.online_net.reset_noise()
        self.target_net.reset_noise()

        # Current log-distribution for taken actions
        log_probs = self.online_net(states)  # (B, n_actions, n_atoms)
        log_probs_a = log_probs[range(batch_size), actions]  # (B, n_atoms)

        with torch.no_grad():
            # Double DQN: online net selects actions, target net evaluates
            # Note: do NOT reset online noise here — it would invalidate the
            # computation graph from the forward pass above.
            next_q_online = self.online_net.q_values(next_states)
            next_actions = next_q_online.argmax(dim=1)

            next_log_probs = self.target_net(next_states)
            next_probs = next_log_probs.exp()  # (B, n_actions, n_atoms)
            next_probs_a = next_probs[range(batch_size), next_actions]  # (B, n_atoms)

            # Project target distribution: Tz = r + gamma^n * z
            gamma_n = self.gamma**self.n_step
            Tz = rewards.unsqueeze(1) + gamma_n * (1.0 - dones.unsqueeze(1)) * support.unsqueeze(0)
            Tz = Tz.clamp(self.v_min, self.v_max)

            # Map onto support indices
            b = (Tz - self.v_min) / self.delta_z
            lo = b.floor().long().clamp(0, self.n_atoms - 1)
            hi = b.ceil().long().clamp(0, self.n_atoms - 1)

            # Distribute probability to lower and upper neighbours
            target_probs = torch.zeros_like(next_probs_a)
            offset = torch.arange(batch_size, device=states.device).unsqueeze(1) * self.n_atoms
            target_probs.view(-1).index_add_(
                0,
                (offset + lo).view(-1),
                (next_probs_a * (hi.float() - b)).view(-1),
            )
            target_probs.view(-1).index_add_(
                0,
                (offset + hi).view(-1),
                (next_probs_a * (b - lo.float())).view(-1),
            )

        # Cross-entropy loss per sample
        element_wise_loss = -(target_probs * log_probs_a).sum(dim=1)

        weights = batch.get("weights")
        if weights is not None:
            loss = (weights * element_wise_loss).mean()
        else:
            loss = element_wise_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Cross-entropy as TD-error proxy for PER
        td_errors = element_wise_loss.detach()

        # Q-values for logging
        with torch.no_grad():
            q_values = self.online_net.q_values(states)
            q_taken = q_values[range(batch_size), actions]

        return {
            "train/loss": loss.item(),
            "train/q_mean": q_taken.mean().item(),
            "td_errors": td_errors,
        }

    # -- target sync ---------------------------------------------------------

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    # -- checkpointing -------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.online_net.load_state_dict(state["online_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])
