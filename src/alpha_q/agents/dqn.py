"""Vanilla DQN agent (Mnih et al., 2015)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from alpha_q.agents import register
from alpha_q.agents.base import BaseAgent
from alpha_q.networks.cnn import NatureCNN


@register("dqn")
class DQNAgent(BaseAgent):
    """Deep Q-Network agent with a target network and Huber loss."""

    def __init__(self, config: dict, device: torch.device, n_actions: int) -> None:
        agent_cfg = config["agent"]
        in_channels = config["env"].get("frame_stack", 4)

        self.device = device
        self.gamma = agent_cfg["gamma"]
        self.n_step = config.get("replay", {}).get("n_step", 1)
        self.grad_clip = agent_cfg["grad_clip"]

        self.online_net = NatureCNN(in_channels, n_actions).to(device)
        self.target_net = NatureCNN(in_channels, n_actions).to(device)
        self.sync_target()
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=agent_cfg["lr"])

    # ── action selection ──────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, state: np.ndarray, *, eval_mode: bool = False) -> int:
        t = torch.as_tensor(state, device=self.device).unsqueeze(0)
        q_values = self.online_net(t)
        return int(q_values.argmax(dim=1).item())

    # ── learning ──────────────────────────────────────────────────────────

    def learn(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Q(s, a) from online net
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            target = rewards + (self.gamma**self.n_step) * next_q * (1.0 - dones)

        td_errors = (q_values - target).detach().abs()
        element_wise_loss = F.smooth_l1_loss(q_values, target, reduction="none")

        # Weight by importance-sampling correction when using PER
        weights = batch.get("weights")
        if weights is not None:
            loss = (weights * element_wise_loss).mean()
        else:
            loss = element_wise_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "train/loss": loss.item(),
            "train/q_mean": q_values.mean().item(),
            "td_errors": td_errors,
        }

    # ── target sync ───────────────────────────────────────────────────────

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ── checkpointing ─────────────────────────────────────────────────────

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
