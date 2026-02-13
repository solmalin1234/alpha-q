"""Double DQN agent (van Hasselt et al., 2016)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from alpha_q.agents import register
from alpha_q.agents.dqn import DQNAgent


@register("double_dqn")
class DoubleDQNAgent(DQNAgent):
    """Double DQN — decouple action selection from evaluation.

    The only change from vanilla DQN is in the target computation:
    the *online* network selects the best next action, and the *target*
    network evaluates its Q-value.  This reduces overestimation bias.
    """

    def learn(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Q(s, a) from online net
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: online net picks the action, target net evaluates it
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "train/loss": loss.item(),
            "train/q_mean": q_values.mean().item(),
        }
