"""Neural network architecture for the Catan AI agent.

Uses a shared backbone with dual heads:
- Policy head: outputs action probabilities (masked to legal actions)
- Value head: estimates expected return from current state
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import _ACTION_SPACE_SIZE, _STATE_FEATURE_SIZE


class CatanNetwork(nn.Module):
    """Actor-Critic network for Catan."""

    def __init__(
        self,
        state_size: int = _STATE_FEATURE_SIZE,
        action_size: int = _ACTION_SPACE_SIZE,
        hidden_size: int = 512,
        num_layers: int = 3,
    ):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Shared backbone
        layers = []
        in_size = state_size
        for i in range(num_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            if i < num_layers - 1:
                layers.append(nn.Dropout(0.1))
            in_size = hidden_size
        self.backbone = nn.Sequential(*layers)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Tanh(),  # Value in [-1, 1]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01 if m.out_features == self.action_size else 1.0)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch, state_size)
            action_mask: (batch, action_size) binary mask of legal actions

        Returns:
            log_probs: (batch, action_size) log probabilities
            value: (batch, 1) state value estimate
        """
        features = self.backbone(state)

        # Policy
        logits = self.policy_head(features)
        if action_mask is not None:
            # Set illegal actions to very negative logit
            logits = logits + (action_mask.log().clamp(min=-1e8))
        log_probs = F.log_softmax(logits, dim=-1)

        # Value
        value = self.value_head(features)

        return log_probs, value

    def get_action(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """Sample an action from the policy.

        Returns:
            action_idx: sampled action index
            log_prob: log probability of the action
            value: estimated state value
        """
        with torch.no_grad():
            log_probs, value = self.forward(state.unsqueeze(0), action_mask.unsqueeze(0))
            probs = log_probs.exp().squeeze(0)

            if deterministic:
                action_idx = probs.argmax().item()
            else:
                action_idx = torch.multinomial(probs, 1).item()

            return action_idx, log_probs[0, action_idx].item(), value.item()
