"""PPO Agent for Catan.

Wraps the neural network with action selection and experience storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from catan.constants import GamePhase
from catan.game import Action, GameState, get_legal_actions
from .features import (
    action_to_index,
    encode_state,
    index_to_action_from_list,
    legal_action_mask,
    _ACTION_SPACE_SIZE,
    _STATE_FEATURE_SIZE,
)
from .network import CatanNetwork


@dataclass
class Experience:
    """A single step of experience."""
    state: np.ndarray
    action_idx: int
    log_prob: float
    value: float
    reward: float = 0.0
    done: bool = False
    action_mask: np.ndarray = field(default_factory=lambda: np.zeros(1))


class CatanAgent:
    """PPO-based Catan player."""

    def __init__(
        self,
        player_idx: int,
        network: CatanNetwork,
        device: torch.device = torch.device("cpu"),
        deterministic: bool = False,
    ):
        self.player_idx = player_idx
        self.network = network
        self.device = device
        self.deterministic = deterministic
        self.experiences: List[Experience] = []

    def choose_action(self, gs: GameState) -> Action:
        """Choose an action given the current game state."""
        actions = get_legal_actions(gs)
        if not actions:
            raise ValueError("No legal actions available")

        if len(actions) == 1:
            # Only one option — take it
            a = actions[0]
            state = encode_state(gs, self.player_idx)
            mask = legal_action_mask(actions)
            idx = action_to_index(a)
            state_t = torch.from_numpy(state).to(self.device)
            mask_t = torch.from_numpy(mask).to(self.device)
            _, _, val = self.network.get_action(state_t, mask_t, deterministic=True)
            self.experiences.append(Experience(
                state=state, action_idx=idx,
                log_prob=0.0, value=val, action_mask=mask,
            ))
            return a

        state = encode_state(gs, self.player_idx)
        mask = legal_action_mask(actions)

        state_t = torch.from_numpy(state).to(self.device)
        mask_t = torch.from_numpy(mask).to(self.device)

        action_idx, log_prob, value = self.network.get_action(
            state_t, mask_t, deterministic=self.deterministic
        )

        action = index_to_action_from_list(action_idx, actions)
        if action is None:
            # Fallback: pick random legal action
            action = actions[np.random.randint(len(actions))]
            action_idx = action_to_index(action)

        self.experiences.append(Experience(
            state=state, action_idx=action_idx,
            log_prob=log_prob, value=value, action_mask=mask,
        ))

        return action

    def finalize_episode(self, reward: float) -> None:
        """Assign terminal reward and mark last experience as done."""
        if self.experiences:
            self.experiences[-1].reward = reward
            self.experiences[-1].done = True

    def clear_experiences(self) -> None:
        self.experiences.clear()


class RandomAgent:
    """Baseline random player (no learning)."""

    def __init__(self, player_idx: int):
        self.player_idx = player_idx

    def choose_action(self, gs: GameState) -> Action:
        actions = get_legal_actions(gs)
        if not actions:
            raise ValueError("No legal actions")
        # Prefer building over ending turn
        build_actions = [a for a in actions if a.action_type not in
                         (ActionType.END_TURN, ActionType.ROLL_DICE)]
        if build_actions and np.random.random() > 0.3:
            return build_actions[np.random.randint(len(build_actions))]
        return actions[np.random.randint(len(actions))]

    def finalize_episode(self, reward: float) -> None:
        pass

    def clear_experiences(self) -> None:
        pass


# Import at bottom to avoid circular
from catan.constants import ActionType
