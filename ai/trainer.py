"""PPO trainer with self-play for Catan AI.

Runs episodes of 4-player Catan games, collects experience from all
agents, and updates the shared policy network using PPO.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from catan.constants import GamePhase, NUM_PLAYERS
from catan.game import GameState, apply_action, get_legal_actions, new_game
from catan.replay import GameRecorder, save_replay
from .agent import CatanAgent, Experience
from .features import _ACTION_SPACE_SIZE, _STATE_FEATURE_SIZE
from .network import CatanNetwork

MAX_TURNS_PER_GAME = 500  # Safety cutoff


@dataclass
class TrainingConfig:
    """Hyperparameters for PPO training."""
    # Training loop
    num_epochs: int = 200
    games_per_epoch: int = 50
    # PPO
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    mini_batch_size: int = 256
    # Network
    hidden_size: int = 512
    num_layers: int = 3
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 20
    # Logging
    log_dir: str = "runs"


@dataclass
class EpochStats:
    """Stats collected during one epoch of training."""
    epoch: int = 0
    games_played: int = 0
    wins: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(NUM_PLAYERS)})
    avg_game_length: float = 0.0
    avg_reward: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    total_steps: int = 0
    time_elapsed: float = 0.0

    # Strategy tracking
    avg_settlements: float = 0.0
    avg_cities: float = 0.0
    avg_roads: float = 0.0
    avg_dev_cards: float = 0.0
    avg_knights: float = 0.0
    longest_road_wins: int = 0
    largest_army_wins: int = 0


class Trainer:
    """PPO self-play trainer."""

    def __init__(self, config: TrainingConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create network
        self.network = CatanNetwork(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=config.lr)

        # Logging
        self.writer: Optional[SummaryWriter] = None
        self.epoch_history: List[EpochStats] = []

        # Optional spectator callback: called with (game_state, action, epoch, game_idx)
        self.on_action: Optional[Callable] = None

    def train(self) -> None:
        """Run the full training loop."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.config.log_dir)

        print(f"Training on {self.device}")
        print(f"State features: {_STATE_FEATURE_SIZE}, Action space: {_ACTION_SPACE_SIZE}")
        print(f"Network params: {sum(p.numel() for p in self.network.parameters()):,}")
        print()

        for epoch in range(1, self.config.num_epochs + 1):
            stats = self._run_epoch(epoch)
            self.epoch_history.append(stats)
            self._log_epoch(stats)

            if epoch % self.config.checkpoint_interval == 0:
                self._save_checkpoint(epoch)

        self.writer.close()
        self._save_checkpoint(self.config.num_epochs)
        print("\nTraining complete!")

    def _run_epoch(self, epoch: int) -> EpochStats:
        """Play games and update the network."""
        t0 = time.time()
        stats = EpochStats(epoch=epoch)
        all_experiences: List[Experience] = []
        game_lengths: List[int] = []
        rewards: List[float] = []

        # Strategy tracking accumulators
        total_settlements = 0
        total_cities = 0
        total_roads = 0
        total_dev_cards = 0
        total_knights = 0
        total_winners = 0

        for game_idx in range(self.config.games_per_epoch):
            seed = epoch * 10000 + game_idx
            gs = new_game(seed=seed)

            # Create 4 agents sharing the same network
            agents = [
                CatanAgent(i, self.network, self.device)
                for i in range(NUM_PLAYERS)
            ]

            recorder = GameRecorder(seed=seed, epoch=epoch, game_index=game_idx)

            turn_count = 0
            while gs.phase != GamePhase.GAME_OVER and turn_count < MAX_TURNS_PER_GAME:
                pid = gs.current_player_idx
                agent = agents[pid]

                legal = get_legal_actions(gs)
                if not legal:
                    break

                action = agent.choose_action(gs)
                apply_action(gs, action)
                recorder.record(gs, action, pid)
                turn_count += 1

                # Notify spectators
                if self.on_action is not None:
                    self.on_action(gs, action, epoch, game_idx)

            # Save replay for won games
            replay = recorder.finalize(gs)
            if gs.winner is not None:
                save_replay(replay)

            # Assign rewards
            for agent in agents:
                if gs.winner is not None:
                    reward = 1.0 if agent.player_idx == gs.winner else -0.33
                else:
                    reward = 0.0  # Draw / timeout
                agent.finalize_episode(reward)
                all_experiences.extend(agent.experiences)
                rewards.append(reward)

            # Track stats
            if gs.winner is not None:
                stats.wins[gs.winner] += 1
                winner = gs.players[gs.winner]
                total_settlements += winner.num_settlements
                total_cities += winner.num_cities
                total_roads += winner.num_roads
                total_dev_cards += len(winner.dev_cards) + len(winner.new_dev_cards)
                total_knights += winner.knights_played
                if winner.has_longest_road:
                    stats.longest_road_wins += 1
                if winner.has_largest_army:
                    stats.largest_army_wins += 1
                total_winners += 1

            game_lengths.append(turn_count)
            stats.games_played += 1

        # Update network
        if all_experiences:
            p_loss, v_loss, ent = self._ppo_update(all_experiences)
            stats.policy_loss = p_loss
            stats.value_loss = v_loss
            stats.entropy = ent

        stats.avg_game_length = np.mean(game_lengths).item() if game_lengths else 0
        stats.avg_reward = np.mean(rewards).item() if rewards else 0
        stats.total_steps = len(all_experiences)
        stats.time_elapsed = time.time() - t0

        if total_winners > 0:
            stats.avg_settlements = total_settlements / total_winners
            stats.avg_cities = total_cities / total_winners
            stats.avg_roads = total_roads / total_winners
            stats.avg_dev_cards = total_dev_cards / total_winners
            stats.avg_knights = total_knights / total_winners

        return stats

    def _ppo_update(self, experiences: List[Experience]) -> Tuple[float, float, float]:
        """Run PPO update on collected experiences."""
        # Prepare data
        states = torch.from_numpy(np.array([e.state for e in experiences])).to(self.device)
        actions = torch.tensor([e.action_idx for e in experiences], dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor([e.log_prob for e in experiences], dtype=torch.float32).to(self.device)
        masks = torch.from_numpy(np.array([e.action_mask for e in experiences])).to(self.device)

        # Compute returns and advantages using GAE
        returns, advantages = self._compute_gae(experiences)
        returns = torch.from_numpy(returns).to(self.device)
        advantages = torch.from_numpy(advantages).to(self.device)

        # PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        dataset_size = len(experiences)
        for _ in range(self.config.ppo_epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.config.mini_batch_size):
                end = min(start + self.config.mini_batch_size, dataset_size)
                batch_idx = indices[start:end]
                batch_idx_t = torch.from_numpy(batch_idx).to(self.device)

                b_states = states[batch_idx_t]
                b_actions = actions[batch_idx_t]
                b_old_lp = old_log_probs[batch_idx_t]
                b_masks = masks[batch_idx_t]
                b_returns = returns[batch_idx_t]
                b_advantages = advantages[batch_idx_t]

                # Forward pass
                log_probs, values = self.network(b_states, b_masks)
                new_log_probs = log_probs.gather(1, b_actions.unsqueeze(1)).squeeze(1)

                # Policy loss (clipped)
                ratio = (new_log_probs - b_old_lp).exp()
                surr1 = ratio * b_advantages
                surr2 = ratio.clamp(1 - self.config.clip_epsilon,
                                    1 + self.config.clip_epsilon) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values.squeeze(1), b_returns)

                # Entropy bonus
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1).mean()

                # Total loss
                loss = (policy_loss
                        + self.config.value_loss_coef * value_loss
                        - self.config.entropy_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        return (
            total_policy_loss / max(n_updates, 1),
            total_value_loss / max(n_updates, 1),
            total_entropy / max(n_updates, 1),
        )

    def _compute_gae(self, experiences: List[Experience]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        n = len(experiences)
        returns = np.zeros(n, dtype=np.float32)
        advantages = np.zeros(n, dtype=np.float32)

        last_gae = 0.0
        last_val = 0.0

        for t in reversed(range(n)):
            exp = experiences[t]
            if exp.done:
                last_gae = 0.0
                last_val = 0.0
                next_val = 0.0
            else:
                next_val = last_val

            delta = exp.reward + self.config.gamma * next_val - exp.value
            last_gae = delta + self.config.gamma * self.config.gae_lambda * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + exp.value
            last_val = exp.value

        # Normalize advantages
        if n > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def _log_epoch(self, stats: EpochStats) -> None:
        """Log epoch stats to console and TensorBoard."""
        e = stats.epoch
        wins_str = " ".join(f"P{i}:{stats.wins[i]}" for i in range(NUM_PLAYERS))
        print(
            f"Epoch {e:4d} | Games: {stats.games_played:3d} | "
            f"Wins: [{wins_str}] | "
            f"Avg len: {stats.avg_game_length:6.1f} | "
            f"Steps: {stats.total_steps:6d} | "
            f"P_loss: {stats.policy_loss:.4f} | "
            f"V_loss: {stats.value_loss:.4f} | "
            f"Ent: {stats.entropy:.4f} | "
            f"Time: {stats.time_elapsed:.1f}s"
        )

        if self.writer:
            self.writer.add_scalar("training/policy_loss", stats.policy_loss, e)
            self.writer.add_scalar("training/value_loss", stats.value_loss, e)
            self.writer.add_scalar("training/entropy", stats.entropy, e)
            self.writer.add_scalar("training/avg_reward", stats.avg_reward, e)
            self.writer.add_scalar("game/avg_length", stats.avg_game_length, e)
            self.writer.add_scalar("game/total_steps", stats.total_steps, e)
            # Win distribution
            for i in range(NUM_PLAYERS):
                self.writer.add_scalar(f"wins/player_{i}", stats.wins[i], e)
            # Strategy metrics
            self.writer.add_scalar("strategy/avg_settlements", stats.avg_settlements, e)
            self.writer.add_scalar("strategy/avg_cities", stats.avg_cities, e)
            self.writer.add_scalar("strategy/avg_roads", stats.avg_roads, e)
            self.writer.add_scalar("strategy/avg_dev_cards", stats.avg_dev_cards, e)
            self.writer.add_scalar("strategy/avg_knights", stats.avg_knights, e)
            self.writer.add_scalar("strategy/longest_road_wins", stats.longest_road_wins, e)
            self.writer.add_scalar("strategy/largest_army_wins", stats.largest_army_wins, e)
            self.writer.flush()

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint."""
        path = Path(self.config.checkpoint_dir) / f"catan_agent_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path)
        print(f"  → Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> int:
        """Load a checkpoint. Returns the epoch number."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]
