"""Strategy analysis for trained AI agents.

Analyzes what strategies the AI has learned by running evaluation games
and examining building patterns, resource usage, and decision tendencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from catan.constants import (
    ActionType, DevCard, GamePhase, NUM_PLAYERS, Resource,
)
from catan.game import GameState, apply_action, get_legal_actions, new_game
from .agent import CatanAgent
from .network import CatanNetwork


@dataclass
class StrategyProfile:
    """Aggregated strategy metrics from evaluation games."""
    games_played: int = 0
    wins: int = 0
    win_rate: float = 0.0

    # Average per winning game
    avg_vp_at_win: float = 0.0
    avg_turns_to_win: float = 0.0

    # Building patterns
    avg_settlements: float = 0.0
    avg_cities: float = 0.0
    avg_roads: float = 0.0

    # Dev card usage
    avg_knights_played: float = 0.0
    avg_dev_cards_bought: float = 0.0
    pct_longest_road: float = 0.0
    pct_largest_army: float = 0.0

    # Resource efficiency
    avg_resources_per_turn: float = 0.0

    # Action distribution
    action_distribution: Dict[str, float] = field(default_factory=dict)

    # Hex preference (which numbers/resources does it prioritize)
    settlement_terrain_pref: Dict[str, float] = field(default_factory=dict)
    settlement_number_pref: Dict[int, float] = field(default_factory=dict)


def analyze_strategy(
    checkpoint_path: str,
    num_games: int = 200,
    device: Optional[torch.device] = None,
) -> StrategyProfile:
    """Run evaluation games and build a strategy profile."""
    if device is None:
        device = torch.device("cpu")

    network = CatanNetwork()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()

    profile = StrategyProfile()
    action_counts: Dict[str, int] = {}
    terrain_counts: Dict[str, int] = {}
    number_counts: Dict[int, int] = {}

    total_turns_at_win = 0
    total_vp_at_win = 0
    total_settlements = 0
    total_cities = 0
    total_roads = 0
    total_knights = 0
    total_dev_bought = 0
    longest_road_count = 0
    largest_army_count = 0

    for game_idx in range(num_games):
        gs = new_game(seed=game_idx + 10000)
        agents = [
            CatanAgent(i, network, device, deterministic=True)
            for i in range(NUM_PLAYERS)
        ]

        # Track per-game actions for player 0
        game_actions: Dict[str, int] = {}
        dev_bought = 0

        turn = 0
        while gs.phase != GamePhase.GAME_OVER and turn < 500:
            pid = gs.current_player_idx
            legal = get_legal_actions(gs)
            if not legal:
                break

            action = agents[pid].choose_action(gs)

            # Track actions of all agents (they share the same network)
            atype = action.action_type.name
            action_counts[atype] = action_counts.get(atype, 0) + 1

            if pid == 0:
                game_actions[atype] = game_actions.get(atype, 0) + 1
                if action.action_type == ActionType.BUY_DEV_CARD:
                    dev_bought += 1

            apply_action(gs, action)
            turn += 1

        profile.games_played += 1

        if gs.winner is not None:
            winner = gs.players[gs.winner]
            if gs.winner == 0:  # Track player 0 stats
                profile.wins += 1
                total_turns_at_win += turn
                total_vp_at_win += winner.victory_points
                total_settlements += winner.num_settlements
                total_cities += winner.num_cities
                total_roads += winner.num_roads
                total_knights += winner.knights_played
                total_dev_bought += dev_bought
                if winner.has_longest_road:
                    longest_road_count += 1
                if winner.has_largest_army:
                    largest_army_count += 1

                # Track settlement locations
                for vid in winner.settlement_vertices | winner.city_vertices:
                    v = gs.board.vertices[vid]
                    for hid in v.adjacent_hexes:
                        h = gs.board.hexes[hid]
                        tname = h.terrain.name.lower()
                        terrain_counts[tname] = terrain_counts.get(tname, 0) + 1
                        if h.number > 0:
                            number_counts[h.number] = number_counts.get(h.number, 0) + 1

    # Compute averages
    if profile.wins > 0:
        profile.win_rate = profile.wins / profile.games_played
        profile.avg_turns_to_win = total_turns_at_win / profile.wins
        profile.avg_vp_at_win = total_vp_at_win / profile.wins
        profile.avg_settlements = total_settlements / profile.wins
        profile.avg_cities = total_cities / profile.wins
        profile.avg_roads = total_roads / profile.wins
        profile.avg_knights_played = total_knights / profile.wins
        profile.avg_dev_cards_bought = total_dev_bought / profile.wins
        profile.pct_longest_road = longest_road_count / profile.wins
        profile.pct_largest_army = largest_army_count / profile.wins

    total_actions = sum(action_counts.values()) or 1
    profile.action_distribution = {
        k: v / total_actions for k, v in sorted(action_counts.items())
    }

    total_terrain = sum(terrain_counts.values()) or 1
    profile.settlement_terrain_pref = {
        k: v / total_terrain for k, v in sorted(terrain_counts.items())
    }

    total_numbers = sum(number_counts.values()) or 1
    profile.settlement_number_pref = {
        k: v / total_numbers for k, v in sorted(number_counts.items())
    }

    return profile


def print_strategy_report(profile: StrategyProfile) -> None:
    """Print a human-readable strategy report."""
    print("=" * 60)
    print("STRATEGY ANALYSIS REPORT")
    print("=" * 60)
    print(f"Games played: {profile.games_played}")
    print(f"Win rate: {profile.win_rate:.1%}")
    print()

    print("--- Victory Pattern ---")
    print(f"Avg VP at win: {profile.avg_vp_at_win:.1f}")
    print(f"Avg turns to win: {profile.avg_turns_to_win:.0f}")
    print()

    print("--- Building Pattern ---")
    print(f"Avg settlements: {profile.avg_settlements:.1f}")
    print(f"Avg cities: {profile.avg_cities:.1f}")
    print(f"Avg roads: {profile.avg_roads:.1f}")
    print()

    print("--- Development Cards ---")
    print(f"Avg dev cards bought: {profile.avg_dev_cards_bought:.1f}")
    print(f"Avg knights played: {profile.avg_knights_played:.1f}")
    print(f"Longest road %: {profile.pct_longest_road:.1%}")
    print(f"Largest army %: {profile.pct_largest_army:.1%}")
    print()

    print("--- Settlement Location Preferences ---")
    print("Terrain preference:")
    for terrain, pct in sorted(profile.settlement_terrain_pref.items(),
                                key=lambda x: -x[1]):
        print(f"  {terrain:12s}: {'█' * int(pct * 50)} {pct:.1%}")
    print("Number preference:")
    for num, pct in sorted(profile.settlement_number_pref.items(),
                           key=lambda x: -x[1]):
        print(f"  {num:2d}: {'█' * int(pct * 50)} {pct:.1%}")
    print()

    print("--- Action Distribution ---")
    for action, pct in sorted(profile.action_distribution.items(),
                              key=lambda x: -x[1])[:10]:
        print(f"  {action:25s}: {pct:.2%}")
    print("=" * 60)
