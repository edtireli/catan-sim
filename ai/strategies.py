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
    games_completed: int = 0  # games that reached 10 VP

    # Average across all games (best-performing agent per game)
    avg_vp: float = 0.0
    avg_max_vp: float = 0.0  # highest VP reached in any game
    avg_turns: float = 0.0
    avg_vp_at_win: float = 0.0
    avg_turns_to_win: float = 0.0

    # Building patterns (averaged across all games, best player)
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

    total_turns = 0
    total_vp = 0
    max_vps: List[int] = []
    total_turns_at_win = 0
    total_vp_at_win = 0
    total_settlements = 0
    total_cities = 0
    total_roads = 0
    total_knights = 0
    total_dev_bought = 0
    longest_road_count = 0
    largest_army_count = 0

    MAX_TURNS = 1000  # More generous for analysis

    for game_idx in range(num_games):
        gs = new_game(seed=game_idx + 10000)
        agents = [
            CatanAgent(i, network, device, deterministic=False)
            for i in range(NUM_PLAYERS)
        ]

        dev_bought = 0

        turn = 0
        while gs.phase != GamePhase.GAME_OVER and turn < MAX_TURNS:
            pid = gs.current_player_idx
            legal = get_legal_actions(gs)
            if not legal:
                break

            action = agents[pid].choose_action(gs)

            # Track actions of all agents (they share the same network)
            atype = action.action_type.name
            action_counts[atype] = action_counts.get(atype, 0) + 1

            if action.action_type == ActionType.BUY_DEV_CARD:
                dev_bought += 1

            apply_action(gs, action)
            turn += 1

        profile.games_played += 1
        total_turns += turn

        # Find best player this game (winner, or highest VP)
        if gs.winner is not None:
            best = gs.players[gs.winner]
            profile.games_completed += 1
            profile.wins += 1
            total_turns_at_win += turn
            total_vp_at_win += best.victory_points
        else:
            best = max(gs.players, key=lambda p: p.victory_points)

        max_vps.append(best.victory_points)
        total_vp += best.victory_points
        total_settlements += best.num_settlements
        total_cities += best.num_cities
        total_roads += best.num_roads
        total_knights += best.knights_played
        total_dev_bought += dev_bought  # across all players
        if best.has_longest_road:
            longest_road_count += 1
        if best.has_largest_army:
            largest_army_count += 1

        # Track settlement locations for all players' buildings
        for player in gs.players:
            for vid in player.settlement_vertices | player.city_vertices:
                v = gs.board.vertices[vid]
                for hid in v.adjacent_hexes:
                    h = gs.board.hexes[hid]
                    tname = h.terrain.name.lower()
                    terrain_counts[tname] = terrain_counts.get(tname, 0) + 1
                    if h.number > 0:
                        number_counts[h.number] = number_counts.get(h.number, 0) + 1

    # Compute averages
    n = profile.games_played or 1
    profile.win_rate = profile.wins / n
    profile.avg_vp = total_vp / n
    profile.avg_max_vp = max(max_vps) if max_vps else 0
    profile.avg_turns = total_turns / n
    profile.avg_settlements = total_settlements / n
    profile.avg_cities = total_cities / n
    profile.avg_roads = total_roads / n
    profile.avg_knights_played = total_knights / n
    profile.avg_dev_cards_bought = total_dev_bought / n
    profile.pct_longest_road = longest_road_count / n
    profile.pct_largest_army = largest_army_count / n

    if profile.wins > 0:
        profile.avg_turns_to_win = total_turns_at_win / profile.wins
        profile.avg_vp_at_win = total_vp_at_win / profile.wins

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
    print(f"Games completed (10 VP): {profile.games_completed}")
    print(f"Completion rate: {profile.win_rate:.1%}")
    print()

    print("--- Performance ---")
    print(f"Avg VP (best player): {profile.avg_vp:.1f}")
    print(f"Max VP reached: {profile.avg_max_vp:.0f}")
    print(f"Avg game length: {profile.avg_turns:.0f} turns")
    if profile.wins > 0:
        print(f"Avg VP at win: {profile.avg_vp_at_win:.1f}")
        print(f"Avg turns to win: {profile.avg_turns_to_win:.0f}")
    print()

    print("--- Building Pattern (best player avg) ---")
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


def compare_checkpoints(checkpoint_paths: List[str], num_games: int = 100) -> None:
    """Compare strategy profiles across multiple checkpoints to show learning progression."""
    profiles = []
    for path in checkpoint_paths:
        label = Path(path).stem.replace("catan_agent_", "")
        print(f"Evaluating {label}...")
        p = analyze_strategy(path, num_games=num_games)
        profiles.append((label, p))

    print()
    print("=" * 70)
    print("TRAINING PROGRESSION")
    print("=" * 70)

    # Header
    labels = [lbl for lbl, _ in profiles]
    header = f"{'Metric':<28s}" + "".join(f"{l:>12s}" for l in labels)
    print(header)
    print("-" * len(header))

    # Rows
    def row(name: str, get_val, fmt: str = ".1f"):
        vals = [get_val(p) for _, p in profiles]
        line = f"{name:<28s}" + "".join(f"{v:>12{fmt}}" for v in vals)
        print(line)

    row("Win rate (%)", lambda p: p.win_rate * 100)
    row("Avg turns to win", lambda p: p.avg_turns_to_win, ".0f")
    row("Avg VP at win", lambda p: p.avg_vp_at_win)
    row("Avg settlements", lambda p: p.avg_settlements)
    row("Avg cities", lambda p: p.avg_cities)
    row("Avg roads", lambda p: p.avg_roads)
    row("Avg dev cards bought", lambda p: p.avg_dev_cards_bought)
    row("Avg knights played", lambda p: p.avg_knights_played)
    row("Longest road wins (%)", lambda p: p.pct_longest_road * 100)
    row("Largest army wins (%)", lambda p: p.pct_largest_army * 100)

    # Key strategic shifts
    print()
    print("--- Key Insights ---")
    first = profiles[0][1]
    last = profiles[-1][1]
    if last.win_rate > first.win_rate:
        print(f"  ✓ Win rate improved: {first.win_rate:.1%} → {last.win_rate:.1%}")
    if last.avg_turns_to_win < first.avg_turns_to_win and last.avg_turns_to_win > 0:
        print(f"  ✓ Faster wins: {first.avg_turns_to_win:.0f} → {last.avg_turns_to_win:.0f} turns")
    if last.avg_cities > first.avg_cities:
        print(f"  ✓ More city upgrades: {first.avg_cities:.1f} → {last.avg_cities:.1f}")
    if last.pct_largest_army > first.pct_largest_army:
        print(f"  ✓ More army-focused: {first.pct_largest_army:.1%} → {last.pct_largest_army:.1%}")
    if last.pct_longest_road > first.pct_longest_road:
        print(f"  ✓ More road-focused: {first.pct_longest_road:.1%} → {last.pct_longest_road:.1%}")

    # Dominant strategy
    if last.win_rate > 0:
        if last.pct_largest_army > 0.4:
            strat = "Army Builder (knight-heavy)"
        elif last.pct_longest_road > 0.4:
            strat = "Road Runner (road-heavy)"
        elif last.avg_cities > 2.5:
            strat = "City Developer (ore/grain focus)"
        elif last.avg_settlements > 4:
            strat = "Wide Settler (spread out)"
        else:
            strat = "Balanced"
        print(f"\n  Dominant strategy at latest checkpoint: {strat}")

    print("=" * 70)
