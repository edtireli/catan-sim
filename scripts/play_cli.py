#!/usr/bin/env python3
"""Play a quick CLI game against AI for testing.

Usage:
    python scripts/play_cli.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from catan.constants import GamePhase, Resource, PLAYER_COLORS
from catan.game import GameState, apply_action, get_legal_actions, new_game
from ai.agent import RandomAgent


def main():
    gs = new_game(seed=42)
    agents = [RandomAgent(i) for i in range(4)]

    print("=== Catan CLI — 4 Random Bots ===\n")

    turn = 0
    while gs.phase != GamePhase.GAME_OVER and turn < 2000:
        pid = gs.current_player_idx
        actions = get_legal_actions(gs)
        if not actions:
            print(f"No actions available for player {pid}!")
            break

        agent = agents[pid]
        action = agent.choose_action(gs)
        apply_action(gs, action)

        if gs.phase == GamePhase.MAIN_TURN or gs.phase == GamePhase.ROLL_DICE:
            turn += 1
            if turn % 50 == 0:
                print(f"Turn {turn}:")
                for p in gs.players:
                    res = dict(p.resources)
                    print(f"  P{p.index} ({p.color}): VP={p.victory_points} "
                          f"S={p.num_settlements} C={p.num_cities} R={p.num_roads} "
                          f"resources={sum(res.values())}")

    if gs.winner is not None:
        w = gs.players[gs.winner]
        print(f"\n🏆 Player {w.index} ({w.color}) wins with {w.victory_points} VP!")
        print(f"   Settlements: {w.num_settlements}, Cities: {w.num_cities}, Roads: {w.num_roads}")
        print(f"   Knights: {w.knights_played}, Longest road: {w.has_longest_road}, Largest army: {w.has_largest_army}")
    else:
        print("\nGame timed out (no winner)")

    print(f"\nTotal turns: {turn}")


if __name__ == "__main__":
    main()
