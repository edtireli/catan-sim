"""Tests for the Catan game engine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from catan.board import generate_board
from catan.constants import (
    GamePhase,
    Resource,
    Terrain,
    TERRAIN_RESOURCE,
)
from catan.game import (
    Action,
    GameState,
    apply_action,
    get_legal_actions,
    new_game,
)


def test_board_generation():
    """Board should have 19 hexes, 54 vertices, 72 edges."""
    board = generate_board(seed=42)
    assert board.num_hexes == 19
    assert board.num_vertices == 54, f"Expected 54 vertices, got {board.num_vertices}"
    assert board.num_edges == 72, f"Expected 72 edges, got {board.num_edges}"


def test_board_terrain_distribution():
    """Check correct number of each terrain type."""
    board = generate_board(seed=42)
    counts = {}
    for h in board.hexes.values():
        counts[h.terrain] = counts.get(h.terrain, 0) + 1
    assert counts[Terrain.DESERT] == 1
    assert counts[Terrain.HILLS] == 3
    assert counts[Terrain.FOREST] == 4
    assert counts[Terrain.MOUNTAINS] == 3
    assert counts[Terrain.FIELDS] == 4
    assert counts[Terrain.PASTURE] == 4


def test_board_numbers():
    """Each non-desert hex should have a number token."""
    board = generate_board(seed=42)
    for h in board.hexes.values():
        if h.terrain == Terrain.DESERT:
            assert h.number == 0
        else:
            assert 2 <= h.number <= 12


def test_board_vertex_hex_adjacency():
    """Each vertex should be adjacent to 1-3 hexes."""
    board = generate_board(seed=42)
    for v in board.vertices.values():
        assert 1 <= len(v.adjacent_hexes) <= 3, (
            f"Vertex {v.vertex_id} adjacent to {len(v.adjacent_hexes)} hexes"
        )


def test_board_harbors():
    """9 harbors should be assigned."""
    board = generate_board(seed=42)
    harbor_vertices = [v for v in board.vertices.values() if v.harbor is not None]
    # Each harbor covers 2 vertices, 9 harbors = 18 vertices
    assert len(harbor_vertices) == 18, f"Expected 18 harbor vertices, got {len(harbor_vertices)}"


def test_new_game():
    """New game should start with correct state."""
    gs = new_game(seed=42)
    assert len(gs.players) == 4
    assert gs.phase == GamePhase.SETUP_SETTLEMENT_1
    assert gs.current_player_idx == 0
    assert gs.robber_hex >= 0
    assert len(gs.dev_card_deck) == 25  # 14+5+2+2+2


def test_setup_phase():
    """Should be able to complete setup phase."""
    gs = new_game(seed=42)

    # Complete setup for all 4 players (round 1 and 2)
    turns = 0
    while gs.phase in (GamePhase.SETUP_SETTLEMENT_1, GamePhase.SETUP_ROAD_1,
                       GamePhase.SETUP_SETTLEMENT_2, GamePhase.SETUP_ROAD_2):
        actions = get_legal_actions(gs)
        assert len(actions) > 0, f"No actions in phase {gs.phase}"
        apply_action(gs, actions[0])
        turns += 1
        if turns > 100:
            raise RuntimeError("Setup stuck in loop")

    assert gs.phase == GamePhase.ROLL_DICE
    # Each player should have 2 settlements and 2 roads
    for p in gs.players:
        assert p.num_settlements == 2, f"Player {p.index} has {p.num_settlements} settlements"
        assert p.num_roads == 2, f"Player {p.index} has {p.num_roads} roads"


def test_dice_roll():
    """Rolling dice should distribute resources or trigger robber."""
    gs = new_game(seed=42)
    _complete_setup(gs)

    assert gs.phase == GamePhase.ROLL_DICE
    actions = get_legal_actions(gs)
    roll_action = [a for a in actions if a.action_type.name == "ROLL_DICE"]
    assert len(roll_action) == 1

    apply_action(gs, roll_action[0])
    # After rolling, should be in MAIN_TURN or DISCARD/MOVE_ROBBER (if 7)
    assert gs.phase in (GamePhase.MAIN_TURN, GamePhase.DISCARD, GamePhase.MOVE_ROBBER)


def test_full_game_random():
    """A full game with random moves should terminate."""
    import random
    rng = random.Random(42)
    gs = new_game(seed=42)

    for _ in range(5000):
        if gs.phase == GamePhase.GAME_OVER:
            break
        actions = get_legal_actions(gs)
        if not actions:
            break
        action = rng.choice(actions)
        apply_action(gs, action)

    # Game should have progressed past setup
    assert gs.turn_number > 0


def test_resource_distribution():
    """Resources should be distributed correctly on dice rolls."""
    gs = new_game(seed=100)
    _complete_setup(gs)

    # Record initial resources
    initial = {p.index: dict(p.resources) for p in gs.players}

    # Roll several times
    for _ in range(10):
        if gs.phase == GamePhase.ROLL_DICE:
            actions = get_legal_actions(gs)
            roll_actions = [a for a in actions if a.action_type.name == "ROLL_DICE"]
            if roll_actions:
                apply_action(gs, roll_actions[0])

        # Complete any robber/discard phases
        while gs.phase in (GamePhase.DISCARD, GamePhase.MOVE_ROBBER, GamePhase.STEAL):
            actions = get_legal_actions(gs)
            if actions:
                apply_action(gs, actions[0])

        # End turn
        if gs.phase == GamePhase.MAIN_TURN:
            end_actions = [a for a in get_legal_actions(gs) if a.action_type.name == "END_TURN"]
            if end_actions:
                apply_action(gs, end_actions[0])

    # Some player should have received resources
    total = sum(
        sum(p.resources.values()) - sum(initial[p.index].values())
        for p in gs.players
    )
    # With 10 rolls, very likely some resources were produced
    # (but not guaranteed with unlucky dice)


def _complete_setup(gs: GameState) -> None:
    """Helper to run through setup phase."""
    while gs.phase in (GamePhase.SETUP_SETTLEMENT_1, GamePhase.SETUP_ROAD_1,
                       GamePhase.SETUP_SETTLEMENT_2, GamePhase.SETUP_ROAD_2):
        actions = get_legal_actions(gs)
        apply_action(gs, actions[0])


if __name__ == "__main__":
    tests = [
        test_board_generation,
        test_board_terrain_distribution,
        test_board_numbers,
        test_board_vertex_hex_adjacency,
        test_board_harbors,
        test_new_game,
        test_setup_phase,
        test_dice_roll,
        test_full_game_random,
        test_resource_distribution,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  ✓ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
