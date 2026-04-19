"""Tests for AI feature encoding and action space."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from catan.constants import GamePhase
from catan.game import get_legal_actions, new_game, apply_action
from ai.features import (
    action_to_index,
    encode_state,
    legal_action_mask,
    index_to_action_from_list,
    _ACTION_SPACE_SIZE,
    _STATE_FEATURE_SIZE,
)


def test_state_encoding_shape():
    """State encoding should produce correct shape."""
    gs = new_game(seed=42)
    state = encode_state(gs, 0)
    assert state.shape == (_STATE_FEATURE_SIZE,), f"Expected {_STATE_FEATURE_SIZE}, got {state.shape}"
    assert state.dtype == np.float32


def test_state_encoding_values():
    """State values should be in reasonable range."""
    gs = new_game(seed=42)
    state = encode_state(gs, 0)
    assert np.all(np.isfinite(state)), "State has non-finite values"
    assert np.max(np.abs(state)) < 100, f"State has extreme values: max={np.max(np.abs(state))}"


def test_action_mask():
    """Action mask should have correct size and at least one legal action."""
    gs = new_game(seed=42)
    actions = get_legal_actions(gs)
    mask = legal_action_mask(actions)
    assert mask.shape == (_ACTION_SPACE_SIZE,)
    assert mask.sum() == len(actions), f"Mask sum {mask.sum()} != {len(actions)} actions"
    assert mask.sum() > 0, "No legal actions"


def test_action_roundtrip():
    """Converting action to index and back should work."""
    gs = new_game(seed=42)
    actions = get_legal_actions(gs)
    for a in actions:
        idx = action_to_index(a)
        assert 0 <= idx < _ACTION_SPACE_SIZE, f"Action index {idx} out of bounds"
        recovered = index_to_action_from_list(idx, actions)
        assert recovered is not None, f"Could not recover action from index {idx}"


def test_action_indices_unique():
    """Each legal action should map to a unique index."""
    gs = new_game(seed=42)
    # Run through setup to get more diverse actions
    _complete_setup(gs)
    actions = get_legal_actions(gs)
    indices = [action_to_index(a) for a in actions]
    assert len(set(indices)) == len(indices), "Duplicate action indices found"


def test_state_different_perspectives():
    """State encoding should differ for different player perspectives."""
    gs = new_game(seed=42)
    _complete_setup(gs)
    state0 = encode_state(gs, 0)
    state1 = encode_state(gs, 1)
    # States should differ (different player perspective)
    assert not np.allclose(state0, state1), "States for different players should differ"


def _complete_setup(gs):
    while gs.phase in (GamePhase.SETUP_SETTLEMENT_1, GamePhase.SETUP_ROAD_1,
                       GamePhase.SETUP_SETTLEMENT_2, GamePhase.SETUP_ROAD_2):
        actions = get_legal_actions(gs)
        apply_action(gs, actions[0])


if __name__ == "__main__":
    tests = [
        test_state_encoding_shape,
        test_state_encoding_values,
        test_action_mask,
        test_action_roundtrip,
        test_action_indices_unique,
        test_state_different_perspectives,
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
