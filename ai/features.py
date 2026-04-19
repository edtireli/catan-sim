"""Encode game state into feature vectors for the neural network.

The feature vector is a fixed-size 1-D numpy array capturing all
information an AI player can observe (no hidden information from other
players' dev cards, but own dev cards are visible).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from catan.board import Board
from catan.constants import (
    MAX_CITIES,
    MAX_ROADS,
    MAX_SETTLEMENTS,
    NUM_RESOURCE_TYPES,
    DevCard,
    GamePhase,
    HarborType,
    Resource,
    Terrain,
)
from catan.game import Action, ActionType, GameState
from catan.player import Player

# -----------------------------------------------------------------------
# Action indexing – map every possible action to a flat integer
# -----------------------------------------------------------------------

# Maximum counts (used for sizing the flat action space)
MAX_VERTICES = 54
MAX_EDGES = 72
MAX_HEXES = 19
MAX_PLAYERS = 4
NUM_RESOURCES = 5


def action_space_size() -> int:
    """Total number of discrete action slots."""
    n = 0
    n += 1                            # ROLL_DICE
    n += 1                            # END_TURN
    n += MAX_EDGES                    # BUILD_ROAD (one per edge)
    n += MAX_VERTICES                 # BUILD_SETTLEMENT
    n += MAX_VERTICES                 # BUILD_CITY
    n += 1                            # BUY_DEV_CARD
    n += MAX_HEXES                    # PLAY_KNIGHT (robber destination)
    n += 1                            # PLAY_ROAD_BUILDING
    n += NUM_RESOURCES * NUM_RESOURCES  # PLAY_YEAR_OF_PLENTY (r1, r2)
    n += NUM_RESOURCES                # PLAY_MONOPOLY
    n += NUM_RESOURCES * NUM_RESOURCES  # TRADE_BANK (give, get)
    n += MAX_HEXES                    # PLACE_ROBBER
    n += MAX_PLAYERS + 1              # STEAL_FROM (including None)
    n += 1                            # DISCARD_RESOURCES (simplified: choose best)
    n += MAX_VERTICES                 # PLACE_SETUP_SETTLEMENT
    n += MAX_EDGES                    # PLACE_SETUP_ROAD
    n += MAX_EDGES                    # PLACE_ROAD_BUILDING_ROAD
    return n


_ACTION_SPACE_SIZE = action_space_size()

# Offsets into the flat action vector
_OFFSETS: Dict[ActionType, int] = {}
_off = 0
_OFFSETS[ActionType.ROLL_DICE] = _off; _off += 1
_OFFSETS[ActionType.END_TURN] = _off; _off += 1
_OFFSETS[ActionType.BUILD_ROAD] = _off; _off += MAX_EDGES
_OFFSETS[ActionType.BUILD_SETTLEMENT] = _off; _off += MAX_VERTICES
_OFFSETS[ActionType.BUILD_CITY] = _off; _off += MAX_VERTICES
_OFFSETS[ActionType.BUY_DEV_CARD] = _off; _off += 1
_OFFSETS[ActionType.PLAY_KNIGHT] = _off; _off += MAX_HEXES
_OFFSETS[ActionType.PLAY_ROAD_BUILDING] = _off; _off += 1
_OFFSETS[ActionType.PLAY_YEAR_OF_PLENTY] = _off; _off += NUM_RESOURCES * NUM_RESOURCES
_OFFSETS[ActionType.PLAY_MONOPOLY] = _off; _off += NUM_RESOURCES
_OFFSETS[ActionType.TRADE_BANK] = _off; _off += NUM_RESOURCES * NUM_RESOURCES
_OFFSETS[ActionType.PLACE_ROBBER] = _off; _off += MAX_HEXES
_OFFSETS[ActionType.STEAL_FROM] = _off; _off += MAX_PLAYERS + 1
_OFFSETS[ActionType.DISCARD_RESOURCES] = _off; _off += 1
_OFFSETS[ActionType.PLACE_SETUP_SETTLEMENT] = _off; _off += MAX_VERTICES
_OFFSETS[ActionType.PLACE_SETUP_ROAD] = _off; _off += MAX_EDGES
_OFFSETS[ActionType.PLACE_ROAD_BUILDING_ROAD] = _off; _off += MAX_EDGES


def action_to_index(action: Action) -> int:
    """Convert an Action to a flat index."""
    at = action.action_type
    base = _OFFSETS[at]

    if at in (ActionType.ROLL_DICE, ActionType.END_TURN,
              ActionType.BUY_DEV_CARD, ActionType.PLAY_ROAD_BUILDING,
              ActionType.DISCARD_RESOURCES):
        return base

    if at in (ActionType.BUILD_ROAD, ActionType.PLACE_SETUP_ROAD,
              ActionType.PLACE_ROAD_BUILDING_ROAD):
        return base + action.edge

    if at in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY,
              ActionType.PLACE_SETUP_SETTLEMENT):
        return base + action.vertex

    if at in (ActionType.PLAY_KNIGHT, ActionType.PLACE_ROBBER):
        return base + action.hex_id

    if at == ActionType.PLAY_YEAR_OF_PLENTY:
        return base + int(action.resource) * NUM_RESOURCES + int(action.resource2)

    if at == ActionType.PLAY_MONOPOLY:
        return base + int(action.resource)

    if at == ActionType.TRADE_BANK:
        return base + int(action.give_resource) * NUM_RESOURCES + int(action.get_resource)

    if at == ActionType.STEAL_FROM:
        if action.target_player is None:
            return base + MAX_PLAYERS
        return base + action.target_player

    return base


def legal_action_mask(actions: List[Action]) -> np.ndarray:
    """Create a binary mask of legal actions."""
    mask = np.zeros(_ACTION_SPACE_SIZE, dtype=np.float32)
    for a in actions:
        mask[action_to_index(a)] = 1.0
    return mask


def index_to_action_from_list(idx: int, actions: List[Action]) -> Optional[Action]:
    """Given a flat index and the list of legal actions, return matching action."""
    for a in actions:
        if action_to_index(a) == idx:
            return a
    return None


# -----------------------------------------------------------------------
# State encoding
# -----------------------------------------------------------------------

def state_feature_size() -> int:
    """Size of the state feature vector."""
    n = 0
    # Per-hex features (19 hexes)
    # terrain (6 one-hot) + number (1 normalized) + robber (1) = 8
    n += MAX_HEXES * 8
    # Per-vertex features (54 vertices)
    # building type (3: none/settlement/city) + owner (4 one-hot) + harbor (7 one-hot) = 14
    n += MAX_VERTICES * 14
    # Per-edge features (72 edges)
    # road owner (5: none + 4 players) = 5
    n += MAX_EDGES * 5
    # Current player features
    # resources (5) + dev cards (5 counts) + knights played + VP +
    # longest road flag + largest army flag + buildings remaining (3) = 20
    n += 20
    # Other players (3 opponents) – observable info only
    # resource count (1) + dev card count (1) + knights (1) + VP (1) +
    # longest road (1) + largest army (1) + settlements (1) + cities (1) + roads (1) = 9
    n += (MAX_PLAYERS - 1) * 9
    # Game state
    # phase (12 one-hot) + turn number (1) = 13
    n += 13
    return n


_STATE_FEATURE_SIZE = state_feature_size()


def encode_state(gs: GameState, player_idx: int) -> np.ndarray:
    """Encode game state from the perspective of player_idx."""
    feat = np.zeros(_STATE_FEATURE_SIZE, dtype=np.float32)
    board = gs.board
    player = gs.players[player_idx]
    off = 0

    # --- Hex features ---
    for hid in range(min(MAX_HEXES, board.num_hexes)):
        h = board.hexes[hid]
        # Terrain one-hot (6)
        feat[off + int(h.terrain)] = 1.0
        off += 6
        # Number (normalized to [0, 1])
        feat[off] = h.number / 12.0
        off += 1
        # Robber
        feat[off] = 1.0 if h.has_robber else 0.0
        off += 1
    # Pad remaining hex slots
    off = MAX_HEXES * 8

    # --- Vertex features ---
    for vid in range(min(MAX_VERTICES, board.num_vertices)):
        v = board.vertices[vid]
        # Building type (3)
        if v.building is None:
            feat[off] = 1.0
        elif v.building == "settlement":
            feat[off + 1] = 1.0
        elif v.building == "city":
            feat[off + 2] = 1.0
        off += 3
        # Owner (4 one-hot, relative to current player)
        if v.building_owner is not None:
            rel = (v.building_owner - player_idx) % len(gs.players)
            feat[off + rel] = 1.0
        off += 4
        # Harbor (7 one-hot: none + 6 types)
        if v.harbor is None:
            feat[off] = 1.0
        else:
            feat[off + 1 + int(v.harbor)] = 1.0
        off += 7
    off = MAX_HEXES * 8 + MAX_VERTICES * 14

    # --- Edge features ---
    for eid in range(min(MAX_EDGES, board.num_edges)):
        e = board.edges[eid]
        if e.road_owner is None:
            feat[off] = 1.0
        else:
            rel = (e.road_owner - player_idx) % len(gs.players)
            feat[off + 1 + rel] = 1.0
        off += 5
    off = MAX_HEXES * 8 + MAX_VERTICES * 14 + MAX_EDGES * 5

    # --- Current player features ---
    for r in Resource:
        feat[off] = player.resources[r] / 10.0
        off += 1
    # Dev card counts
    for dc in DevCard:
        feat[off] = sum(1 for c in player.dev_cards if c == dc) / 5.0
        off += 1
    feat[off] = player.knights_played / 5.0; off += 1
    feat[off] = player.victory_points / 10.0; off += 1
    feat[off] = 1.0 if player.has_longest_road else 0.0; off += 1
    feat[off] = 1.0 if player.has_largest_army else 0.0; off += 1
    feat[off] = (MAX_SETTLEMENTS - player.num_settlements) / MAX_SETTLEMENTS; off += 1
    feat[off] = (4 - player.num_cities) / 4.0; off += 1
    feat[off] = (15 - player.num_roads) / 15.0; off += 1

    # --- Other players ---
    others = [gs.players[(player_idx + i) % len(gs.players)] for i in range(1, len(gs.players))]
    for op in others:
        feat[off] = op.total_resources / 20.0; off += 1
        feat[off] = len(op.dev_cards) / 10.0; off += 1
        feat[off] = op.knights_played / 5.0; off += 1
        feat[off] = op.victory_points / 10.0; off += 1
        feat[off] = 1.0 if op.has_longest_road else 0.0; off += 1
        feat[off] = 1.0 if op.has_largest_army else 0.0; off += 1
        feat[off] = op.num_settlements / 5.0; off += 1
        feat[off] = op.num_cities / 4.0; off += 1
        feat[off] = op.num_roads / 15.0; off += 1

    # --- Game state ---
    phase_idx = int(gs.phase)
    if phase_idx < 12:
        feat[off + phase_idx] = 1.0
    off += 12
    feat[off] = min(gs.turn_number, 200) / 200.0; off += 1

    return feat
