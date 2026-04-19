"""Complete Catan game state and action logic.

This module ties together board, players, development cards, trading, and
all rule enforcement for a standard 4-player Catan game.
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .board import Board, Edge, HexTile, Vertex, generate_board
from .constants import (
    BANK_RESOURCE_COUNT,
    CITY_COST,
    DEV_CARD_COST,
    DEV_CARD_COUNTS,
    ActionType,
    DevCard,
    GamePhase,
    HarborType,
    HARBOR_RESOURCE,
    LARGEST_ARMY_MIN,
    LONGEST_ROAD_MIN,
    MAX_CITIES,
    MAX_ROADS,
    MAX_SETTLEMENTS,
    NUM_PLAYERS,
    PLAYER_COLORS,
    ROAD_COST,
    Resource,
    SETTLEMENT_COST,
    Terrain,
    TERRAIN_RESOURCE,
    VICTORY_POINTS_TO_WIN,
)
from .player import Player


@dataclass
class Action:
    """Represents an action a player can take."""
    action_type: ActionType
    # Generic payload — meaning depends on action_type
    vertex: Optional[int] = None       # vertex id for build / setup
    edge: Optional[int] = None         # edge id for road
    hex_id: Optional[int] = None       # hex id for robber placement
    target_player: Optional[int] = None  # player index for stealing
    resource: Optional[Resource] = None  # for monopoly / year of plenty
    resource2: Optional[Resource] = None # second resource for year of plenty
    give_resource: Optional[Resource] = None   # for bank trade
    get_resource: Optional[Resource] = None    # for bank trade
    discard: Optional[Dict[Resource, int]] = None  # for discard

    def __repr__(self) -> str:
        parts = [self.action_type.name]
        if self.vertex is not None:
            parts.append(f"v={self.vertex}")
        if self.edge is not None:
            parts.append(f"e={self.edge}")
        if self.hex_id is not None:
            parts.append(f"hex={self.hex_id}")
        if self.target_player is not None:
            parts.append(f"target={self.target_player}")
        if self.resource is not None:
            parts.append(f"res={self.resource.name}")
        return f"Action({', '.join(parts)})"


@dataclass
class GameState:
    """Full mutable game state."""

    board: Board = field(default_factory=Board)
    players: List[Player] = field(default_factory=list)
    bank: Dict[Resource, int] = field(
        default_factory=lambda: {r: BANK_RESOURCE_COUNT for r in Resource}
    )

    # Development card deck (shuffled)
    dev_card_deck: List[DevCard] = field(default_factory=list)

    # Turn tracking
    current_player_idx: int = 0
    phase: GamePhase = GamePhase.SETUP_SETTLEMENT_1
    turn_number: int = 0
    dice_roll: Optional[Tuple[int, int]] = None

    # Setup tracking
    setup_round: int = 1  # 1 or 2
    setup_turns_done: int = 0

    # Robber
    robber_hex: int = -1  # hex_id where robber sits

    # Longest road / largest army tracking
    longest_road_player: Optional[int] = None
    largest_army_player: Optional[int] = None

    # Winner
    winner: Optional[int] = None

    # Action log
    action_log: List[Tuple[int, Action]] = field(default_factory=list)

    # Discards pending (player_idx -> number to discard)
    pending_discards: Dict[int, int] = field(default_factory=dict)
    discard_order: List[int] = field(default_factory=list)

    # RNG
    _rng: random.Random = field(default_factory=random.Random)


def new_game(seed: Optional[int] = None, num_players: int = NUM_PLAYERS) -> GameState:
    """Create a fresh game state."""
    rng = random.Random(seed)
    gs = GameState()
    gs._rng = rng
    gs.board = generate_board(seed=rng.randint(0, 2**31))

    # Create players
    for i in range(num_players):
        gs.players.append(Player(index=i, color=PLAYER_COLORS[i]))

    # Build dev card deck
    deck: List[DevCard] = []
    for card, count in DEV_CARD_COUNTS.items():
        deck.extend([card] * count)
    rng.shuffle(deck)
    gs.dev_card_deck = deck

    # Place robber on desert
    for h in gs.board.hexes.values():
        if h.terrain == Terrain.DESERT:
            gs.robber_hex = h.hex_id
            h.has_robber = True
            break

    gs.phase = GamePhase.SETUP_SETTLEMENT_1
    gs.current_player_idx = 0
    return gs


# ===================================================================
# Legal action generation
# ===================================================================

def get_legal_actions(gs: GameState) -> List[Action]:
    """Return all legal actions for the current player in the current phase."""
    phase = gs.phase
    pid = gs.current_player_idx
    player = gs.players[pid]
    board = gs.board
    actions: List[Action] = []

    if phase == GamePhase.SETUP_SETTLEMENT_1 or phase == GamePhase.SETUP_SETTLEMENT_2:
        # Place a settlement on any unoccupied vertex (distance rule)
        for vid, v in board.vertices.items():
            if _can_place_settlement_setup(board, vid):
                actions.append(Action(ActionType.PLACE_SETUP_SETTLEMENT, vertex=vid))

    elif phase == GamePhase.SETUP_ROAD_1 or phase == GamePhase.SETUP_ROAD_2:
        # Place a road adjacent to the settlement just placed
        last_settlement = _last_settlement(gs, pid)
        if last_settlement is not None:
            for eid in board.vertices[last_settlement].adjacent_edges:
                edge = board.edges[eid]
                if edge.road_owner is None:
                    actions.append(Action(ActionType.PLACE_SETUP_ROAD, edge=eid))

    elif phase == GamePhase.ROLL_DICE:
        actions.append(Action(ActionType.ROLL_DICE))
        # Can play knight before rolling
        if (DevCard.KNIGHT in player.dev_cards and
                player.dev_cards_played_this_turn == 0):
            for hid in range(gs.board.num_hexes):
                if hid != gs.robber_hex:
                    actions.append(Action(ActionType.PLAY_KNIGHT, hex_id=hid))

    elif phase == GamePhase.DISCARD:
        # Current player in discard_order must discard
        if gs.discard_order:
            disc_pid = gs.discard_order[0]
            disc_player = gs.players[disc_pid]
            n = gs.pending_discards[disc_pid]
            # Generate a representative set of discard options
            discard_options = _generate_discard_options(disc_player, n)
            for d in discard_options:
                actions.append(Action(ActionType.DISCARD_RESOURCES, discard=d))

    elif phase == GamePhase.MOVE_ROBBER:
        for hid in range(gs.board.num_hexes):
            if hid != gs.robber_hex:
                actions.append(Action(ActionType.PLACE_ROBBER, hex_id=hid))

    elif phase == GamePhase.STEAL:
        # Steal from a player adjacent to the new robber hex
        targets = _steal_targets(gs)
        if not targets:
            # No one to steal from → auto-skip (add a dummy end action)
            actions.append(Action(ActionType.STEAL_FROM, target_player=None))
        else:
            for tp in targets:
                actions.append(Action(ActionType.STEAL_FROM, target_player=tp))

    elif phase == GamePhase.MAIN_TURN:
        actions.append(Action(ActionType.END_TURN))
        _add_build_actions(gs, pid, actions)
        _add_dev_card_actions(gs, pid, actions)
        _add_bank_trade_actions(gs, pid, actions)

    elif phase == GamePhase.ROAD_BUILDING_1 or phase == GamePhase.ROAD_BUILDING_2:
        for eid in range(gs.board.num_edges):
            if _can_place_road(gs, pid, eid):
                actions.append(Action(ActionType.PLACE_ROAD_BUILDING_ROAD, edge=eid))
        # If no roads can be placed, allow skipping
        if not actions:
            actions.append(Action(ActionType.END_TURN))

    elif phase == GamePhase.GAME_OVER:
        pass  # No actions

    return actions


# ===================================================================
# Action execution
# ===================================================================

def apply_action(gs: GameState, action: Action) -> GameState:
    """Apply an action to the game state (mutates in place). Returns gs."""
    pid = gs.current_player_idx
    player = gs.players[pid]
    board = gs.board

    gs.action_log.append((pid, action))

    at = action.action_type

    if at == ActionType.PLACE_SETUP_SETTLEMENT:
        _place_settlement(gs, pid, action.vertex, free=True)
        if gs.phase == GamePhase.SETUP_SETTLEMENT_1:
            gs.phase = GamePhase.SETUP_ROAD_1
        else:
            gs.phase = GamePhase.SETUP_ROAD_2
            # Collect resources from adjacent hexes for 2nd settlement
            if gs.setup_round == 2:
                _collect_setup_resources(gs, pid, action.vertex)

    elif at == ActionType.PLACE_SETUP_ROAD:
        _place_road(gs, pid, action.edge, free=True)
        gs.setup_turns_done += 1
        if gs.phase == GamePhase.SETUP_ROAD_1:
            if gs.setup_turns_done < len(gs.players):
                gs.current_player_idx = (pid + 1) % len(gs.players)
                gs.phase = GamePhase.SETUP_SETTLEMENT_1
            else:
                # Start round 2 (reverse order)
                gs.setup_round = 2
                gs.setup_turns_done = 0
                gs.current_player_idx = len(gs.players) - 1
                gs.phase = GamePhase.SETUP_SETTLEMENT_2
        else:  # SETUP_ROAD_2
            if gs.setup_turns_done < len(gs.players):
                gs.current_player_idx = pid - 1
                gs.phase = GamePhase.SETUP_SETTLEMENT_2
            else:
                # Setup complete → start normal play
                gs.current_player_idx = 0
                gs.players[0].start_turn()
                gs.phase = GamePhase.ROLL_DICE
                gs.turn_number = 1

    elif at == ActionType.ROLL_DICE:
        d1 = gs._rng.randint(1, 6)
        d2 = gs._rng.randint(1, 6)
        gs.dice_roll = (d1, d2)
        total = d1 + d2
        if total == 7:
            _handle_seven(gs)
        else:
            _distribute_resources(gs, total)
            gs.phase = GamePhase.MAIN_TURN

    elif at == ActionType.END_TURN:
        _end_turn(gs)

    elif at == ActionType.BUILD_ROAD:
        player.pay(ROAD_COST)
        _return_to_bank(gs, ROAD_COST)
        _place_road(gs, pid, action.edge)
        _check_longest_road(gs)

    elif at == ActionType.BUILD_SETTLEMENT:
        player.pay(SETTLEMENT_COST)
        _return_to_bank(gs, SETTLEMENT_COST)
        _place_settlement(gs, pid, action.vertex)
        _check_longest_road(gs)
        _check_victory(gs)

    elif at == ActionType.BUILD_CITY:
        player.pay(CITY_COST)
        _return_to_bank(gs, CITY_COST)
        _place_city(gs, pid, action.vertex)
        _check_victory(gs)

    elif at == ActionType.BUY_DEV_CARD:
        player.pay(DEV_CARD_COST)
        _return_to_bank(gs, DEV_CARD_COST)
        card = gs.dev_card_deck.pop()
        if card == DevCard.VICTORY_POINT:
            player.hidden_vp_cards += 1
            player.new_dev_cards.append(card)
        else:
            player.new_dev_cards.append(card)
        _check_victory(gs)

    elif at == ActionType.PLAY_KNIGHT:
        player.dev_cards.remove(DevCard.KNIGHT)
        player.knights_played += 1
        player.dev_cards_played_this_turn += 1
        _check_largest_army(gs)
        # Move robber
        if gs.phase == GamePhase.ROLL_DICE:
            # Pre-roll knight: will go to move robber, then come back to roll
            gs.robber_hex = _move_robber(gs, action.hex_id)
            gs.phase = GamePhase.STEAL
        else:
            gs.robber_hex = _move_robber(gs, action.hex_id)
            gs.phase = GamePhase.STEAL

    elif at == ActionType.PLAY_ROAD_BUILDING:
        player.dev_cards.remove(DevCard.ROAD_BUILDING)
        player.dev_cards_played_this_turn += 1
        gs.phase = GamePhase.ROAD_BUILDING_1

    elif at == ActionType.PLAY_YEAR_OF_PLENTY:
        player.dev_cards.remove(DevCard.YEAR_OF_PLENTY)
        player.dev_cards_played_this_turn += 1
        r1 = action.resource
        r2 = action.resource2
        if gs.bank[r1] > 0:
            player.receive(r1)
            gs.bank[r1] -= 1
        if r2 is not None and gs.bank[r2] > 0:
            player.receive(r2)
            gs.bank[r2] -= 1

    elif at == ActionType.PLAY_MONOPOLY:
        player.dev_cards.remove(DevCard.MONOPOLY)
        player.dev_cards_played_this_turn += 1
        res = action.resource
        for other in gs.players:
            if other.index != pid:
                amount = other.resources[res]
                other.resources[res] = 0
                player.resources[res] += amount

    elif at == ActionType.TRADE_BANK:
        give = action.give_resource
        get = action.get_resource
        ratio = _trade_ratio(player, give)
        player.resources[give] -= ratio
        gs.bank[give] += ratio
        player.resources[get] += 1
        gs.bank[get] -= 1

    elif at == ActionType.PLACE_ROBBER:
        gs.robber_hex = _move_robber(gs, action.hex_id)
        gs.phase = GamePhase.STEAL

    elif at == ActionType.STEAL_FROM:
        if action.target_player is not None:
            _steal(gs, pid, action.target_player)
        # Return to appropriate phase
        if gs.dice_roll is None:
            gs.phase = GamePhase.ROLL_DICE
        else:
            gs.phase = GamePhase.MAIN_TURN

    elif at == ActionType.DISCARD_RESOURCES:
        disc_pid = gs.discard_order[0]
        disc_player = gs.players[disc_pid]
        for r, amt in action.discard.items():
            disc_player.resources[r] -= amt
            gs.bank[r] += amt
        gs.discard_order.pop(0)
        del gs.pending_discards[disc_pid]
        if not gs.discard_order:
            gs.phase = GamePhase.MOVE_ROBBER

    elif at == ActionType.PLACE_ROAD_BUILDING_ROAD:
        _place_road(gs, pid, action.edge, free=True)
        if gs.phase == GamePhase.ROAD_BUILDING_1:
            gs.phase = GamePhase.ROAD_BUILDING_2
        else:
            gs.phase = GamePhase.MAIN_TURN
            _check_longest_road(gs)

    return gs


# ===================================================================
# Internal helpers
# ===================================================================

def _last_settlement(gs: GameState, pid: int) -> Optional[int]:
    """Return the vertex_id of the most recently placed settlement by pid."""
    for player_idx, action in reversed(gs.action_log):
        if player_idx == pid and action.action_type == ActionType.PLACE_SETUP_SETTLEMENT:
            return action.vertex
    return None


def _can_place_settlement_setup(board: Board, vid: int) -> bool:
    """Check if a settlement can be placed at vertex during setup (no adjacency req to own road)."""
    v = board.vertices[vid]
    if v.building is not None:
        return False
    # Distance rule: no adjacent vertex has a building
    for adj_vid in v.adjacent_vertices:
        if board.vertices[adj_vid].building is not None:
            return False
    return True


def _can_place_settlement(gs: GameState, pid: int, vid: int) -> bool:
    """Check if player can place a settlement at vertex during normal play."""
    board = gs.board
    v = board.vertices[vid]
    if v.building is not None:
        return False
    # Distance rule
    for adj_vid in v.adjacent_vertices:
        if board.vertices[adj_vid].building is not None:
            return False
    # Must be connected to one of player's roads
    for eid in v.adjacent_edges:
        if board.edges[eid].road_owner == pid:
            return True
    return False


def _can_place_road(gs: GameState, pid: int, eid: int) -> bool:
    """Check if player can place a road on edge."""
    board = gs.board
    edge = board.edges[eid]
    if edge.road_owner is not None:
        return False
    player = gs.players[pid]
    if not player.can_build_road:
        return False
    # Must connect to player's existing road, settlement, or city
    v1, v2 = edge.vertices
    for vid in (v1, v2):
        v = board.vertices[vid]
        # Connected if player has a building here
        if v.building_owner == pid:
            return True
        # Connected if player has a road to this vertex
        for adj_eid in v.adjacent_edges:
            if adj_eid != eid and board.edges[adj_eid].road_owner == pid:
                # But not through an opponent's building
                if v.building_owner is None or v.building_owner == pid:
                    return True
    return False


def _place_settlement(gs: GameState, pid: int, vid: int, free: bool = False) -> None:
    board = gs.board
    v = board.vertices[vid]
    v.building = "settlement"
    v.building_owner = pid
    player = gs.players[pid]
    player.settlement_vertices.add(vid)
    # Check for harbor access
    if v.harbor is not None:
        player.harbors.add(v.harbor)


def _place_road(gs: GameState, pid: int, eid: int, free: bool = False) -> None:
    board = gs.board
    edge = board.edges[eid]
    edge.road_owner = pid
    gs.players[pid].road_edges.add(eid)


def _place_city(gs: GameState, pid: int, vid: int) -> None:
    board = gs.board
    v = board.vertices[vid]
    v.building = "city"
    player = gs.players[pid]
    player.settlement_vertices.discard(vid)
    player.city_vertices.add(vid)


def _collect_setup_resources(gs: GameState, pid: int, vid: int) -> None:
    """Collect one resource from each hex adjacent to the 2nd settlement."""
    board = gs.board
    player = gs.players[pid]
    v = board.vertices[vid]
    for hid in v.adjacent_hexes:
        h = board.hexes[hid]
        res = TERRAIN_RESOURCE.get(h.terrain)
        if res is not None and gs.bank[res] > 0:
            player.receive(res)
            gs.bank[res] -= 1


def _distribute_resources(gs: GameState, roll: int) -> None:
    """Distribute resources based on dice roll."""
    board = gs.board
    for h in board.hexes.values():
        if h.number != roll or h.has_robber:
            continue
        res = TERRAIN_RESOURCE.get(h.terrain)
        if res is None:
            continue
        # Count how many resources are needed
        receivers: List[Tuple[int, int]] = []  # (player_idx, amount)
        for vid in h.vertex_ids:
            v = board.vertices[vid]
            if v.building_owner is not None:
                amt = 2 if v.building == "city" else 1
                receivers.append((v.building_owner, amt))
        total_needed = sum(a for _, a in receivers)
        if total_needed <= gs.bank[res]:
            for pi, amt in receivers:
                gs.players[pi].receive(res, amt)
                gs.bank[res] -= amt
        # If bank doesn't have enough, no one gets any (standard rule)


def _handle_seven(gs: GameState) -> None:
    """Handle a 7 being rolled."""
    # 1. Players with >7 cards must discard half
    gs.pending_discards.clear()
    gs.discard_order.clear()
    for p in gs.players:
        if p.total_resources > 7:
            gs.pending_discards[p.index] = p.total_resources // 2
            gs.discard_order.append(p.index)
    if gs.discard_order:
        gs.phase = GamePhase.DISCARD
    else:
        gs.phase = GamePhase.MOVE_ROBBER


def _move_robber(gs: GameState, hex_id: int) -> int:
    """Move robber to hex_id. Returns the hex_id."""
    board = gs.board
    # Remove from old position
    if gs.robber_hex >= 0:
        board.hexes[gs.robber_hex].has_robber = False
    board.hexes[hex_id].has_robber = True
    return hex_id


def _steal_targets(gs: GameState) -> List[int]:
    """Get players that can be stolen from at the robber hex."""
    pid = gs.current_player_idx
    board = gs.board
    robber_hex = board.hexes[gs.robber_hex]
    targets: Set[int] = set()
    for vid in robber_hex.vertex_ids:
        v = board.vertices[vid]
        if v.building_owner is not None and v.building_owner != pid:
            if gs.players[v.building_owner].total_resources > 0:
                targets.add(v.building_owner)
    return sorted(targets)


def _steal(gs: GameState, thief: int, victim: int) -> None:
    """Steal a random resource from victim."""
    v_player = gs.players[victim]
    if v_player.total_resources == 0:
        return
    # Build pool of stealable resources
    pool: List[Resource] = []
    for r in Resource:
        pool.extend([r] * v_player.resources[r])
    stolen = gs._rng.choice(pool)
    v_player.resources[stolen] -= 1
    gs.players[thief].resources[stolen] += 1


def _trade_ratio(player: Player, resource: Resource) -> int:
    """Get the bank trade ratio for a resource."""
    # Check specific harbor
    harbor_map = {
        Resource.BRICK: HarborType.BRICK,
        Resource.LUMBER: HarborType.LUMBER,
        Resource.ORE: HarborType.ORE,
        Resource.GRAIN: HarborType.GRAIN,
        Resource.WOOL: HarborType.WOOL,
    }
    if harbor_map[resource] in player.harbors:
        return 2
    if HarborType.GENERIC in player.harbors:
        return 3
    return 4


def _end_turn(gs: GameState) -> None:
    """End the current player's turn and advance."""
    n = len(gs.players)
    gs.current_player_idx = (gs.current_player_idx + 1) % n
    gs.turn_number += 1
    gs.dice_roll = None
    gs.players[gs.current_player_idx].start_turn()
    gs.phase = GamePhase.ROLL_DICE


def _return_to_bank(gs: GameState, cost: Dict[Resource, int]) -> None:
    for r, amt in cost.items():
        gs.bank[r] += amt


def _add_build_actions(gs: GameState, pid: int, actions: List[Action]) -> None:
    """Add all legal build actions."""
    player = gs.players[pid]
    board = gs.board

    # Roads
    if player.can_afford(ROAD_COST) and player.can_build_road:
        for eid in range(board.num_edges):
            if _can_place_road(gs, pid, eid):
                actions.append(Action(ActionType.BUILD_ROAD, edge=eid))

    # Settlements
    if player.can_afford(SETTLEMENT_COST) and player.can_build_settlement:
        for vid in range(board.num_vertices):
            if _can_place_settlement(gs, pid, vid):
                actions.append(Action(ActionType.BUILD_SETTLEMENT, vertex=vid))

    # Cities
    if player.can_afford(CITY_COST) and player.can_build_city:
        for vid in player.settlement_vertices:
            actions.append(Action(ActionType.BUILD_CITY, vertex=vid))

    # Dev cards
    if player.can_afford(DEV_CARD_COST) and len(gs.dev_card_deck) > 0:
        actions.append(Action(ActionType.BUY_DEV_CARD))


def _add_dev_card_actions(gs: GameState, pid: int, actions: List[Action]) -> None:
    """Add development card play actions."""
    player = gs.players[pid]
    if player.dev_cards_played_this_turn > 0:
        return  # Max 1 dev card per turn

    if DevCard.KNIGHT in player.dev_cards:
        for hid in range(gs.board.num_hexes):
            if hid != gs.robber_hex:
                actions.append(Action(ActionType.PLAY_KNIGHT, hex_id=hid))

    if DevCard.ROAD_BUILDING in player.dev_cards:
        if player.can_build_road:
            actions.append(Action(ActionType.PLAY_ROAD_BUILDING))

    if DevCard.YEAR_OF_PLENTY in player.dev_cards:
        for r1 in Resource:
            for r2 in Resource:
                if gs.bank[r1] > 0 and (r1 == r2 and gs.bank[r1] > 1 or r1 != r2 and gs.bank[r2] > 0):
                    actions.append(Action(ActionType.PLAY_YEAR_OF_PLENTY, resource=r1, resource2=r2))

    if DevCard.MONOPOLY in player.dev_cards:
        for r in Resource:
            actions.append(Action(ActionType.PLAY_MONOPOLY, resource=r))


def _add_bank_trade_actions(gs: GameState, pid: int, actions: List[Action]) -> None:
    """Add bank/harbor trade actions."""
    player = gs.players[pid]
    for give in Resource:
        ratio = _trade_ratio(player, give)
        if player.resources[give] >= ratio:
            for get in Resource:
                if get != give and gs.bank[get] > 0:
                    actions.append(Action(
                        ActionType.TRADE_BANK,
                        give_resource=give,
                        get_resource=get,
                    ))


def _generate_discard_options(player: Player, n: int) -> List[Dict[Resource, int]]:
    """Generate discard options. For AI, we generate a manageable set."""
    # Generate up to ~50 options by iterating combinations
    options: List[Dict[Resource, int]] = []
    resources = [r for r in Resource if player.resources[r] > 0]

    def _recurse(remaining: int, idx: int, current: Dict[Resource, int]) -> None:
        if len(options) >= 50:
            return
        if remaining == 0:
            options.append(dict(current))
            return
        if idx >= len(resources):
            return
        r = resources[idx]
        max_discard = min(remaining, player.resources[r])
        for amt in range(max_discard + 1):
            if amt > 0:
                current[r] = amt
            _recurse(remaining - amt, idx + 1, current)
            if amt > 0:
                del current[r]

    _recurse(n, 0, {})

    # If no valid options found (shouldn't happen), create a simple one
    if not options:
        d: Dict[Resource, int] = {}
        left = n
        for r in Resource:
            take = min(left, player.resources[r])
            if take > 0:
                d[r] = take
                left -= take
            if left == 0:
                break
        options.append(d)

    return options


# ===================================================================
# Longest road calculation
# ===================================================================

def _calc_longest_road(gs: GameState, pid: int) -> int:
    """Calculate the longest continuous road for a player using DFS."""
    board = gs.board
    player = gs.players[pid]

    if not player.road_edges:
        return 0

    # Build adjacency graph of player's roads
    # Node = vertex_id, edges = player's road edges
    best = 0

    def dfs(vertex: int, visited_edges: Set[int], length: int) -> None:
        nonlocal best
        best = max(best, length)
        v = board.vertices[vertex]
        for eid in v.adjacent_edges:
            if eid in visited_edges:
                continue
            edge = board.edges[eid]
            if edge.road_owner != pid:
                continue
            # Check if path is blocked by opponent's building
            other_v = edge.vertices[0] if edge.vertices[1] == vertex else edge.vertices[1]
            ov = board.vertices[other_v]
            if ov.building_owner is not None and ov.building_owner != pid:
                continue  # Blocked by opponent
            visited_edges.add(eid)
            dfs(other_v, visited_edges, length + 1)
            visited_edges.remove(eid)

    # Start DFS from each endpoint of each road
    for eid in player.road_edges:
        edge = board.edges[eid]
        for start_v in edge.vertices:
            dfs(start_v, {eid}, 1)

    return best


def _check_longest_road(gs: GameState) -> None:
    """Update longest road ownership."""
    best_len = LONGEST_ROAD_MIN - 1
    best_player = None

    for p in gs.players:
        length = _calc_longest_road(gs, p.index)
        p.longest_road_length = length
        if length > best_len:
            best_len = length
            best_player = p.index

    # Update ownership
    old_owner = gs.longest_road_player
    if best_player is not None and best_player != old_owner:
        if old_owner is not None:
            gs.players[old_owner].has_longest_road = False
        gs.players[best_player].has_longest_road = True
        gs.longest_road_player = best_player
    elif best_player is None and old_owner is not None:
        gs.players[old_owner].has_longest_road = False
        gs.longest_road_player = None


def _check_largest_army(gs: GameState) -> None:
    """Update largest army ownership."""
    best_count = LARGEST_ARMY_MIN - 1
    best_player = None

    for p in gs.players:
        if p.knights_played > best_count:
            best_count = p.knights_played
            best_player = p.index

    old_owner = gs.largest_army_player
    if best_player is not None and best_player != old_owner:
        if old_owner is not None:
            gs.players[old_owner].has_largest_army = False
        gs.players[best_player].has_largest_army = True
        gs.largest_army_player = best_player


def _check_victory(gs: GameState) -> None:
    """Check if any player has won."""
    for p in gs.players:
        if p.victory_points >= VICTORY_POINTS_TO_WIN:
            gs.winner = p.index
            gs.phase = GamePhase.GAME_OVER
            break
