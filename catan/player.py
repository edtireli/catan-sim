"""Player state for Catan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from .constants import (
    DevCard,
    HarborType,
    MAX_CITIES,
    MAX_ROADS,
    MAX_SETTLEMENTS,
    NUM_RESOURCE_TYPES,
    Resource,
)


@dataclass
class Player:
    """Tracks a single player's state."""

    index: int  # 0-3
    color: str = ""

    # Resources in hand
    resources: Dict[Resource, int] = field(default_factory=lambda: {r: 0 for r in Resource})

    # Development cards in hand (not yet played)
    dev_cards: List[DevCard] = field(default_factory=list)

    # Dev cards bought this turn (cannot be played same turn)
    new_dev_cards: List[DevCard] = field(default_factory=list)

    # Counts of played dev card types
    knights_played: int = 0
    dev_cards_played_this_turn: int = 0

    # Buildings placed on the board (vertex/edge IDs)
    settlement_vertices: Set[int] = field(default_factory=set)
    city_vertices: Set[int] = field(default_factory=set)
    road_edges: Set[int] = field(default_factory=set)

    # Harbors this player has access to
    harbors: Set[HarborType] = field(default_factory=set)

    # Special awards
    has_longest_road: bool = False
    has_largest_army: bool = False
    longest_road_length: int = 0

    # Hidden VP cards (revealed only to win)
    hidden_vp_cards: int = 0

    @property
    def total_resources(self) -> int:
        return sum(self.resources.values())

    @property
    def num_settlements(self) -> int:
        return len(self.settlement_vertices)

    @property
    def num_cities(self) -> int:
        return len(self.city_vertices)

    @property
    def num_roads(self) -> int:
        return len(self.road_edges)

    @property
    def can_build_settlement(self) -> bool:
        return self.num_settlements < MAX_SETTLEMENTS

    @property
    def can_build_city(self) -> bool:
        return self.num_cities < MAX_CITIES

    @property
    def can_build_road(self) -> bool:
        return self.num_roads < MAX_ROADS

    @property
    def victory_points(self) -> int:
        vp = 0
        vp += self.num_settlements
        vp += self.num_cities * 2
        if self.has_longest_road:
            vp += 2
        if self.has_largest_army:
            vp += 2
        vp += self.hidden_vp_cards
        return vp

    def can_afford(self, cost: Dict[Resource, int]) -> bool:
        return all(self.resources.get(r, 0) >= amt for r, amt in cost.items())

    def pay(self, cost: Dict[Resource, int]) -> None:
        for r, amt in cost.items():
            self.resources[r] -= amt

    def receive(self, resource: Resource, amount: int = 1) -> None:
        self.resources[resource] += amount

    def start_turn(self) -> None:
        """Call at the beginning of each turn."""
        # Move new dev cards to playable hand
        self.dev_cards.extend(self.new_dev_cards)
        self.new_dev_cards.clear()
        self.dev_cards_played_this_turn = 0
