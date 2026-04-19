"""Game constants for standard Catan."""

from enum import IntEnum, auto

# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

class Resource(IntEnum):
    BRICK = 0
    LUMBER = 1
    ORE = 2
    GRAIN = 3
    WOOL = 4

RESOURCE_NAMES = {
    Resource.BRICK: "brick",
    Resource.LUMBER: "lumber",
    Resource.ORE: "ore",
    Resource.GRAIN: "grain",
    Resource.WOOL: "wool",
}

NUM_RESOURCE_TYPES = 5

# Bank starts with 19 of each resource
BANK_RESOURCE_COUNT = 19

# ---------------------------------------------------------------------------
# Terrain / hex types
# ---------------------------------------------------------------------------

class Terrain(IntEnum):
    HILLS = 0      # produces brick
    FOREST = 1     # produces lumber
    MOUNTAINS = 2  # produces ore
    FIELDS = 3     # produces grain
    PASTURE = 4    # produces wool
    DESERT = 5     # produces nothing

TERRAIN_RESOURCE = {
    Terrain.HILLS: Resource.BRICK,
    Terrain.FOREST: Resource.LUMBER,
    Terrain.MOUNTAINS: Resource.ORE,
    Terrain.FIELDS: Resource.GRAIN,
    Terrain.PASTURE: Resource.WOOL,
    Terrain.DESERT: None,
}

# Standard tile distribution (19 tiles)
TERRAIN_COUNTS = {
    Terrain.HILLS: 3,
    Terrain.FOREST: 4,
    Terrain.MOUNTAINS: 3,
    Terrain.FIELDS: 4,
    Terrain.PASTURE: 4,
    Terrain.DESERT: 1,
}

# Number tokens placed on tiles (18 tokens, no token on desert)
NUMBER_TOKENS = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]

# ---------------------------------------------------------------------------
# Development cards
# ---------------------------------------------------------------------------

class DevCard(IntEnum):
    KNIGHT = 0
    VICTORY_POINT = 1
    ROAD_BUILDING = 2
    YEAR_OF_PLENTY = 3
    MONOPOLY = 4

DEV_CARD_COUNTS = {
    DevCard.KNIGHT: 14,
    DevCard.VICTORY_POINT: 5,
    DevCard.ROAD_BUILDING: 2,
    DevCard.YEAR_OF_PLENTY: 2,
    DevCard.MONOPOLY: 2,
}

# ---------------------------------------------------------------------------
# Building costs
# ---------------------------------------------------------------------------

ROAD_COST = {Resource.BRICK: 1, Resource.LUMBER: 1}
SETTLEMENT_COST = {Resource.BRICK: 1, Resource.LUMBER: 1, Resource.GRAIN: 1, Resource.WOOL: 1}
CITY_COST = {Resource.ORE: 3, Resource.GRAIN: 2}
DEV_CARD_COST = {Resource.ORE: 1, Resource.GRAIN: 1, Resource.WOOL: 1}

# ---------------------------------------------------------------------------
# Building limits per player
# ---------------------------------------------------------------------------

MAX_ROADS = 15
MAX_SETTLEMENTS = 5
MAX_CITIES = 4

# ---------------------------------------------------------------------------
# Harbors
# ---------------------------------------------------------------------------

class HarborType(IntEnum):
    GENERIC = 0    # 3:1 any resource
    BRICK = 1      # 2:1 brick
    LUMBER = 2     # 2:1 lumber
    ORE = 3        # 2:1 ore
    GRAIN = 4      # 2:1 grain
    WOOL = 5       # 2:1 wool

HARBOR_RESOURCE = {
    HarborType.BRICK: Resource.BRICK,
    HarborType.LUMBER: Resource.LUMBER,
    HarborType.ORE: Resource.ORE,
    HarborType.GRAIN: Resource.GRAIN,
    HarborType.WOOL: Resource.WOOL,
}

# Standard harbor distribution (9 harbors)
HARBOR_COUNTS = {
    HarborType.GENERIC: 4,
    HarborType.BRICK: 1,
    HarborType.LUMBER: 1,
    HarborType.ORE: 1,
    HarborType.GRAIN: 1,
    HarborType.WOOL: 1,
}

# ---------------------------------------------------------------------------
# Game phases
# ---------------------------------------------------------------------------

class GamePhase(IntEnum):
    SETUP_SETTLEMENT_1 = 0
    SETUP_ROAD_1 = 1
    SETUP_SETTLEMENT_2 = 2
    SETUP_ROAD_2 = 3
    ROLL_DICE = 4
    DISCARD = 5
    MOVE_ROBBER = 6
    STEAL = 7
    MAIN_TURN = 8
    GAME_OVER = 9
    ROAD_BUILDING_1 = 10
    ROAD_BUILDING_2 = 11

# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

class ActionType(IntEnum):
    ROLL_DICE = 0
    END_TURN = 1
    BUILD_ROAD = 2
    BUILD_SETTLEMENT = 3
    BUILD_CITY = 4
    BUY_DEV_CARD = 5
    PLAY_KNIGHT = 6
    PLAY_ROAD_BUILDING = 7
    PLAY_YEAR_OF_PLENTY = 8
    PLAY_MONOPOLY = 9
    TRADE_BANK = 10
    PLACE_ROBBER = 11
    STEAL_FROM = 12
    DISCARD_RESOURCES = 13
    PLACE_SETUP_SETTLEMENT = 14
    PLACE_SETUP_ROAD = 15
    PLACE_ROAD_BUILDING_ROAD = 16

# ---------------------------------------------------------------------------
# Victory
# ---------------------------------------------------------------------------

VICTORY_POINTS_TO_WIN = 10
LONGEST_ROAD_MIN = 5
LARGEST_ARMY_MIN = 3

# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------

NUM_PLAYERS = 4

PLAYER_COLORS = ["red", "blue", "white", "orange"]
