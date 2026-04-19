"""Board topology and generation for standard Catan.

Coordinate system:
- Hexes use axial coordinates (q, r) with flat-top orientation.
- Vertices and edges are identified by unique integer IDs, computed from
  the geometric positions of hex corners.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .constants import (
    HARBOR_COUNTS,
    HarborType,
    NUMBER_TOKENS,
    TERRAIN_COUNTS,
    Terrain,
)

# ---------------------------------------------------------------------------
# Hex coordinate helpers
# ---------------------------------------------------------------------------

# The 19 land hex positions in axial coords (flat-top, center at origin)
HEX_COORDS: List[Tuple[int, int]] = [
    # Center
    (0, 0),
    # Ring 1
    (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
    # Ring 2
    (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -1),
    (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 1),
]

assert len(HEX_COORDS) == 19

HEX_SIZE = 1.0  # unit size for geometry calculations


def _axial_to_pixel(q: int, r: int) -> Tuple[float, float]:
    """Convert axial hex coords to pixel center (flat-top)."""
    x = HEX_SIZE * 1.5 * q
    y = HEX_SIZE * math.sqrt(3) * (r + q / 2.0)
    return (x, y)


def _hex_corner(cx: float, cy: float, i: int) -> Tuple[float, float]:
    """Corner i (0-5) of a flat-top hex centred at (cx, cy)."""
    angle = math.radians(60 * i)
    return (
        round(cx + HEX_SIZE * math.cos(angle), 6),
        round(cy + HEX_SIZE * math.sin(angle), 6),
    )


def _snap(v: float, precision: int = 4) -> float:
    return round(v, precision)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HexTile:
    """A single hex on the board."""
    hex_id: int
    q: int
    r: int
    terrain: Terrain
    number: int  # 0 for desert
    cx: float = 0.0  # pixel center x
    cy: float = 0.0  # pixel center y
    vertex_ids: List[int] = field(default_factory=list)  # 6 vertices, index 0-5
    edge_ids: List[int] = field(default_factory=list)     # 6 edges, index 0-5
    has_robber: bool = False


@dataclass
class Vertex:
    """An intersection where up to 3 hexes meet."""
    vertex_id: int
    x: float
    y: float
    adjacent_hexes: List[int] = field(default_factory=list)   # hex_ids
    adjacent_edges: List[int] = field(default_factory=list)    # edge_ids
    adjacent_vertices: List[int] = field(default_factory=list) # vertex_ids
    building: Optional[str] = None       # None / "settlement" / "city"
    building_owner: Optional[int] = None # player index
    harbor: Optional[HarborType] = None


@dataclass
class Edge:
    """A path between two adjacent vertices."""
    edge_id: int
    vertices: Tuple[int, int]  # (v1, v2)
    adjacent_hexes: List[int] = field(default_factory=list)
    road_owner: Optional[int] = None  # player index


# ---------------------------------------------------------------------------
# Board generation
# ---------------------------------------------------------------------------

@dataclass
class Board:
    hexes: Dict[int, HexTile] = field(default_factory=dict)
    vertices: Dict[int, Vertex] = field(default_factory=dict)
    edges: Dict[int, Edge] = field(default_factory=dict)
    # quick lookups
    hex_by_coord: Dict[Tuple[int, int], int] = field(default_factory=dict)
    _edge_by_verts: Dict[FrozenSet[int], int] = field(default_factory=dict)

    # counts
    num_hexes: int = 0
    num_vertices: int = 0
    num_edges: int = 0

    def edge_between(self, v1: int, v2: int) -> Optional[int]:
        return self._edge_by_verts.get(frozenset((v1, v2)))


def generate_board(seed: Optional[int] = None) -> Board:
    """Generate a standard Catan board with randomised tiles and numbers."""
    rng = random.Random(seed)
    board = Board()

    # --- 1. Create terrain tiles ------------------------------------------
    terrain_pool: List[Terrain] = []
    for t, count in TERRAIN_COUNTS.items():
        terrain_pool.extend([t] * count)
    rng.shuffle(terrain_pool)

    # --- 2. Assign number tokens ------------------------------------------
    numbers = list(NUMBER_TOKENS)
    rng.shuffle(numbers)
    num_iter = iter(numbers)

    # --- 3. Build hex tiles -----------------------------------------------
    for idx, (q, r) in enumerate(HEX_COORDS):
        terrain = terrain_pool[idx]
        number = 0 if terrain == Terrain.DESERT else next(num_iter)
        cx, cy = _axial_to_pixel(q, r)
        tile = HexTile(
            hex_id=idx, q=q, r=r,
            terrain=terrain, number=number,
            cx=cx, cy=cy,
            has_robber=(terrain == Terrain.DESERT),
        )
        board.hexes[idx] = tile
        board.hex_by_coord[(q, r)] = idx

    # --- 4. Build vertices and edges from hex corners ---------------------
    # Map rounded (x, y) -> vertex_id
    pos_to_vid: Dict[Tuple[float, float], int] = {}
    next_vid = 0
    next_eid = 0

    for tile in board.hexes.values():
        corners = [_hex_corner(tile.cx, tile.cy, i) for i in range(6)]
        tile_vids: List[int] = []

        for cx, cy in corners:
            key = (_snap(cx), _snap(cy))
            if key not in pos_to_vid:
                pos_to_vid[key] = next_vid
                board.vertices[next_vid] = Vertex(vertex_id=next_vid, x=key[0], y=key[1])
                next_vid += 1
            vid = pos_to_vid[key]
            tile_vids.append(vid)
            if tile.hex_id not in board.vertices[vid].adjacent_hexes:
                board.vertices[vid].adjacent_hexes.append(tile.hex_id)

        tile.vertex_ids = tile_vids

        # Edges: consecutive pairs of corners
        for i in range(6):
            v1 = tile_vids[i]
            v2 = tile_vids[(i + 1) % 6]
            edge_key = frozenset((v1, v2))
            if edge_key not in board._edge_by_verts:
                eid = next_eid
                board.edges[eid] = Edge(edge_id=eid, vertices=(v1, v2))
                board._edge_by_verts[edge_key] = eid
                next_eid += 1
                # vertex adjacency
                board.vertices[v1].adjacent_edges.append(eid)
                board.vertices[v2].adjacent_edges.append(eid)
                if v2 not in board.vertices[v1].adjacent_vertices:
                    board.vertices[v1].adjacent_vertices.append(v2)
                if v1 not in board.vertices[v2].adjacent_vertices:
                    board.vertices[v2].adjacent_vertices.append(v1)

            eid = board._edge_by_verts[edge_key]
            if tile.hex_id not in board.edges[eid].adjacent_hexes:
                board.edges[eid].adjacent_hexes.append(tile.hex_id)
            tile.edge_ids.append(eid)

    board.num_hexes = len(board.hexes)
    board.num_vertices = len(board.vertices)
    board.num_edges = len(board.edges)

    # --- 5. Assign harbors ------------------------------------------------
    _assign_harbors(board, rng)

    return board


# ---------------------------------------------------------------------------
# Harbor placement
# ---------------------------------------------------------------------------

# Coastal vertex pairs that form harbor landing spots.
# We identify coastal vertices as those touching fewer than 3 hexes.

def _assign_harbors(board: Board, rng: random.Random) -> None:
    """Assign harbors to coastal vertex pairs."""
    # Find coastal edges (edges touching exactly 1 hex)
    coastal_edges: List[int] = []
    for eid, edge in board.edges.items():
        if len(edge.adjacent_hexes) == 1:
            coastal_edges.append(eid)

    # Group coastal edges into harbor landing pairs
    # We space them out roughly evenly around the coast
    # Sort coastal edges by angle from center for consistent ordering
    def _edge_angle(eid: int) -> float:
        e = board.edges[eid]
        v1, v2 = board.vertices[e.vertices[0]], board.vertices[e.vertices[1]]
        mx = (v1.x + v2.x) / 2
        my = (v1.y + v2.y) / 2
        return math.atan2(my, mx)

    coastal_edges.sort(key=_edge_angle)

    # Pick 9 roughly evenly spaced coastal edges for harbors
    if len(coastal_edges) < 9:
        return  # shouldn't happen on standard board

    step = len(coastal_edges) / 9
    harbor_edge_indices = [int(i * step) for i in range(9)]

    # Build harbor type pool
    harbor_pool: List[HarborType] = []
    for ht, count in HARBOR_COUNTS.items():
        harbor_pool.extend([ht] * count)
    rng.shuffle(harbor_pool)

    for i, ei in enumerate(harbor_edge_indices):
        eid = coastal_edges[ei]
        edge = board.edges[eid]
        ht = harbor_pool[i]
        # Both vertices of this edge get the harbor
        for vid in edge.vertices:
            board.vertices[vid].harbor = ht
