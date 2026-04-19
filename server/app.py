"""FastAPI server for Catan web interface.

Serves the React frontend and handles game communication via WebSocket.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from catan.board import Board, HexTile, Vertex, Edge
from catan.constants import (
    ActionType,
    DevCard,
    GamePhase,
    HarborType,
    Resource,
    Terrain,
    TERRAIN_RESOURCE,
    PLAYER_COLORS,
)
from catan.game import Action, GameState, apply_action, get_legal_actions, new_game
from ai.agent import CatanAgent, RandomAgent
from ai.features import action_to_index, index_to_action_from_list
from ai.network import CatanNetwork

app = FastAPI(title="Catan Simulator")

# Serve static frontend
FRONTEND_DIR = Path(__file__).parent.parent / "web" / "dist"
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    @app.get("/")
    async def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")


# -----------------------------------------------------------------------
# Game session management
# -----------------------------------------------------------------------

class GameSession:
    """Manages a single game session."""

    def __init__(
        self,
        human_player: int = 0,
        ai_difficulty: str = "medium",
        seed: Optional[int] = None,
    ):
        self.gs = new_game(seed=seed)
        self.human_player = human_player
        self.ai_difficulty = ai_difficulty

        # Load AI agents
        self.agents: Dict[int, Any] = {}
        network = self._load_network(ai_difficulty)
        for i in range(4):
            if i != human_player:
                if network is not None:
                    self.agents[i] = CatanAgent(i, network, deterministic=(ai_difficulty == "hard"))
                else:
                    self.agents[i] = RandomAgent(i)

    def _load_network(self, difficulty: str) -> Optional[CatanNetwork]:
        """Try to load a trained network."""
        checkpoint_dir = Path("checkpoints")
        if not checkpoint_dir.exists():
            return None

        # Map difficulty to checkpoint
        checkpoints = sorted(checkpoint_dir.glob("*.pt"))
        if not checkpoints:
            return None

        if difficulty == "easy":
            # Use earliest checkpoint
            cp_path = checkpoints[0]
        elif difficulty == "hard":
            # Use latest checkpoint
            cp_path = checkpoints[-1]
        else:
            # Medium: use middle checkpoint
            cp_path = checkpoints[len(checkpoints) // 2]

        network = CatanNetwork()
        checkpoint = torch.load(cp_path, map_location="cpu", weights_only=False)
        network.load_state_dict(checkpoint["model_state_dict"])
        network.eval()
        return network

    def process_ai_turns(self) -> List[Dict]:
        """Run AI turns until it's the human player's turn. Returns action log."""
        action_log = []
        safety = 0
        while (self.gs.current_player_idx != self.human_player
               and self.gs.phase != GamePhase.GAME_OVER
               and safety < 200):
            pid = self.gs.current_player_idx
            agent = self.agents.get(pid)
            if agent is None:
                break
            action = agent.choose_action(self.gs)
            apply_action(self.gs, action)
            action_log.append({"player": pid, "action": _serialize_action(action)})
            safety += 1
        return action_log


# Active sessions
sessions: Dict[str, GameSession] = {}


# -----------------------------------------------------------------------
# Serialization helpers
# -----------------------------------------------------------------------

def _serialize_board(board: Board) -> Dict:
    hexes = []
    for h in board.hexes.values():
        hexes.append({
            "id": h.hex_id,
            "q": h.q,
            "r": h.r,
            "terrain": h.terrain.name.lower(),
            "number": h.number,
            "cx": round(h.cx, 4),
            "cy": round(h.cy, 4),
            "hasRobber": h.has_robber,
            "vertices": h.vertex_ids,
        })

    vertices = []
    for v in board.vertices.values():
        vertices.append({
            "id": v.vertex_id,
            "x": round(v.x, 4),
            "y": round(v.y, 4),
            "building": v.building,
            "owner": v.building_owner,
            "harbor": v.harbor.name.lower() if v.harbor is not None else None,
        })

    edges = []
    for e in board.edges.values():
        v1 = board.vertices[e.vertices[0]]
        v2 = board.vertices[e.vertices[1]]
        edges.append({
            "id": e.edge_id,
            "v1": e.vertices[0],
            "v2": e.vertices[1],
            "x1": round(v1.x, 4),
            "y1": round(v1.y, 4),
            "x2": round(v2.x, 4),
            "y2": round(v2.y, 4),
            "road": e.road_owner,
        })

    return {"hexes": hexes, "vertices": vertices, "edges": edges}


def _serialize_player(player, is_self: bool = False) -> Dict:
    data = {
        "index": player.index,
        "color": player.color,
        "victoryPoints": player.victory_points,
        "numSettlements": player.num_settlements,
        "numCities": player.num_cities,
        "numRoads": player.num_roads,
        "knightsPlayed": player.knights_played,
        "hasLongestRoad": player.has_longest_road,
        "hasLargestArmy": player.has_largest_army,
        "longestRoadLength": player.longest_road_length,
        "totalResources": player.total_resources,
        "numDevCards": len(player.dev_cards) + len(player.new_dev_cards),
    }
    if is_self:
        data["resources"] = {r.name.lower(): player.resources[r] for r in Resource}
        data["devCards"] = [c.name.lower() for c in player.dev_cards]
        data["newDevCards"] = [c.name.lower() for c in player.new_dev_cards]
    return data


def _serialize_game_state(gs: GameState, human_player: int) -> Dict:
    return {
        "board": _serialize_board(gs.board),
        "players": [
            _serialize_player(p, is_self=(p.index == human_player))
            for p in gs.players
        ],
        "currentPlayer": gs.current_player_idx,
        "phase": gs.phase.name,
        "turnNumber": gs.turn_number,
        "diceRoll": list(gs.dice_roll) if gs.dice_roll else None,
        "winner": gs.winner,
        "robberHex": gs.robber_hex,
        "colors": PLAYER_COLORS,
    }


def _serialize_action(action: Action) -> Dict:
    d: Dict[str, Any] = {"type": action.action_type.name}
    if action.vertex is not None:
        d["vertex"] = action.vertex
    if action.edge is not None:
        d["edge"] = action.edge
    if action.hex_id is not None:
        d["hexId"] = action.hex_id
    if action.target_player is not None:
        d["targetPlayer"] = action.target_player
    if action.resource is not None:
        d["resource"] = action.resource.name.lower()
    if action.resource2 is not None:
        d["resource2"] = action.resource2.name.lower()
    if action.give_resource is not None:
        d["giveResource"] = action.give_resource.name.lower()
    if action.get_resource is not None:
        d["getResource"] = action.get_resource.name.lower()
    return d


def _serialize_legal_actions(actions: List[Action]) -> List[Dict]:
    return [_serialize_action(a) for a in actions]


def _deserialize_action(data: Dict) -> Action:
    """Convert JSON action back to Action object."""
    at = ActionType[data["type"]]
    kwargs: Dict[str, Any] = {"action_type": at}

    if "vertex" in data:
        kwargs["vertex"] = data["vertex"]
    if "edge" in data:
        kwargs["edge"] = data["edge"]
    if "hexId" in data:
        kwargs["hex_id"] = data["hexId"]
    if "targetPlayer" in data:
        kwargs["target_player"] = data["targetPlayer"]
    if "resource" in data:
        kwargs["resource"] = Resource[data["resource"].upper()]
    if "resource2" in data:
        kwargs["resource2"] = Resource[data["resource2"].upper()]
    if "giveResource" in data:
        kwargs["give_resource"] = Resource[data["giveResource"].upper()]
    if "getResource" in data:
        kwargs["get_resource"] = Resource[data["getResource"].upper()]
    if "discard" in data:
        kwargs["discard"] = {Resource[k.upper()]: v for k, v in data["discard"].items()}

    return Action(**kwargs)


# -----------------------------------------------------------------------
# WebSocket handler
# -----------------------------------------------------------------------

@app.websocket("/ws/game")
async def game_websocket(websocket: WebSocket):
    await websocket.accept()
    session: Optional[GameSession] = None

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "new_game":
                difficulty = data.get("difficulty", "medium")
                seed = data.get("seed")
                human_player = data.get("humanPlayer", 0)
                session = GameSession(
                    human_player=human_player,
                    ai_difficulty=difficulty,
                    seed=seed,
                )
                # Run AI turns if human isn't first in setup
                ai_actions = session.process_ai_turns()
                actions = get_legal_actions(session.gs)
                await websocket.send_json({
                    "type": "game_state",
                    "state": _serialize_game_state(session.gs, human_player),
                    "legalActions": _serialize_legal_actions(actions),
                    "aiActions": ai_actions,
                })

            elif msg_type == "action" and session is not None:
                action = _deserialize_action(data["action"])
                apply_action(session.gs, action)

                # Run AI turns
                ai_actions = session.process_ai_turns()

                # Handle discard phase - if human needs to discard
                actions = get_legal_actions(session.gs)

                await websocket.send_json({
                    "type": "game_state",
                    "state": _serialize_game_state(session.gs, session.human_player),
                    "legalActions": _serialize_legal_actions(actions),
                    "aiActions": ai_actions,
                })

            elif msg_type == "get_state" and session is not None:
                actions = get_legal_actions(session.gs)
                await websocket.send_json({
                    "type": "game_state",
                    "state": _serialize_game_state(session.gs, session.human_player),
                    "legalActions": _serialize_legal_actions(actions),
                })

    except WebSocketDisconnect:
        pass


# -----------------------------------------------------------------------
# REST endpoints for training info
# -----------------------------------------------------------------------

@app.get("/api/training/stats")
async def get_training_stats():
    """Return training epoch history if available."""
    stats_path = Path("training_logs/stats.json")
    if stats_path.exists():
        return json.loads(stats_path.read_text())
    return {"epochs": []}


@app.get("/api/checkpoints")
async def list_checkpoints():
    """List available model checkpoints."""
    cp_dir = Path("checkpoints")
    if not cp_dir.exists():
        return {"checkpoints": []}
    return {
        "checkpoints": [
            {"name": f.name, "epoch": int(f.stem.split("_")[-1])}
            for f in sorted(cp_dir.glob("*.pt"))
        ]
    }
