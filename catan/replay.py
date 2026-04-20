"""Game replay recording and playback.

Records complete game histories for analysis. Replays are compact:
just the seed + action list, which can be re-executed to reconstruct
full game state at any point.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from catan.constants import ActionType, GamePhase, Resource
from catan.game import Action, GameState, apply_action, get_legal_actions, new_game


@dataclass
class ReplayFrame:
    """A single action in a replay."""
    player: int
    action_type: str
    vertex: Optional[int] = None
    edge: Optional[int] = None
    hex_id: Optional[int] = None
    target_player: Optional[int] = None
    resource: Optional[str] = None
    resource2: Optional[str] = None
    give_resource: Optional[str] = None
    get_resource: Optional[str] = None
    discard: Optional[Dict[str, int]] = None
    # Snapshot fields captured after the action
    phase: str = ""
    turn_number: int = 0
    dice_roll: Optional[List[int]] = None
    vps: List[int] = field(default_factory=list)


@dataclass
class ReplayData:
    """Complete recorded game."""
    seed: int
    winner: Optional[int]
    num_turns: int
    final_vps: List[int]
    epoch: int = 0
    game_idx: int = 0
    timestamp: float = 0.0
    frames: List[ReplayFrame] = field(default_factory=list)


def action_to_frame(gs: GameState, player_idx: int, action: Action) -> ReplayFrame:
    """Convert a game action + post-action state into a replay frame."""
    frame = ReplayFrame(
        player=player_idx,
        action_type=action.action_type.name,
        vertex=action.vertex,
        edge=action.edge,
        hex_id=action.hex_id,
        target_player=action.target_player,
        phase=gs.phase.name,
        turn_number=gs.turn_number,
        dice_roll=list(gs.dice_roll) if gs.dice_roll else None,
        vps=[p.victory_points for p in gs.players],
    )
    if action.resource is not None:
        frame.resource = action.resource.name.lower()
    if action.resource2 is not None:
        frame.resource2 = action.resource2.name.lower()
    if action.give_resource is not None:
        frame.give_resource = action.give_resource.name.lower()
    if action.get_resource is not None:
        frame.get_resource = action.get_resource.name.lower()
    if action.discard is not None:
        frame.discard = {r.name.lower(): v for r, v in action.discard.items()}
    return frame


def frame_to_action(frame: ReplayFrame) -> Action:
    """Reconstruct an Action from a replay frame."""
    kwargs: Dict[str, Any] = {"action_type": ActionType[frame.action_type]}
    if frame.vertex is not None:
        kwargs["vertex"] = frame.vertex
    if frame.edge is not None:
        kwargs["edge"] = frame.edge
    if frame.hex_id is not None:
        kwargs["hex_id"] = frame.hex_id
    if frame.target_player is not None:
        kwargs["target_player"] = frame.target_player
    if frame.resource is not None:
        kwargs["resource"] = Resource[frame.resource.upper()]
    if frame.resource2 is not None:
        kwargs["resource2"] = Resource[frame.resource2.upper()]
    if frame.give_resource is not None:
        kwargs["give_resource"] = Resource[frame.give_resource.upper()]
    if frame.get_resource is not None:
        kwargs["get_resource"] = Resource[frame.get_resource.upper()]
    if frame.discard is not None:
        kwargs["discard"] = {Resource[k.upper()]: v for k, v in frame.discard.items()}
    return Action(**kwargs)


class GameRecorder:
    """Records a game in progress."""

    def __init__(self, seed: int, epoch: int = 0, game_idx: int = 0):
        self.replay = ReplayData(
            seed=seed,
            winner=None,
            num_turns=0,
            final_vps=[],
            epoch=epoch,
            game_idx=game_idx,
            timestamp=time.time(),
        )

    def record(self, gs: GameState, action: Action, player_idx: int) -> None:
        """Record an action and the resulting state."""
        frame = action_to_frame(gs, player_idx, action)
        self.replay.frames.append(frame)

    def finalize(self, gs: GameState) -> ReplayData:
        """Finalize the recording with game results."""
        self.replay.winner = gs.winner
        self.replay.num_turns = gs.turn_number
        self.replay.final_vps = [p.victory_points for p in gs.players]
        return self.replay


def save_replay(replay: ReplayData, directory: str = "replays") -> str:
    """Save a replay to disk. Returns the file path."""
    os.makedirs(directory, exist_ok=True)

    # Filename: epoch_game_winner_timestamp
    winner_str = f"p{replay.winner}" if replay.winner is not None else "draw"
    vp_str = "-".join(str(v) for v in replay.final_vps)
    filename = f"e{replay.epoch:04d}_g{replay.game_idx:03d}_{winner_str}_vp{vp_str}.json"
    path = os.path.join(directory, filename)

    data = {
        "seed": replay.seed,
        "winner": replay.winner,
        "numTurns": replay.num_turns,
        "finalVPs": replay.final_vps,
        "epoch": replay.epoch,
        "gameIdx": replay.game_idx,
        "timestamp": replay.timestamp,
        "frames": [
            {k: v for k, v in asdict(f).items() if v is not None}
            for f in replay.frames
        ],
    }

    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    return path


def load_replay(path: str) -> ReplayData:
    """Load a replay from disk."""
    with open(path) as f:
        data = json.load(f)

    frames = []
    for fd in data["frames"]:
        frames.append(ReplayFrame(
            player=fd["player"],
            action_type=fd["action_type"],
            vertex=fd.get("vertex"),
            edge=fd.get("edge"),
            hex_id=fd.get("hex_id"),
            target_player=fd.get("target_player"),
            resource=fd.get("resource"),
            resource2=fd.get("resource2"),
            give_resource=fd.get("give_resource"),
            get_resource=fd.get("get_resource"),
            discard=fd.get("discard"),
            phase=fd.get("phase", ""),
            turn_number=fd.get("turn_number", 0),
            dice_roll=fd.get("dice_roll"),
            vps=fd.get("vps", []),
        ))

    return ReplayData(
        seed=data["seed"],
        winner=data["winner"],
        num_turns=data["numTurns"],
        final_vps=data["finalVPs"],
        epoch=data.get("epoch", 0),
        game_idx=data.get("gameIdx", 0),
        timestamp=data.get("timestamp", 0),
        frames=frames,
    )


def list_replays(directory: str = "replays") -> List[Dict[str, Any]]:
    """List available replays with metadata (without loading frames)."""
    replays = []
    path = Path(directory)
    if not path.exists():
        return replays

    for f in sorted(path.glob("*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            replays.append({
                "filename": f.name,
                "seed": data["seed"],
                "winner": data["winner"],
                "numTurns": data["numTurns"],
                "finalVPs": data["finalVPs"],
                "epoch": data.get("epoch", 0),
                "gameIdx": data.get("gameIdx", 0),
                "numFrames": len(data.get("frames", [])),
            })
        except (json.JSONDecodeError, KeyError):
            continue

    return replays


def replay_to_state(replay: ReplayData, up_to_frame: int) -> GameState:
    """Reconstruct game state by replaying actions up to a given frame index."""
    gs = new_game(seed=replay.seed)
    limit = min(up_to_frame, len(replay.frames))
    for i in range(limit):
        action = frame_to_action(replay.frames[i])
        apply_action(gs, action)
    return gs
