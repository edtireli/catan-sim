# Catan Sim

A full-featured Catan board game simulator with reinforcement learning AI training, strategy visualization, and a playable web interface.

## Features

- **Complete Game Engine** — All standard Catan rules: resource production, building, trading, development cards, robber, longest road, largest army, harbors
- **AI Training (PPO)** — Self-play reinforcement learning using Proximal Policy Optimization
- **Training Visualization** — Track win rates, strategy evolution, and reward curves across training epochs
- **Playable Web UI** — Play against trained AI bots at varying difficulty levels
- **Strategy Analysis** — Examine what strategies the AI discovers and exploits

## Architecture

```
catan/          # Game engine — board, rules, game state
ai/             # RL training — PPO agent, neural network, self-play
server/         # FastAPI backend with WebSocket for real-time play
web/            # React + TypeScript frontend
scripts/        # Training and utility scripts
```

## Quick Start

### Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cd web && npm install && cd ..
```

### Train AI

```bash
python scripts/train.py --epochs 500 --games-per-epoch 100
```

Training progress is logged to TensorBoard:

```bash
tensorboard --logdir runs/
```

### Play Against AI

```bash
# Start the server
python scripts/serve.py

# Open http://localhost:8000 in your browser
```

### Run Tests

```bash
pytest tests/
```

## Game Rules Implemented

- Board generation with randomized tiles and numbers
- Resource production from dice rolls
- Building: roads, settlements, cities
- Development cards: knight, victory point, road building, year of plenty, monopoly
- Trading: 4:1 bank, 3:1 / 2:1 harbors, player-to-player
- Robber: activated on 7, discard excess cards, move and steal
- Longest road (5+) and largest army (3+ knights)
- Victory at 10 points

## AI Approach

The AI uses **Proximal Policy Optimization (PPO)** with self-play:

1. **State encoding**: Board topology, resource counts, building positions, development cards, game phase
2. **Action masking**: Only legal actions are considered at each step
3. **Self-play**: 4 AI agents play against each other, learning from wins and losses
4. **Curriculum**: Difficulty ramps as training progresses

The trained policy network outputs action probabilities over all legal moves, while a value head estimates the current win probability.

## License

MIT
