#!/usr/bin/env python3
"""Train AI agents via self-play PPO.

Usage:
    python scripts/train.py                     # defaults
    python scripts/train.py --epochs 500        # more training
    python scripts/train.py --resume latest     # resume from checkpoint
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.trainer import Trainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Train Catan AI via self-play")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--games-per-epoch", type=int, default=50, help="Games per epoch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=512, help="Network hidden layer size")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default="runs", help="TensorBoard log directory")
    parser.add_argument("--checkpoint-interval", type=int, default=20, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint (path or 'latest')")
    args = parser.parse_args()

    config = TrainingConfig(
        num_epochs=args.epochs,
        games_per_epoch=args.games_per_epoch,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        checkpoint_interval=args.checkpoint_interval,
    )

    trainer = Trainer(config)

    if args.resume:
        if args.resume == "latest":
            cp_dir = Path(args.checkpoint_dir)
            checkpoints = sorted(cp_dir.glob("*.pt"))
            if not checkpoints:
                print("No checkpoints found to resume from.")
                sys.exit(1)
            resume_path = str(checkpoints[-1])
        else:
            resume_path = args.resume
        epoch = trainer.load_checkpoint(resume_path)
        print(f"Resumed from epoch {epoch}")

    trainer.train()

    # Save stats for the web dashboard
    stats_dir = Path("training_logs")
    stats_dir.mkdir(exist_ok=True)
    stats_data = {
        "epochs": [
            {
                "epoch": s.epoch,
                "avg_game_length": s.avg_game_length,
                "avg_reward": s.avg_reward,
                "policy_loss": s.policy_loss,
                "value_loss": s.value_loss,
                "entropy": s.entropy,
                "wins": s.wins,
                "avg_settlements": s.avg_settlements,
                "avg_cities": s.avg_cities,
                "avg_roads": s.avg_roads,
                "avg_dev_cards": s.avg_dev_cards,
                "avg_knights": s.avg_knights,
                "longest_road_wins": s.longest_road_wins,
                "largest_army_wins": s.largest_army_wins,
            }
            for s in trainer.epoch_history
        ]
    }
    (stats_dir / "stats.json").write_text(json.dumps(stats_data, indent=2))
    print(f"\nTraining stats saved to {stats_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
