#!/usr/bin/env python3
"""Analyze the strategy learned by a trained AI.

Usage:
    python scripts/analyze.py                        # analyze latest checkpoint
    python scripts/analyze.py --checkpoint path.pt   # specific checkpoint
    python scripts/analyze.py --games 500            # more games for accuracy
    python scripts/analyze.py --compare              # compare all checkpoints
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.strategies import analyze_strategy, compare_checkpoints, print_strategy_report


def main():
    parser = argparse.ArgumentParser(description="Analyze AI strategy")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--games", type=int, default=200, help="Number of evaluation games")
    parser.add_argument("--compare", action="store_true", help="Compare all checkpoints")
    args = parser.parse_args()

    cp_dir = Path("checkpoints")
    checkpoints = sorted(cp_dir.glob("*.pt"))
    if not checkpoints:
        print("No checkpoints found. Train first with: python scripts/train.py")
        sys.exit(1)

    if args.compare:
        print(f"Comparing {len(checkpoints)} checkpoints with {args.games} games each...\n")
        compare_checkpoints([str(c) for c in checkpoints], num_games=args.games)
    else:
        if args.checkpoint is None:
            args.checkpoint = str(checkpoints[-1])

        print(f"Analyzing checkpoint: {args.checkpoint}")
        print(f"Running {args.games} evaluation games...\n")

        profile = analyze_strategy(args.checkpoint, num_games=args.games)
        print_strategy_report(profile)


if __name__ == "__main__":
    main()
