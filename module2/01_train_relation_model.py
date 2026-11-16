"""Train the MAD-FM Module 2 five-relation projection model."""

from __future__ import annotations

import argparse
from pathlib import Path

from workflow_utils import command_for, latest_checkpoint, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--genes", type=Path, required=True)
    parser.add_argument("--protein-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--negatives", type=int, default=49)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = command_for(
        "training_core.py",
        "--pairs", args.pairs.resolve(),
        "--genes", args.genes.resolve(),
        "--protein-embeddings", args.protein_embeddings.resolve(),
        "--out", args.output_dir.resolve(),
        "--run-name", "madfm_module2_five_relation",
        "--sources", "go", "coexp", "hic", "string", "tf",
        "--epochs", args.epochs,
        "--batch-size", args.batch_size,
        "--lr", args.learning_rate,
        "--negatives-per-anchor", args.negatives,
        "--evaluation-negatives-per-anchor", args.negatives,
        "--patience", args.patience,
        "--seed", args.seed,
        "--split-seed", args.seed,
    )
    run(command, args.dry_run)
    if not args.dry_run:
        print(f"Selected checkpoint: {latest_checkpoint(args.output_dir.resolve())}")


if __name__ == "__main__":
    main()

