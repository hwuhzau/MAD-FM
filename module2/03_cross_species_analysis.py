"""Run Module 2 transfer, target-species fine-tuning and matched comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from workflow_utils import command_for, latest_checkpoint, run


TASKS = {
    "arabidopsis": "arabidopsis_TARGET",
    "brca": "homo_BRCA",
    "luad": "homo_LUAD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[*TASKS, "all"], default="all")
    parser.add_argument("--mode", choices=["transfer", "finetune", "all"], default="all")
    parser.add_argument(
        "--data-root", type=Path, required=True,
        help="root containing <task>/training and <task>/ranking",
    )
    parser.add_argument("--maize-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-repeats", type=int, default=100)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def train_task(args, task: str, task_data: Path, task_output: Path) -> Path | None:
    training = task_data / "training"
    model_root = task_output / "model"
    command = command_for(
        "training_core.py",
        "--pairs", training / "pairs_train_v2.jsonl",
        "--genes", training / "genes.parquet",
        "--protein-embeddings", training / "protein_emb_esm2_480.npy",
        "--out", model_root,
        "--run-name", f"{task}_self_finetuned",
        "--sources", "go", "coexp", "hic", "string", "tf",
        "--epochs", args.epochs,
        "--batch-size", args.batch_size,
        "--lr", "0.0002",
        "--negatives-per-anchor", 49,
        "--evaluation-negatives-per-anchor", 49,
        "--patience", 5,
        "--seed", 42,
        "--split-seed", 42,
    )
    run(command, args.dry_run)
    return None if args.dry_run else latest_checkpoint(model_root)


def run_transfer(args, task_data: Path, task_output: Path) -> None:
    command = command_for(
        "ranking_grid.py",
        "--model", args.maize_model.resolve(),
        "--data-dir", task_data / "ranking",
        "--output-root", task_output / "ranking_results",
        "--experiment-name", "ranking_hyperparameter_grid_v3",
        "--seed-ratios", "0.1", "0.3", "0.5", "0.7",
        "--similarity-top-ratios", "0.1", "0.3", "0.5", "0.7", "1.0",
        "--evaluation-ratios", "0.01", "0.05", "0.10",
        "--num-repeats", args.num_repeats,
        "--max-parallel", args.max_parallel,
        "--allow-nonfull-data",
        "--no-run-plots",
        *([] if args.device is None else ["--device", args.device]),
    )
    run(command, args.dry_run)


def compare(args, task: str, task_data: Path, task_output: Path, self_model: Path) -> None:
    command = command_for(
        "species_comparison.py",
        "--task", task,
        "--task-dir", task_output,
        "--data-dir", task_data / "ranking",
        "--self-model", self_model,
        "--maize-model", args.maize_model.resolve(),
        "--source-experiment", "ranking_hyperparameter_grid_v3",
        "--output-dir", task_output / "four_strategy_comparison",
        "--num-repeats", args.num_repeats,
        *([] if args.device is None else ["--device", args.device]),
    )
    run(command, args.dry_run)


def main() -> None:
    args = parse_args()
    selected = list(TASKS.values()) if args.task == "all" else [TASKS[args.task]]
    for task in selected:
        task_data = args.data_root.resolve() / task
        task_output = args.output_dir.resolve() / task
        print(f"\n===== {task} =====", flush=True)
        self_model = None
        if args.mode in {"finetune", "all"}:
            self_model = train_task(args, task, task_data, task_output)
        if args.mode in {"transfer", "all"}:
            run_transfer(args, task_data, task_output)
        if args.mode == "all" and not args.dry_run:
            assert self_model is not None
            compare(args, task, task_data, task_output, self_model)


if __name__ == "__main__":
    main()

