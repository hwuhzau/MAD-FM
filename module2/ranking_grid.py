"""V3 five-head percentage-depth sensitivity analysis for candidate-gene ranking.

The trained model is loaded and projected only once.  CPU workers then evaluate
seed-ratio and similarity-aggregation combinations from a shared similarity
context. Ranking quality is measured at Top 1%, 5%, and 10% of the genes
actually ranked. Completed combinations are detected and skipped by default.
Figures use Arial 10-point text, the approved five-color palette, and PNG,
PDF, and SVG exports. Recall figures use capacity-normalized Recall@K.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import itertools
import json
import math
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np
import pandas as pd
import torch
from scipy.stats import friedmanchisquare, wilcoxon

import ranking_core
from ranking_core import (
    DEFAULT_DATA_DIR,
    EVALUATION_PROTOCOL_VERSION,
    EvaluationParameters,
    RankingContext,
    atomic_write_csv,
    atomic_write_json,
    benjamini_hochberg,
    execute_evaluation,
    prepare_ranking_context,
    random_ranking_baseline,
    resolve_requested_model,
    run_identifier,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_ROOT = SCRIPT_DIR / "results" / "experiments"


# =============================================================================
# USER-EDITABLE HYPERPARAMETERS
# Command-line arguments override these values when supplied.
# =============================================================================
GRID_MODEL_RUN = (
    "seed42_20261005_101049_759729_shared_residual_dim512_bs32_neg49_"
    "lr0.0002_src-go-coexp-hic-string-tf_v3"
)
GRID_SEED_RATIOS = [0.10, 0.30, 0.50, 0.70]
# Spans local evidence through the all-seed mean.  Including 1.00 provides a
# meaningful endpoint and reveals whether performance plateaus or turns over.
GRID_SIMILARITY_TOP_RATIOS = [0.10, 0.30, 0.50, 0.70, 1.00]
GRID_NUM_REPEATS = 100
GRID_SPLIT_SEED = 42
GRID_RRF_K = 60.0
GRID_EVALUATION_RATIOS = [0.01, 0.05, 0.10]
GRID_PRIMARY_EVALUATION_RATIO = 0.01
GRID_SELECTION_METRIC_FAMILY = "ndcg"
GRID_MAX_PARALLEL = 4
GRID_PROJECTION_BATCH_SIZE = 1024
GRID_EXPERIMENT_NAME = "ranking_hyperparameter_grid5_v3"
GRID_SAVE_RUN_PLOTS = False
PLOT_FONT_FAMILY = (
    "Arial" if Path("C:/Windows/Fonts/arial.ttf").exists() else "DejaVu Sans"
)
PLOT_FONT_SIZE_PT = 10
PLOT_DPI = 300
# User-approved V3 palette used consistently across all data series.
PALE_PINK_RED = "#E8A4A4"
SKY_BLUE = "#87CEEB"
BLUE_VIOLET = "#5C6B9E"
PALE_PINK = "#E6B8B8"
PALE_CYAN = "#A8D5E5"
MODEL_COLOR = BLUE_VIOLET
RANDOM_COLOR = PALE_PINK
RANDOM_EDGE_COLOR = "#6B6B6B"
ACCENT_COLOR = PALE_PINK_RED
GRID_COLOR = "#D9D9D9"
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "v3_approved_palette",
    [PALE_PINK, PALE_PINK_RED, PALE_CYAN, SKY_BLUE, BLUE_VIOLET],
)
# =============================================================================


def configure_plot_style() -> None:
    """Apply one readable, publication-ready style to every v3 figure."""
    plt.rcParams.update(
        {
            "font.family": PLOT_FONT_FAMILY,
            "font.sans-serif": [PLOT_FONT_FAMILY],
            "font.size": PLOT_FONT_SIZE_PT,
            "axes.titlesize": PLOT_FONT_SIZE_PT,
            "axes.labelsize": PLOT_FONT_SIZE_PT,
            "xtick.labelsize": PLOT_FONT_SIZE_PT,
            "ytick.labelsize": PLOT_FONT_SIZE_PT,
            "legend.fontsize": PLOT_FONT_SIZE_PT,
            "figure.titlesize": PLOT_FONT_SIZE_PT,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    """Save matching raster and true-vector versions of one figure."""
    base_path = Path(base_path)
    for text_artist in fig.findobj(match=Text):
        text_artist.set_fontfamily(PLOT_FONT_FAMILY)
        text_artist.set_fontsize(PLOT_FONT_SIZE_PT)
    fig.savefig(
        base_path.with_suffix(".png"),
        dpi=PLOT_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        base_path.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        base_path.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


ranking_core.configure_plot_style = configure_plot_style
ranking_core.save_figure = save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch ranking evaluation across hyperparameter combinations"
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", type=Path)
    model_group.add_argument(
        "--model-run",
        help=(
            "run-directory name under models/protein_multihead/all_output; "
            f"source default: {GRID_MODEL_RUN!r}; use 'auto' for newest"
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--heads", nargs="+")
    parser.add_argument(
        "--seed-ratios",
        type=float,
        nargs="+",
        default=list(GRID_SEED_RATIOS),
    )
    parser.add_argument(
        "--similarity-top-ratios",
        type=float,
        nargs="+",
        default=list(GRID_SIMILARITY_TOP_RATIOS),
    )
    parser.add_argument("--num-repeats", type=int, default=GRID_NUM_REPEATS)
    parser.add_argument("--split-seed", type=int, default=GRID_SPLIT_SEED)
    parser.add_argument("--rrf-k", type=float, default=GRID_RRF_K)
    parser.add_argument(
        "--evaluation-ratios",
        type=float,
        nargs="+",
        default=list(GRID_EVALUATION_RATIOS),
        help="fractions of the actually ranked gene universe (default: 0.01 0.05 0.10)",
    )
    parser.add_argument(
        "--primary-evaluation-ratio",
        type=float,
        default=GRID_PRIMARY_EVALUATION_RATIO,
        help="headline screening depth used for configuration selection (default: 0.01)",
    )
    parser.add_argument(
        "--selection-metric-family",
        choices=["ndcg", "recall", "average_precision"],
        default=GRID_SELECTION_METRIC_FAMILY,
        help="metric family used at the primary screening depth (default: ndcg)",
    )
    parser.add_argument("--max-parallel", type=int, default=GRID_MAX_PARALLEL)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--projection-batch-size", type=int, default=GRID_PROJECTION_BATCH_SIZE
    )
    parser.add_argument(
        "--experiment-name", default=GRID_EXPERIMENT_NAME
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--allow-nonfull-data", action="store_true")
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="run completed configurations again and overwrite their result files",
    )
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--save-run-plots",
        dest="save_run_plots",
        action="store_true",
        help="also save seven diagnostic figures inside every configuration",
    )
    plot_group.add_argument(
        "--no-run-plots",
        dest="save_run_plots",
        action="store_false",
        help="save only concise experiment-level figures (default)",
    )
    parser.set_defaults(save_run_plots=GRID_SAVE_RUN_PLOTS)
    return parser.parse_args()


def percentage_token(ratio: float) -> str:
    percent = ratio * 100.0
    text = f"{percent:.6g}".replace(".", "p")
    return f"top_{text}pct"


def percentage_label(ratio: float) -> str:
    return f"Top {ratio * 100:g}%"


def percentage_metric(metric_family: str, ratio: float) -> str:
    return f"{metric_family}_at_{percentage_token(ratio)}"


def raw_metric(metric_family: str, k: int) -> str:
    return f"{metric_family}_at_{int(k)}"


CAPACITY_RECALL_FAMILY = "capacity_normalized_recall"


def capacity_normalized_recall(
    hits: float,
    actual_k: int,
    hidden_count: int,
) -> float:
    """Return Hits@K divided by the maximum attainable hits at this K."""
    denominator = min(int(actual_k), int(hidden_count))
    if denominator <= 0:
        return float("nan")
    return float(np.clip(float(hits) / denominator, 0.0, 1.0))


def capacity_recall_scale(actual_k: int, hidden_count: int) -> float:
    """Scale standard Recall@K into capacity-normalized Recall@K."""
    denominator = min(int(actual_k), int(hidden_count))
    if denominator <= 0:
        return float("nan")
    return float(hidden_count) / denominator


def random_metric_expectation(
    metric_family: str,
    actual_k: int,
    ranking_count: int,
    hidden_count: int,
) -> float:
    """Return the exact uniform-random expectation for a plotted metric."""
    if metric_family == CAPACITY_RECALL_FAMILY:
        expected_hits = actual_k * hidden_count / ranking_count
        return capacity_normalized_recall(
            expected_hits,
            actual_k,
            hidden_count,
        )
    return float(
        random_ranking_baseline(
            raw_metric(metric_family, actual_k),
            ranking_count,
            hidden_count,
        )
    )


def cutoff_k(ranking_count: int, ratio: float) -> int:
    return max(1, min(ranking_count, int(math.ceil(ranking_count * ratio))))


def cutoff_table(
    context: RankingContext,
    seed_ratios: list[float],
    evaluation_ratios: list[float],
) -> pd.DataFrame:
    rows: list[dict] = []
    known_count = len(context.known_ids)
    combined_count = len(context.all_ids)
    for seed_ratio in sorted(set(seed_ratios)):
        seed_count = int(math.floor(known_count * seed_ratio + 0.5))
        hidden_count = known_count - seed_count
        ranking_count = combined_count - seed_count
        for evaluation_ratio in sorted(set(evaluation_ratios)):
            k = cutoff_k(ranking_count, evaluation_ratio)
            rows.append(
                {
                    "seed_ratio": float(seed_ratio),
                    "seed_count": seed_count,
                    "hidden_count": hidden_count,
                    "ranking_count": ranking_count,
                    "evaluation_ratio": float(evaluation_ratio),
                    "evaluation_label": percentage_label(evaluation_ratio),
                    "actual_k": k,
                    "actual_fraction": k / ranking_count,
                    "random_expected_hits": k * hidden_count / ranking_count,
                }
            )
    return pd.DataFrame(rows)


def cutoff_map_for_seed(cutoffs: pd.DataFrame, seed_ratio: float) -> dict[float, int]:
    frame = cutoffs[np.isclose(cutoffs.seed_ratio, seed_ratio)]
    return {
        float(row.evaluation_ratio): int(row.actual_k)
        for _, row in frame.iterrows()
    }


def canonicalize_run_metrics(
    row: dict,
    seed_ratio: float,
    cutoffs: pd.DataFrame,
) -> None:
    mapping = cutoff_map_for_seed(cutoffs, seed_ratio)
    for ratio, k in mapping.items():
        for family in ["recall", "average_precision", "ndcg", "enrichment"]:
            raw_prefix = f"rrf_{raw_metric(family, k)}"
            canonical_prefix = f"rrf_{percentage_metric(family, ratio)}"
            for suffix in ["mean", "std", "ci95_low", "ci95_high"]:
                raw_column = f"{raw_prefix}_{suffix}"
                if raw_column in row:
                    row[f"{canonical_prefix}_{suffix}"] = row[raw_column]
        cutoff = cutoffs[
            np.isclose(cutoffs.seed_ratio, seed_ratio)
            & np.isclose(cutoffs.evaluation_ratio, ratio)
        ].iloc[0]
        scale = capacity_recall_scale(k, int(cutoff.hidden_count))
        recall_prefix = f"rrf_{percentage_metric('recall', ratio)}"
        capacity_prefix = (
            f"rrf_{percentage_metric(CAPACITY_RECALL_FAMILY, ratio)}"
        )
        for suffix in ["mean", "std", "ci95_low", "ci95_high"]:
            source_column = f"{recall_prefix}_{suffix}"
            if source_column in row:
                row[f"{capacity_prefix}_{suffix}"] = float(
                    np.clip(row[source_column] * scale, 0.0, 1.0)
                )


def canonicalize_repeat_metrics(
    frame: pd.DataFrame,
    seed_ratio: float,
    cutoffs: pd.DataFrame,
) -> pd.DataFrame:
    mapping = cutoff_map_for_seed(cutoffs, seed_ratio)
    for ratio, k in mapping.items():
        for family in ["recall", "average_precision", "ndcg", "enrichment"]:
            source = raw_metric(family, k)
            if source in frame.columns:
                frame[percentage_metric(family, ratio)] = frame[source]
        cutoff = cutoffs[
            np.isclose(cutoffs.seed_ratio, seed_ratio)
            & np.isclose(cutoffs.evaluation_ratio, ratio)
        ].iloc[0]
        scale = capacity_recall_scale(k, int(cutoff.hidden_count))
        frame[percentage_metric(CAPACITY_RECALL_FAMILY, ratio)] = np.clip(
            frame[percentage_metric("recall", ratio)] * scale,
            0.0,
            1.0,
        )
    frame["normalized_mean_rank"] = frame["mean_rank"] / frame["ranking_count"]
    frame["normalized_median_rank"] = frame["median_rank"] / frame["ranking_count"]
    return frame


def parameter_key(parameters: EvaluationParameters) -> dict:
    payload = asdict(parameters)
    payload["ks"] = list(parameters.ks)
    return payload


def is_completed(
    run_dir: Path,
    parameters: EvaluationParameters,
    model_sha256: str,
    active_heads: list[str],
) -> bool:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        summary.get("status") == "complete"
        and summary.get("evaluation_protocol_version") == EVALUATION_PROTOCOL_VERSION
        and summary.get("parameters") == parameter_key(parameters)
        and summary.get("model", {}).get("sha256") == model_sha256
        and summary.get("aggregation", {}).get("active_heads") == active_heads
    )


def validate_experiment_identity(
    config_path: Path, context: RankingContext, experiment_name: str
) -> None:
    """Prevent one experiment directory from mixing models or head sets."""
    if not config_path.exists():
        return
    try:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot validate existing experiment metadata: {config_path}"
        ) from exc
    conflicts: list[str] = []
    if existing.get("model_sha256") != context.model_sha256:
        conflicts.append(
            f"model SHA: {existing.get('model_sha256')} -> {context.model_sha256}"
        )
    if existing.get("active_heads") != context.active_heads:
        conflicts.append(
            f"active heads: {existing.get('active_heads')} -> {context.active_heads}"
        )
    existing_data = existing.get("data_directory")
    if existing_data and Path(existing_data).resolve() != Path(context.data_dir).resolve():
        conflicts.append(f"data directory: {existing_data} -> {context.data_dir}")
    if conflicts:
        raise RuntimeError(
            f"experiment {experiment_name!r} already belongs to a different protocol:\n"
            + "\n".join(f"  - {item}" for item in conflicts)
            + "\nUse a new --experiment-name so results are not overwritten or mixed."
        )


def worker_run(
    context: RankingContext,
    parameters: EvaluationParameters,
    output_dir: str,
    generate_plots: bool,
) -> dict:
    return execute_evaluation(
        context=context,
        parameters=parameters,
        output_dir=Path(output_dir),
        overwrite=True,
        generate_plots=generate_plots,
    )


def metric_direction(metric: str) -> str:
    return "min" if metric in {"mean_rank", "median_rank"} else "max"


def collect_experiment_results(
    experiment_dir: Path,
    configurations: list[EvaluationParameters],
    context: RankingContext,
    cutoffs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict] = []
    metric_frames: list[pd.DataFrame] = []
    for parameters in configurations:
        run_dir = experiment_dir / "runs" / run_identifier(parameters)
        summary_path = run_dir / "run_summary.json"
        metric_path = run_dir / "tables" / "metrics_summary_long.csv"
        if not summary_path.exists() or not metric_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            continue
        metrics = pd.read_csv(metric_path)
        metrics.insert(0, "seed_ratio", parameters.seed_ratio)
        metrics.insert(1, "similarity_top_ratio", parameters.similarity_top_ratio)
        metrics.insert(2, "run_id", run_identifier(parameters))
        metric_frames.append(metrics)
        row = {
            "run_id": run_identifier(parameters),
            "seed_ratio": parameters.seed_ratio,
            "similarity_top_ratio": parameters.similarity_top_ratio,
            "num_repeats": parameters.num_repeats,
            "split_seed": parameters.split_seed,
            "rrf_k": parameters.rrf_k,
            "active_heads": ",".join(context.active_heads),
            "head_count": len(context.active_heads),
            "seed_count": summary["data"]["seed_count_per_repeat"],
            "hidden_count": summary["data"]["hidden_count_per_repeat"],
            "ranking_count": summary["data"]["ranking_count_per_repeat"],
            "similarity_top_count": summary["aggregation"][
                "evaluation_top_seed_count"
            ],
            "runtime_seconds": summary["runtime_seconds"],
            "run_directory": str(run_dir),
        }
        for metric, values in summary.get("rrf_metrics", {}).items():
            row[f"rrf_{metric}_mean"] = values["mean"]
            row[f"rrf_{metric}_std"] = values["std"]
            row[f"rrf_{metric}_ci95_low"] = values["ci95_low"]
            row[f"rrf_{metric}_ci95_high"] = values["ci95_high"]
        canonicalize_run_metrics(row, parameters.seed_ratio, cutoffs)
        row["rrf_normalized_mean_rank_mean"] = (
            row["rrf_mean_rank_mean"] / row["ranking_count"]
        )
        row["rrf_normalized_median_rank_mean"] = (
            row["rrf_median_rank_mean"] / row["ranking_count"]
        )
        primary_results = summary.get("statistical_assessment", {}).get(
            "rrf_primary_results", {}
        )
        for key, value in primary_results.items():
            if key not in {"method", "method_label"}:
                row[f"rrf_primary_{key}"] = value
        row["statistical_verdict"] = summary.get(
            "statistical_assessment", {}
        ).get("verdict")
        run_rows.append(row)
    runs = pd.DataFrame(run_rows)
    metrics = (
        pd.concat(metric_frames, ignore_index=True)
        if metric_frames
        else pd.DataFrame()
    )
    return runs, metrics


def load_rrf_repeat_results(
    experiment_dir: Path,
    configurations: list[EvaluationParameters],
    cutoffs: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for parameters in configurations:
        path = (
            experiment_dir
            / "runs"
            / run_identifier(parameters)
            / "tables"
            / "metrics_by_repeat.csv"
        )
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame[frame.method == "rrf"].copy()
        frame = canonicalize_repeat_metrics(
            frame, parameters.seed_ratio, cutoffs
        )
        frame.insert(0, "seed_ratio", parameters.seed_ratio)
        frame.insert(1, "similarity_top_ratio", parameters.similarity_top_ratio)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def select_best_by_seed(runs: pd.DataFrame, selection_metric: str) -> pd.DataFrame:
    column = f"rrf_{selection_metric}_mean"
    ascending = metric_direction(selection_metric) == "min"
    rows = []
    for seed_ratio, frame in runs.groupby("seed_ratio", sort=True):
        best = frame.sort_values(column, ascending=ascending).iloc[0].copy()
        rows.append(best)
    return pd.DataFrame(rows).reset_index(drop=True)


def summarize_top_ratio_robustness(
    runs: pd.DataFrame, selection_metric: str
) -> pd.DataFrame:
    column = f"rrf_{selection_metric}_mean"
    direction = metric_direction(selection_metric)
    best_by_seed = select_best_by_seed(runs, selection_metric)
    win_counts = best_by_seed.similarity_top_ratio.value_counts()
    rows: list[dict] = []
    for top_ratio, frame in runs.groupby("similarity_top_ratio", sort=True):
        values = frame[column].astype(float)
        rows.append(
            {
                "similarity_top_ratio": float(top_ratio),
                "mean_across_seed_ratios": float(values.mean()),
                "minimum_across_seed_ratios": float(values.min()),
                "maximum_across_seed_ratios": float(values.max()),
                "std_across_seed_ratios": float(values.std(ddof=1)),
                "seed_ratio_wins": int(win_counts.get(top_ratio, 0)),
                "selection_direction": direction,
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        "mean_across_seed_ratios", ascending=direction == "min"
    ).reset_index(drop=True)


def value_matrix(
    runs: pd.DataFrame, metric: str
) -> tuple[np.ndarray, list[float], list[float]]:
    column = f"rrf_{metric}_mean"
    if column not in runs.columns:
        raise KeyError(f"metric not available in grid summary: {metric}")
    seed_values = sorted(runs.seed_ratio.unique())
    top_values = sorted(runs.similarity_top_ratio.unique())
    matrix = np.full((len(seed_values), len(top_values)), np.nan, dtype=float)
    for i, seed_ratio in enumerate(seed_values):
        for j, top_ratio in enumerate(top_values):
            row = runs[
                np.isclose(runs.seed_ratio, seed_ratio)
                & np.isclose(runs.similarity_top_ratio, top_ratio)
            ]
            if not row.empty:
                matrix[i, j] = float(row.iloc[0][column])
    return matrix, seed_values, top_values


def metric_label(metric: str) -> str:
    for family, label in [
        (
            "capacity_normalized_recall_at_top_",
            "Capacity-normalized Recall@Top ",
        ),
        ("recall_at_top_", "Recall@Top "),
        ("average_precision_at_top_", "MAP@Top "),
        ("ndcg_at_top_", "NDCG@Top "),
        ("enrichment_at_top_", "Recall enrichment@Top "),
    ]:
        if metric.startswith(family) and metric.endswith("pct"):
            percent = metric[len(family) : -3].replace("p", ".")
            return f"{label}{percent}%"
    replacements = {
        "capacity_normalized_recall_at_": "Capacity-normalized Recall@",
        "recall_at_": "Recall@",
        "average_precision_at_": "MAP@",
        "ndcg_at_": "NDCG@",
        "enrichment_at_": "Enrichment@",
        "mean_rank": "Mean hidden-gene rank",
        "median_rank": "Median hidden-gene rank",
    }
    for prefix, label in replacements.items():
        if metric.startswith(prefix):
            return label + metric[len(prefix) :]
    return metric


def add_figure_note(fig: plt.Figure, note: str) -> None:
    """Add one self-contained protocol line below a saved scientific figure."""
    wrapped_note = "\n".join(
        textwrap.fill(
            line,
            width=105,
            break_long_words=False,
            break_on_hyphens=False,
        )
        for line in note.splitlines()
    )
    fig.text(
        0.5,
        -0.045,
        wrapped_note,
        ha="center",
        va="top",
        fontsize=PLOT_FONT_SIZE_PT,
        color="#4D4D4D",
    )


def primary_protocol_note(
    cutoffs: pd.DataFrame,
    seed_ratio: float,
    aggregation_text: str,
    repeats: int,
) -> str:
    row = cutoffs[np.isclose(cutoffs.seed_ratio, seed_ratio)].iloc[0]
    return (
        f"Ranked genes N={int(row.ranking_count):,} | seed genes={int(row.seed_count)} | "
        f"hidden genes={int(row.hidden_count)} | {aggregation_text} | repeated splits={repeats}"
    )


def compact_protocol_note(note: str) -> str:
    """Wrap the unchanged protocol fields for compact single-panel figures."""
    parts = note.split(" | ")
    if len(parts) < 4:
        return note
    split_at = int(math.ceil(len(parts) / 2))
    return " | ".join(parts[:split_at]) + "\n" + " | ".join(parts[split_at:])


def heatmap_protocol_note(
    cutoffs: pd.DataFrame,
    top_values: list[float],
    evaluation_ratio: float,
) -> str:
    ordered = cutoffs.sort_values(["seed_ratio", "evaluation_ratio"])
    ordered = ordered[np.isclose(ordered.evaluation_ratio, evaluation_ratio)]
    row_text = "; ".join(
        (
            f"{float(row.seed_ratio):g}→{int(row.seed_count)}/"
            f"{int(row.hidden_count)}/{int(row.ranking_count):,}/{int(row.actual_k)}"
        )
        for _, row in ordered.iterrows()
    )
    top_text = ", ".join(f"{value:g}" for value in top_values)
    return (
        "Rows seed ratio→seed/hidden/ranked N/K: "
        f"{row_text} | aggregation ratios={top_text}"
    )


def paired_percentage_tests(
    repeats: pd.DataFrame,
    seed_ratio: float,
    evaluation_ratios: list[float],
    metric_family: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeated-measures tests across similarity aggregation ratios."""
    frame = repeats[np.isclose(repeats.seed_ratio, seed_ratio)].copy()
    top_values = sorted(frame.similarity_top_ratio.unique())
    omnibus_rows: list[dict] = []
    pairwise_rows: list[dict] = []
    for evaluation_ratio in sorted(set(evaluation_ratios)):
        metric = percentage_metric(metric_family, evaluation_ratio)
        pivot = frame.pivot(index="repeat", columns="similarity_top_ratio", values=metric)
        pivot = pivot.dropna(subset=top_values)
        samples = [pivot[value].to_numpy(dtype=float) for value in top_values]
        if len(samples) >= 3 and len(pivot):
            statistic, p_value = friedmanchisquare(*samples)
        else:
            statistic, p_value = float("nan"), float("nan")
        omnibus_rows.append(
            {
                "seed_ratio": seed_ratio,
                "evaluation_ratio": evaluation_ratio,
                "evaluation_label": percentage_label(evaluation_ratio),
                "metric": metric,
                "repeats": len(pivot),
                "friedman_statistic": float(statistic),
                "friedman_p_value": float(p_value),
            }
        )
        for left, right in itertools.combinations(top_values, 2):
            difference = pivot[right].to_numpy() - pivot[left].to_numpy()
            if np.allclose(difference, 0.0):
                pair_p = 1.0
            else:
                pair_p = float(
                    wilcoxon(
                        pivot[right].to_numpy(),
                        pivot[left].to_numpy(),
                        alternative="two-sided",
                        zero_method="wilcox",
                        method="auto",
                    ).pvalue
                )
            pairwise_rows.append(
                {
                    "seed_ratio": seed_ratio,
                    "evaluation_ratio": evaluation_ratio,
                    "evaluation_label": percentage_label(evaluation_ratio),
                    "metric": metric,
                    "top_ratio_left": float(left),
                    "top_ratio_right": float(right),
                    "paired_repeats": len(pivot),
                    "left_mean": float(pivot[left].mean()),
                    "right_mean": float(pivot[right].mean()),
                    "right_minus_left_mean": float(difference.mean()),
                    "wilcoxon_two_sided_p_value": pair_p,
                }
            )
    omnibus = pd.DataFrame(omnibus_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    if not omnibus.empty:
        omnibus["fdr_q_value"] = benjamini_hochberg(omnibus.friedman_p_value)
    if not pairwise.empty:
        pairwise["fdr_q_value"] = benjamini_hochberg(
            pairwise.wilcoxon_two_sided_p_value
        )
        pairwise["significant_fdr_0p05"] = pairwise.fdr_q_value < 0.05
    return omnibus, pairwise


def simulate_uniform_random_repeats(
    model_repeats: pd.DataFrame,
    evaluation_ratios: list[float],
    random_seed: int = 20260902,
) -> pd.DataFrame:
    """Generate one reproducible matched uniform-random ranking per split."""
    rng = np.random.default_rng(random_seed)
    rows: list[dict] = []
    for _, source in model_repeats.sort_values("repeat").iterrows():
        ranking_count = int(source.ranking_count)
        hidden_count = int(source.hidden_count)
        relevant = np.zeros(ranking_count, dtype=bool)
        relevant[rng.choice(ranking_count, size=hidden_count, replace=False)] = True
        ranks = np.flatnonzero(relevant) + 1
        row = {
            "repeat": int(source["repeat"]),
            "seed_count": int(source.seed_count),
            "hidden_count": hidden_count,
            "ranking_count": ranking_count,
            "normalized_mean_rank": float(np.mean(ranks) / ranking_count),
            "normalized_median_rank": float(np.median(ranks) / ranking_count),
        }
        cumulative = np.cumsum(relevant)
        for ratio in evaluation_ratios:
            k = cutoff_k(ranking_count, ratio)
            hits = int(cumulative[k - 1])
            recall = hits / hidden_count
            hit_positions = np.flatnonzero(relevant[:k]) + 1
            if len(hit_positions):
                precision_at_hits = cumulative[hit_positions - 1] / hit_positions
                average_precision = float(
                    precision_at_hits.sum() / min(hidden_count, k)
                )
            else:
                average_precision = 0.0
            discounts = 1.0 / np.log2(np.arange(2, k + 2))
            dcg = float((relevant[:k] * discounts).sum())
            idcg = float(discounts[: min(hidden_count, k)].sum())
            ndcg = dcg / idcg if idcg else 0.0
            expected_hits = k * hidden_count / ranking_count
            row[percentage_metric("recall", ratio)] = recall
            row[percentage_metric(CAPACITY_RECALL_FAMILY, ratio)] = (
                capacity_normalized_recall(hits, k, hidden_count)
            )
            row[percentage_metric("average_precision", ratio)] = average_precision
            row[percentage_metric("ndcg", ratio)] = ndcg
            row[percentage_metric("enrichment", ratio)] = (
                hits / expected_hits if expected_hits else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def significance_stars(q_value: float) -> str:
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return "ns"


def plot_metric_depth_profile(
    model_repeats: pd.DataFrame,
    random_repeats: pd.DataFrame,
    cutoffs: pd.DataFrame,
    seed_ratio: float,
    evaluation_ratios: list[float],
    metric_family: str,
    figure_note: str,
    output_base: Path,
) -> None:
    """Grouped model-versus-random boxplots across screening depths."""
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.13, top=0.80)
    centers = np.arange(1, len(evaluation_ratios) + 1, dtype=float)
    model_positions = centers - 0.18
    random_positions = centers + 0.18
    model_values: list[np.ndarray] = []
    random_values: list[np.ndarray] = []
    p_values: list[float] = []
    cutoff_frame = cutoffs[np.isclose(cutoffs.seed_ratio, seed_ratio)].set_index(
        "evaluation_ratio"
    )
    for ratio in evaluation_ratios:
        metric = percentage_metric(metric_family, ratio)
        observed_metric = model_repeats[metric].to_numpy(dtype=float)
        random_metric = random_repeats[metric].to_numpy(dtype=float)
        cutoff = cutoff_frame.loc[ratio]
        baseline = random_metric_expectation(
            metric_family,
            int(cutoff.actual_k),
            int(cutoff.ranking_count),
            int(cutoff.hidden_count),
        )
        if metric_family == CAPACITY_RECALL_FAMILY:
            test_observed = model_repeats[
                percentage_metric("recall", ratio)
            ].to_numpy(dtype=float)
            test_baseline = random_metric_expectation(
                "recall",
                int(cutoff.actual_k),
                int(cutoff.ranking_count),
                int(cutoff.hidden_count),
            )
            favorable = test_observed - test_baseline
        else:
            favorable = observed_metric - baseline
        p_values.append(
            1.0
            if np.allclose(favorable, 0.0)
            else float(
                wilcoxon(
                    favorable,
                    alternative="greater",
                    zero_method="wilcox",
                    method="auto",
                ).pvalue
            )
        )
        if metric_family == CAPACITY_RECALL_FAMILY:
            model_values.append(observed_metric * 100.0)
            random_values.append(random_metric * 100.0)
        else:
            model_values.append(observed_metric)
            random_values.append(random_metric)
    q_values = benjamini_hochberg(p_values)

    common = {
        "widths": 0.30,
        "patch_artist": True,
        "showfliers": False,
        "medianprops": {"color": "#1A1A1A", "linewidth": 1.8},
        "whiskerprops": {"color": "#555555", "linewidth": 1.0},
        "capprops": {"color": "#555555", "linewidth": 1.0},
    }
    model_boxes = ax.boxplot(
        model_values,
        positions=model_positions,
        boxprops={"edgecolor": MODEL_COLOR, "linewidth": 1.2},
        **common,
    )
    random_boxes = ax.boxplot(
        random_values,
        positions=random_positions,
        boxprops={"edgecolor": RANDOM_EDGE_COLOR, "linewidth": 1.2},
        **common,
    )
    for patch in model_boxes["boxes"]:
        patch.set_facecolor(MODEL_COLOR)
        patch.set_alpha(0.78)
    for patch in random_boxes["boxes"]:
        patch.set_facecolor(RANDOM_COLOR)
        patch.set_alpha(0.82)

    all_values = np.concatenate(model_values + random_values)
    if metric_family == CAPACITY_RECALL_FAMILY:
        lower = 0.0
        upper = 100.0
        span = 100.0
    else:
        lower = min(0.0, float(np.min(all_values)))
        upper = float(np.max(all_values))
        span = max(upper - lower, 0.10)
    for center, observed, random_sample, q_value in zip(
        centers, model_values, random_values, q_values
    ):
        sample_maximum = max(
            float(np.max(observed)),
            float(np.max(random_sample)),
        )
        if metric_family == CAPACITY_RECALL_FAMILY:
            bracket_y = min(sample_maximum + 3.0, 93.0)
            bracket_h = 1.5
        else:
            bracket_y = sample_maximum + 0.055 * span
            bracket_h = 0.018 * span
        ax.plot(
            [center - 0.18, center - 0.18, center + 0.18, center + 0.18],
            [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y],
            color="#333333",
            linewidth=0.9,
        )
        ax.text(
            center,
            bracket_y + 1.4 * bracket_h,
            f"{significance_stars(float(q_value))}  q={float(q_value):.2g}",
            ha="center",
            va="bottom",
            fontsize=PLOT_FONT_SIZE_PT,
        )

    if metric_family == CAPACITY_RECALL_FAMILY:
        display_family = "Capacity-normalized Recall"
        y_label = "Capacity-normalized Recall (%)"
    else:
        display_family = (
            "MAP" if metric_family == "average_precision" else metric_family.upper()
        )
        y_label = display_family
    tick_labels = [
        f"Top {ratio * 100:g}%\n(K={int(cutoff_frame.loc[ratio, 'actual_k'])})"
        for ratio in evaluation_ratios
    ]
    ax.set_xticks(centers, tick_labels)
    ax.set_xlabel("Screening depth (% of genes actually ranked)")
    ax.set_ylabel(y_label)
    fig.suptitle(
        f"RRF {display_family} versus uniform-random ranking",
        y=0.965,
        fontsize=PLOT_FONT_SIZE_PT,
        weight="semibold",
    )
    if metric_family == CAPACITY_RECALL_FAMILY:
        ax.set_ylim(0.0, 100.0)
    else:
        ax.set_ylim(lower, upper + 0.20 * span)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(
        handles=[
            Patch(facecolor=MODEL_COLOR, edgecolor=MODEL_COLOR, label="Hidden genes (RRF)"),
            Patch(
                facecolor=RANDOM_COLOR,
                edgecolor=RANDOM_EDGE_COLOR,
                label="Uniform-random ranking",
            ),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=2,
        columnspacing=1.8,
        handlelength=1.6,
    )
    add_figure_note(fig, figure_note)
    save_figure(fig, output_base)


def plot_top_ratio_boxplots(
    repeats: pd.DataFrame,
    omnibus: pd.DataFrame,
    seed_ratio: float,
    evaluation_ratio: float,
    metric_family: str,
    selected_top_ratio: float,
    actual_k: int,
    figure_note: str,
    output_base: Path,
) -> None:
    metric = percentage_metric(metric_family, evaluation_ratio)
    frame = repeats[np.isclose(repeats.seed_ratio, seed_ratio)].copy()
    top_values = sorted(frame.similarity_top_ratio.unique())
    values = [
        frame[np.isclose(frame.similarity_top_ratio, top)][metric].to_numpy()
        for top in top_values
    ]
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.3), constrained_layout=True)
    positions = np.arange(1, len(top_values) + 1)
    boxes = ax.boxplot(
        values,
        positions=positions,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        boxprops={"edgecolor": "#333333", "linewidth": 1.0},
        medianprops={"color": "#111111", "linewidth": 1.7},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
    )
    palette = [
        PALE_PINK_RED,
        SKY_BLUE,
        BLUE_VIOLET,
        PALE_PINK,
        PALE_CYAN,
    ]
    for patch, color in zip(boxes["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    rng = np.random.default_rng(20260901 + int(round(evaluation_ratio * 1000)))
    for position, sample, color in zip(positions, values, palette):
        jitter = rng.uniform(-0.16, 0.16, size=len(sample))
        ax.scatter(
            position + jitter,
            sample,
            s=10,
            color=color,
            edgecolors="none",
            alpha=0.28,
            zorder=2,
        )
        ax.scatter(
            position,
            np.mean(sample),
            marker="D",
            s=38,
            facecolor="white",
            edgecolor="#111111",
            linewidth=0.9,
            zorder=4,
        )
    ymax = max(float(np.max(sample)) for sample in values)
    ymin = min(float(np.min(sample)) for sample in values)
    span = max(ymax - ymin, 0.05)
    omnibus_row = omnibus[np.isclose(omnibus.evaluation_ratio, evaluation_ratio)].iloc[0]
    ax.set_ylim(ymin - 0.05 * span, ymax + 0.18 * span)
    labels = []
    for top in top_values:
        suffix = ""
        if math.isclose(top, selected_top_ratio):
            suffix = "\nselected"
        labels.append(f"{top:g}{suffix}")
    ax.set_xticks(positions, labels)
    ax.set_xlabel("Similarity aggregation top ratio")
    ax.set_ylabel(f"NDCG@{percentage_label(evaluation_ratio)}")
    ax.set_title(
        f"NDCG distribution at {percentage_label(evaluation_ratio)} (K={actual_k})",
        pad=10,
        weight="semibold",
    )
    ax.text(
        0.99,
        0.98,
        f"Friedman p={float(omnibus_row.friedman_p_value):.2e}, "
        f"q={float(omnibus_row.fdr_q_value):.2e}\n"
        "Overall paired comparison across all aggregation ratios",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=PLOT_FONT_SIZE_PT,
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.9, "pad": 3},
    )
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    add_figure_note(fig, figure_note)
    save_figure(fig, output_base)


def plot_enrichment_profile(
    model_repeats: pd.DataFrame,
    evaluation_ratios: list[float],
    cutoffs: pd.DataFrame,
    seed_ratio: float,
    figure_note: str,
    output_base: Path,
) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    x = np.arange(len(evaluation_ratios), dtype=float)
    means, errors = [], []
    for ratio in evaluation_ratios:
        values = model_repeats[percentage_metric("enrichment", ratio)].to_numpy(
            dtype=float
        )
        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / math.sqrt(len(values)))
        means.append(mean)
        errors.append(1.96 * sem)
    bars = ax.bar(
        x,
        means,
        yerr=errors,
        width=0.40,
        color=MODEL_COLOR,
        edgecolor="#6B6B6B",
        linewidth=0.9,
        error_kw={"ecolor": "#333333", "elinewidth": 1.1, "capsize": 4},
        capsize=3,
    )
    cutoff_frame = cutoffs[np.isclose(cutoffs.seed_ratio, seed_ratio)].set_index(
        "evaluation_ratio"
    )
    labels = [
        f"{ratio * 100:g}%\n(K={int(cutoff_frame.loc[ratio, 'actual_k'])})"
        for ratio in evaluation_ratios
    ]
    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}×",
            ha="center",
            va="bottom",
            fontsize=PLOT_FONT_SIZE_PT,
            weight="semibold",
        )
    ax.axhline(
        1.0,
        color=RANDOM_COLOR,
        linestyle="--",
        linewidth=1.4,
        label="Uniform-random level (1×)",
    )
    ax.set_xticks(x, labels)
    ax.set_xlabel("Screening depth (% of genes actually ranked)")
    ax.set_ylabel("Recall enrichment over random (fold)")
    ax.set_title("Hidden-gene enrichment above random ranking", pad=10, weight="semibold")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    add_figure_note(fig, figure_note)
    save_figure(fig, output_base)


def plot_normalized_rank_boxplots(
    model_repeats: pd.DataFrame,
    random_repeats: pd.DataFrame,
    rank_metric: str,
    q_value: float,
    figure_note: str,
    output_base: Path,
) -> None:
    model_values = model_repeats[rank_metric].to_numpy(dtype=float)
    random_values = random_repeats[rank_metric].to_numpy(dtype=float)
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(4.8, 5.2), constrained_layout=True)
    positions = [0.85, 1.65]
    boxes = ax.boxplot(
        [model_values, random_values],
        positions=positions,
        widths=0.32,
        patch_artist=True,
        showfliers=False,
        boxprops={"edgecolor": "#333333", "linewidth": 1.1},
        medianprops={"color": "#111111", "linewidth": 1.7},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
    )
    for patch, color in zip(boxes["boxes"], [MODEL_COLOR, RANDOM_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)
    for position, sample in zip(positions, [model_values, random_values]):
        ax.scatter(
            position,
            np.mean(sample),
            marker="D",
            s=38,
            facecolor="white",
            edgecolor="#111111",
            linewidth=0.9,
            zorder=4,
        )
    ymax = max(float(model_values.max()), float(random_values.max()))
    ymin = min(float(model_values.min()), float(random_values.min()), 0.0)
    span = max(ymax - ymin, 0.1)
    bracket_y = ymax + 0.06 * span
    bracket_h = 0.02 * span
    ax.plot(
        [positions[0], positions[0], positions[1], positions[1]],
        [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y],
        color="#333333",
        linewidth=0.9,
    )
    ax.text(
        float(np.mean(positions)),
        bracket_y + 1.35 * bracket_h,
        f"{significance_stars(q_value)}  q={q_value:.2g}",
        ha="center",
        va="bottom",
        fontsize=PLOT_FONT_SIZE_PT,
    )
    ax.set_xticks(positions, ["Hidden genes\n(RRF)", "Uniform-random\nranking"])
    ax.set_xlim(0.45, 2.05)
    display = "Mean" if rank_metric == "normalized_mean_rank" else "Median"
    ax.set_ylabel(f"Normalized {display.lower()} hidden-gene rank")
    ax.set_title(
        f"Normalized hidden-gene {display.lower()} rank",
        pad=10,
        weight="semibold",
    )
    ax.set_ylim(ymin, bracket_y + 0.16 * span)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    add_figure_note(fig, figure_note)
    save_figure(fig, output_base)


def plot_heatmap(
    runs: pd.DataFrame,
    metric: str,
    output_base: Path,
    figure_note: str,
) -> None:
    """Plot the observed NDCG for every evaluated parameter combination."""
    absolute, seed_values, top_values = value_matrix(runs, metric)
    finite = absolute[np.isfinite(absolute)]
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    padding = max((upper - lower) * 0.05, 0.005)

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.4), constrained_layout=True)
    x_edges = np.arange(len(top_values) + 1, dtype=float) - 0.5
    y_edges = np.arange(len(seed_values) + 1, dtype=float) - 0.5
    image = ax.pcolormesh(
        x_edges,
        y_edges,
        absolute,
        cmap=HEATMAP_CMAP,
        vmin=lower - padding,
        vmax=upper + padding,
        shading="flat",
    )
    ax.set_xlim(-0.5, len(top_values) - 0.5)
    ax.set_ylim(len(seed_values) - 0.5, -0.5)
    ax.set_xticks(range(len(top_values)), [f"{value:g}" for value in top_values])
    ax.set_yticks(range(len(seed_values)), [f"{value:g}" for value in seed_values])
    ax.set_xlabel("Similarity aggregation top ratio")
    ax.set_ylabel("Known-gene seed ratio")
    ax.set_title(
        f"Observed RRF {metric_label(metric)} across evaluation settings",
        pad=10,
        weight="semibold",
    )

    for i, _seed_ratio in enumerate(seed_values):
        for j, _top_ratio in enumerate(top_values):
            if not np.isfinite(absolute[i, j]):
                continue
            label = f"{absolute[i, j]:.3f}"
            is_maximum = bool(np.isclose(absolute[i, j], upper, rtol=0.0, atol=1e-12))
            normalized = (absolute[i, j] - (lower - padding)) / max(
                upper - lower + 2 * padding, 1e-12
            )
            color = "white" if normalized > 0.62 else "#15202B"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=PLOT_FONT_SIZE_PT,
                color=color,
                weight="bold" if is_maximum else "normal",
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    colorbar.set_label(metric_label(metric))
    add_figure_note(fig, figure_note)
    save_figure(fig, output_base)


def create_experiment_analysis(
    experiment_dir: Path,
    runs: pd.DataFrame,
    metrics: pd.DataFrame,
    repeats: pd.DataFrame,
    cutoffs: pd.DataFrame,
    evaluation_ratios: list[float],
    primary_evaluation_ratio: float,
    selection_metric_family: str,
) -> dict:
    """Select the global Top-1%-NDCG best pair and analyze that protocol."""
    if runs.empty or repeats.empty:
        raise RuntimeError("no completed grid runs are available for analysis")
    analysis_dir = experiment_dir / "analysis"
    figures_dir = analysis_dir / "figures_v3"
    tables_dir = analysis_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    for generated_file in figures_dir.iterdir():
        if generated_file.is_file():
            generated_file.unlink()

    evaluation_ratios = sorted(set(float(value) for value in evaluation_ratios))
    selection_metric = percentage_metric(
        selection_metric_family, primary_evaluation_ratio
    )
    selection_column = f"rrf_{selection_metric}_mean"
    if selection_column not in runs.columns:
        raise KeyError(f"percentage selection metric is absent: {selection_column}")

    globally_ranked = runs.sort_values(
        selection_column,
        ascending=metric_direction(selection_metric) == "min",
    ).reset_index(drop=True)
    globally_ranked.insert(
        0, "selection_rank", np.arange(1, len(globally_ranked) + 1)
    )
    selected = globally_ranked.iloc[0]
    selected_seed_ratio = float(selected.seed_ratio)
    selected_top_ratio = float(selected.similarity_top_ratio)
    tested_top_values = sorted(
        float(value) for value in runs.similarity_top_ratio.unique()
    )
    tested_seed_values = sorted(float(value) for value in runs.seed_ratio.unique())
    top_at_boundary = math.isclose(
        selected_top_ratio, tested_top_values[0]
    ) or math.isclose(selected_top_ratio, tested_top_values[-1])
    seed_at_boundary = math.isclose(
        selected_seed_ratio, tested_seed_values[0]
    ) or math.isclose(selected_seed_ratio, tested_seed_values[-1])
    selected_at_boundary = top_at_boundary or seed_at_boundary
    ranked = runs.copy()
    ranked["selection_rank_within_seed_ratio"] = ranked.groupby("seed_ratio")[
        selection_column
    ].rank(
        method="min",
        ascending=metric_direction(selection_metric) == "min",
    )
    ranked = ranked.sort_values(
        ["seed_ratio", "selection_rank_within_seed_ratio"]
    ).reset_index(drop=True)
    best_by_seed = select_best_by_seed(runs, selection_metric)
    robustness = summarize_top_ratio_robustness(runs, selection_metric)
    omnibus, pairwise = paired_percentage_tests(
        repeats,
        selected_seed_ratio,
        evaluation_ratios,
        selection_metric_family,
    )

    atomic_write_csv(experiment_dir / "grid_runs_summary.csv", runs)
    atomic_write_csv(tables_dir / "grid_metrics_summary_long.csv", metrics)
    atomic_write_csv(tables_dir / "rrf_metrics_by_repeat_percentage.csv", repeats)
    atomic_write_csv(tables_dir / "percentage_cutoff_definitions.csv", cutoffs)
    atomic_write_csv(tables_dir / "ranked_configurations_within_seed_ratio.csv", ranked)
    atomic_write_csv(tables_dir / "globally_ranked_configurations.csv", globally_ranked)
    atomic_write_csv(tables_dir / "best_top_ratio_by_seed_ratio.csv", best_by_seed)
    atomic_write_csv(tables_dir / "top_ratio_robustness.csv", robustness)
    atomic_write_csv(tables_dir / "friedman_tests_by_screening_depth.csv", omnibus)
    atomic_write_csv(tables_dir / "paired_wilcoxon_tests_by_screening_depth.csv", pairwise)

    cutoff_primary = cutoffs[
        np.isclose(cutoffs.seed_ratio, selected_seed_ratio)
    ].set_index("evaluation_ratio")
    repeat_count = int(selected.num_repeats)
    top_values = sorted(float(value) for value in runs.similarity_top_ratio.unique())
    profile_note = primary_protocol_note(
        cutoffs,
        selected_seed_ratio,
        f"selected aggregation={selected_top_ratio:g}",
        repeat_count,
    )
    comparison_note = primary_protocol_note(
        cutoffs,
        selected_seed_ratio,
        f"selected aggregation={selected_top_ratio:g}",
        repeat_count,
    )
    comparison_rows: list[dict] = []
    percentage_summary_rows: list[dict] = []
    for _, run in runs.iterrows():
        seed_cutoffs = cutoff_map_for_seed(cutoffs, float(run.seed_ratio))
        for ratio in evaluation_ratios:
            for family in [
                "recall",
                CAPACITY_RECALL_FAMILY,
                "average_precision",
                "ndcg",
                "enrichment",
            ]:
                metric = percentage_metric(family, ratio)
                prefix = f"rrf_{metric}"
                percentage_summary_rows.append(
                    {
                        "run_id": run.run_id,
                        "seed_ratio": float(run.seed_ratio),
                        "similarity_top_ratio": float(run.similarity_top_ratio),
                        "screening_ratio": ratio,
                        "screening_label": percentage_label(ratio),
                        "actual_k": int(seed_cutoffs[ratio]),
                        "metric": metric,
                        "metric_label": metric_label(metric),
                        "mean": float(run[f"{prefix}_mean"]),
                        "std": float(run[f"{prefix}_std"]),
                        "ci95_low": float(run[f"{prefix}_ci95_low"]),
                        "ci95_high": float(run[f"{prefix}_ci95_high"]),
                    }
                )
    atomic_write_csv(
        tables_dir / "percentage_metrics_summary_long.csv",
        pd.DataFrame(percentage_summary_rows),
    )
    heatmap_value_rows: list[dict] = []
    for ratio in evaluation_ratios:
        metric = percentage_metric("ndcg", ratio)
        column = f"rrf_{metric}_mean"
        heatmap_maximum = float(runs[column].max())
        for seed_ratio, seed_frame in runs.groupby("seed_ratio", sort=True):
            seed_cutoff = cutoffs[
                np.isclose(cutoffs.seed_ratio, float(seed_ratio))
                & np.isclose(cutoffs.evaluation_ratio, ratio)
            ].iloc[0]
            for _, row in seed_frame.iterrows():
                heatmap_value_rows.append(
                    {
                        "screening_ratio": ratio,
                        "screening_label": percentage_label(ratio),
                        "seed_ratio": float(seed_ratio),
                        "seed_count": int(seed_cutoff.seed_count),
                        "hidden_count": int(seed_cutoff.hidden_count),
                        "ranking_count": int(seed_cutoff.ranking_count),
                        "actual_k": int(seed_cutoff.actual_k),
                        "similarity_top_ratio": float(row.similarity_top_ratio),
                        "ndcg": float(row[column]),
                        "heatmap_maximum": bool(
                            np.isclose(
                                float(row[column]),
                                heatmap_maximum,
                                rtol=0.0,
                                atol=1e-12,
                            )
                        ),
                    }
                )
    atomic_write_csv(
        tables_dir / "ndcg_heatmap_actual_values.csv",
        pd.DataFrame(heatmap_value_rows),
    )
    for ratio in evaluation_ratios:
        k = int(cutoff_primary.loc[ratio, "actual_k"])
        ranking_count = int(cutoff_primary.loc[ratio, "ranking_count"])
        hidden_count = int(cutoff_primary.loc[ratio, "hidden_count"])
        for family in [
            "recall",
            CAPACITY_RECALL_FAMILY,
            "average_precision",
            "ndcg",
        ]:
            metric = percentage_metric(family, ratio)
            column = f"rrf_{metric}_mean"
            random_value = random_metric_expectation(
                family,
                k,
                ranking_count,
                hidden_count,
            )
            comparison_rows.append(
                {
                    "seed_ratio": selected_seed_ratio,
                    "screening_ratio": ratio,
                    "screening_label": percentage_label(ratio),
                    "actual_k": k,
                    "metric": metric,
                    "metric_label": metric_label(metric),
                    "selected_top_ratio": selected_top_ratio,
                    "selected_mean": float(selected[column]),
                    "random_expectation": float(random_value),
                    "selected_fold_over_random": (
                        float(selected[column]) / random_value
                        if random_value > 0
                        else float("nan")
                    ),
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    atomic_write_csv(tables_dir / "selected_vs_random_by_screening_depth.csv", comparison)
    selected_repeats = repeats[
        np.isclose(repeats.seed_ratio, selected_seed_ratio)
        & np.isclose(repeats.similarity_top_ratio, selected_top_ratio)
    ].copy()
    random_repeats = simulate_uniform_random_repeats(
        selected_repeats,
        evaluation_ratios,
    )
    atomic_write_csv(
        tables_dir / "matched_uniform_random_repeat_metrics.csv",
        random_repeats,
    )
    rank_metric_names = ["normalized_mean_rank", "normalized_median_rank"]
    rank_p_values: list[float] = []
    for rank_metric in rank_metric_names:
        favorable = 0.5 - selected_repeats[rank_metric].to_numpy(dtype=float)
        rank_p_values.append(
            1.0
            if np.allclose(favorable, 0.0)
            else float(
                wilcoxon(
                    favorable,
                    alternative="greater",
                    zero_method="wilcox",
                    method="auto",
                ).pvalue
            )
        )
    rank_q_values = benjamini_hochberg(rank_p_values)
    atomic_write_csv(
        tables_dir / "normalized_rank_significance.csv",
        pd.DataFrame(
            {
                "metric": rank_metric_names,
                "one_sided_wilcoxon_p_value": rank_p_values,
                "bh_fdr_q_value": rank_q_values,
            }
        ),
    )

    for figure_index, ratio in enumerate(evaluation_ratios, start=1):
        metric = percentage_metric("ndcg", ratio)
        plot_heatmap(
            runs,
            metric,
            figures_dir / f"{figure_index:02d}_heatmap_{metric}",
            heatmap_protocol_note(cutoffs, top_values, ratio),
        )

    for figure_index, (family, file_label) in enumerate(
        [
            (
                CAPACITY_RECALL_FAMILY,
                "capacity_normalized_recall_profile",
            ),
            ("average_precision", "map_depth_profile"),
            ("ndcg", "ndcg_depth_profile"),
        ],
        start=4,
    ):
        plot_metric_depth_profile(
            selected_repeats,
            random_repeats,
            cutoffs,
            selected_seed_ratio,
            evaluation_ratios,
            family,
            profile_note,
            figures_dir / f"{figure_index:02d}_{file_label}",
        )

    plot_enrichment_profile(
        selected_repeats,
        evaluation_ratios,
        cutoffs,
        selected_seed_ratio,
        compact_protocol_note(profile_note),
        figures_dir / "07_recall_enrichment_over_random",
    )
    plot_normalized_rank_boxplots(
        selected_repeats,
        random_repeats,
        "normalized_mean_rank",
        float(rank_q_values[0]),
        compact_protocol_note(comparison_note),
        figures_dir / "08_normalized_mean_rank",
    )
    plot_normalized_rank_boxplots(
        selected_repeats,
        random_repeats,
        "normalized_median_rank",
        float(rank_q_values[1]),
        compact_protocol_note(comparison_note),
        figures_dir / "09_normalized_median_rank",
    )

    guide_records: list[dict] = []
    for index, ratio in enumerate(evaluation_ratios, start=1):
        guide_records.append(
            {
                "figure": f"{index:02d}_heatmap_{percentage_metric('ndcg', ratio)}",
                "question": f"{percentage_label(ratio)}下，各种子比例与聚合比例组合的实际NDCG是多少？",
                "how_to_read": (
                    "单元格直接给出实际NDCG，越高越好；本张热图中的最大实际值仅以粗体数字标出。"
                    "不同seed行的种子数和隐藏基因数不同，聚合比例应优先在同一行内比较，"
                    "不要根据跨行绝对值机械选择seed比例。"
                ),
            }
        )
    for index, (file_label, metric_name) in enumerate(
        [
            (
                "capacity_normalized_recall_profile",
                "Capacity-normalized Recall",
            ),
            ("map_depth_profile", "MAP"),
            ("ndcg_depth_profile", "NDCG"),
        ],
        start=4,
    ):
        guide_records.append(
            {
                "figure": f"{index:02d}_{file_label}",
                "question": f"Top 1%全局最优方案的{metric_name}是否在三个筛选深度均优于随机排序？",
                "how_to_read": (
                    "每个筛选深度包含RRF隐藏基因与匹配随机排序两个箱体；"
                    "Capacity-normalized Recall按Hits@K/min(K,验证基因数)计算并以0–100%显示；"
                    "其他指标保持原定义。箱体越高越好，括号上方为相对精确随机期望的"
                    "单侧Wilcoxon检验及BH-FDR校正q值。"
                ),
            }
        )
    guide_records.extend(
        [
            {
                "figure": "07_recall_enrichment_over_random",
                "question": "不同筛选深度下，模型找回隐藏基因的数量是随机排序的多少倍？",
                "how_to_read": "三根柱分别对应Top 1%、5%、10%，柱高为平均Recall富集倍数，误差线为95%置信区间；1倍虚线是随机水平。",
            },
            {
                "figure": "08_normalized_mean_rank",
                "question": "全部隐藏基因的平均排名是否整体前移？",
                "how_to_read": "固定使用Top 1%全局最优种子比例和聚合比例，比较RRF隐藏基因与匹配随机排序的归一化平均排名；越低越好。",
            },
            {
                "figure": "09_normalized_median_rank",
                "question": "一个典型隐藏基因的排名是否整体前移？",
                "how_to_read": "固定使用Top 1%全局最优方案，在线性纵轴上比较RRF与匹配随机排序的归一化中位排名；越低越好。",
            },
        ]
    )
    guide = pd.DataFrame(guide_records)
    atomic_write_csv(tables_dir / "figure_guide.csv", guide)
    guide_lines = ["百分比排序评测图片阅读指南", "=" * 76, ""]
    for item in guide.to_dict(orient="records"):
        guide_lines.extend(
            [
                item["figure"],
                f"作用：{item['question']}",
                f"读法：{item['how_to_read']}",
                "",
            ]
        )
    (figures_dir / "README_如何阅读批量比较图片.txt").write_text(
        "\n".join(guide_lines), encoding="utf-8"
    )

    terminal_lines = [
        "=" * 104,
        "GRID RESULT | percentage-based candidate ranking evaluation",
        (
            f"Selected protocol: seed ratio={selected_seed_ratio:g}; selection="
            f"{metric_label(selection_metric)}; repeats={int(selected.num_repeats)}"
        ),
        f"Seed ratio and aggregation ratio were jointly selected by global {metric_label(selection_metric)}; this is exploratory.",
        "-" * 104,
        f"Selected aggregation top ratio: {selected_top_ratio:g}",
        (
            "Selection status: BOUNDARY VALUE - extend the search before calling it an optimum."
            if selected_at_boundary
            else "Selection status: INTERIOR VALUE - best observed is not a tested boundary."
        ),
        "-" * 104,
        f"{'Depth':<12}{'K':>7}  {'Metric':<20}{'Actual':>13}{'Random':>13}{'Fold':>13}",
        "-" * 104,
    ]
    for ratio in evaluation_ratios:
        subset = comparison[np.isclose(comparison.screening_ratio, ratio)]
        for row_index, (_, row) in enumerate(subset.iterrows()):
            terminal_lines.append(
                f"{(row.screening_label if row_index == 0 else ''):<12}"
                f"{(int(row.actual_k) if row_index == 0 else ''):>7}"
                f"  {row.metric_label:<20}{float(row.selected_mean):>13.4f}"
                f"{float(row.random_expectation):>13.4f}"
                f"{float(row.selected_fold_over_random):>13.2f}"
            )
    terminal_lines.extend(["-" * 104, "Aggregation-ratio omnibus tests (actual NDCG distributions):"])
    for ratio in evaluation_ratios:
        match = omnibus[np.isclose(omnibus.evaluation_ratio, ratio)]
        if not match.empty:
            row = match.iloc[0]
            terminal_lines.append(
                f"  {percentage_label(ratio):<8} Friedman p={float(row.friedman_p_value):.2e}; "
                f"BH-FDR q={float(row.fdr_q_value):.2e}"
            )
    terminal_lines.extend(
        [
            "-" * 104,
            (
                "Global rank | selected normalized Mean Rank="
                f"{selected_repeats.normalized_mean_rank.mean():.4f}; "
                "Median Rank="
                f"{selected_repeats.normalized_median_rank.mean():.4f}; random=0.5000"
            ),
            "Selection was made on these repeated splits and is exploratory; use nested or external validation for an unbiased final claim.",
            "=" * 104,
        ]
    )
    terminal_report = "\n".join(terminal_lines)
    (analysis_dir / "RESULT_SUMMARY.txt").write_text(
        terminal_report + "\n", encoding="utf-8"
    )

    report = {
        "status": "complete",
        "completed_at_local": datetime.now().isoformat(timespec="seconds"),
        "completed_configurations": len(runs),
        "primary_protocol": {
            "seed_ratio": selected_seed_ratio,
            "screening_ratio": primary_evaluation_ratio,
            "screening_label": percentage_label(primary_evaluation_ratio),
            "selection_metric": selection_metric,
        },
        "selected_aggregation": {
            "seed_ratio": selected_seed_ratio,
            "similarity_top_ratio": selected_top_ratio,
            "score": float(selected[selection_column]),
            "run_id": selected.run_id,
            "at_tested_boundary": selected_at_boundary,
            "interpretation": (
                "Do not call this an optimum; expand the tested aggregation range."
                if selected_at_boundary
                else "This is an interior best value within the tested aggregation grid."
            ),
        },
        "evaluation_ratios": evaluation_ratios,
        "important_methodological_note": (
            "The selected seed ratio and aggregation ratio were jointly chosen "
            "by global NDCG@Top 1% on these repeated splits. "
            "The comparison is exploratory and selection-optimistic; nested or "
            "independent validation is required for an unbiased final claim."
        ),
        "artifacts": {
            "tables": str(tables_dir),
            "figures": str(figures_dir),
            "cutoff_definitions": str(tables_dir / "percentage_cutoff_definitions.csv"),
            "percentage_metric_summary": str(
                tables_dir / "percentage_metrics_summary_long.csv"
            ),
            "percentage_repeat_metrics": str(
                tables_dir / "rrf_metrics_by_repeat_percentage.csv"
            ),
            "heatmap_actual_values": str(
                tables_dir / "ndcg_heatmap_actual_values.csv"
            ),
            "comparison": str(tables_dir / "selected_vs_random_by_screening_depth.csv"),
            "figure_guide": str(tables_dir / "figure_guide.csv"),
        },
        "terminal_report": terminal_report,
    }
    atomic_write_json(analysis_dir / "analysis_report.json", report)
    return report


def main() -> None:
    args = parse_args()
    selected_model = resolve_requested_model(
        args.model,
        args.model_run if args.model_run is not None else GRID_MODEL_RUN,
    )
    if args.max_parallel < 1:
        raise ValueError("max_parallel must be positive")
    evaluation_ratios = sorted(set(float(value) for value in args.evaluation_ratios))
    if not evaluation_ratios or any(not 0.0 < value <= 1.0 for value in evaluation_ratios):
        raise ValueError("all evaluation_ratios must be in (0, 1]")
    if not any(
        np.isclose(args.primary_evaluation_ratio, value)
        for value in evaluation_ratios
    ):
        raise ValueError(
            "primary_evaluation_ratio must be included in evaluation_ratios"
        )
    context = prepare_ranking_context(
        model_path=selected_model,
        data_dir=args.data_dir,
        requested_heads=args.heads,
        device_name=args.device,
        projection_batch_size=args.projection_batch_size,
        allow_nonfull_data=args.allow_nonfull_data,
    )
    cutoffs = cutoff_table(context, args.seed_ratios, evaluation_ratios)
    configurations = [
        EvaluationParameters(
            seed_ratio=seed_ratio,
            similarity_top_ratio=top_ratio,
            num_repeats=args.num_repeats,
            split_seed=args.split_seed,
            rrf_k=args.rrf_k,
            ks=tuple(
                sorted(
                    set(cutoff_map_for_seed(cutoffs, seed_ratio).values())
                )
            ),
        )
        for seed_ratio, top_ratio in itertools.product(
            sorted(set(args.seed_ratios)),
            sorted(set(args.similarity_top_ratios)),
        )
    ]
    experiment_dir = (
        args.output_root.expanduser().resolve() / args.experiment_name
    )
    (experiment_dir / "runs").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "metadata").mkdir(parents=True, exist_ok=True)
    experiment_config_path = experiment_dir / "metadata" / "experiment_config.json"
    validate_experiment_identity(
        experiment_config_path, context, args.experiment_name
    )
    protocol = {
        "experiment_name": args.experiment_name,
        "runner_script": str(Path(__file__).resolve()),
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "created_or_updated_at_local": datetime.now().isoformat(timespec="seconds"),
        "model_path": context.model_path,
        "model_run": Path(context.model_path).parent.name,
        "model_sha256": context.model_sha256,
        "active_heads": context.active_heads,
        "data_directory": context.data_dir,
        "head_weights": "equal",
        "seed_ratios": sorted(set(args.seed_ratios)),
        "similarity_top_ratios": sorted(set(args.similarity_top_ratios)),
        "num_repeats": args.num_repeats,
        "split_seed": args.split_seed,
        "rrf_k": args.rrf_k,
        "screening_depth_definition": (
            "ceil(evaluation_ratio * genes actually ranked after seed removal)"
        ),
        "capacity_normalized_recall_definition": (
            "Hits@K / min(K, hidden validation-gene count); stored on 0-1 scale "
            "and plotted on 0-100% scale"
        ),
        "evaluation_ratios": evaluation_ratios,
        "primary_evaluation_ratio": args.primary_evaluation_ratio,
        "selection_metric_family": args.selection_metric_family,
        "selection_metric": percentage_metric(
            args.selection_metric_family, args.primary_evaluation_ratio
        ),
        "max_parallel": args.max_parallel,
        "save_run_plots": args.save_run_plots,
        "plot_settings": {
            "font_family": PLOT_FONT_FAMILY,
            "font_size_pt": PLOT_FONT_SIZE_PT,
            "dpi": PLOT_DPI,
            "formats": ["png", "pdf", "svg"],
            "vector_formats": ["pdf", "svg"],
        },
    }
    atomic_write_json(experiment_config_path, protocol)
    atomic_write_csv(
        experiment_dir / "metadata" / "percentage_cutoff_definitions.csv",
        cutoffs,
    )

    pending: list[EvaluationParameters] = []
    manifest_rows: list[dict] = []
    for parameters in configurations:
        run_dir = experiment_dir / "runs" / run_identifier(parameters)
        completed = is_completed(
            run_dir, parameters, context.model_sha256, context.active_heads
        )
        status = "skipped_complete" if completed and not args.rerun_completed else "pending"
        manifest_rows.append(
            {
                **parameter_key(parameters),
                "model_run": Path(context.model_path).parent.name,
                "active_heads": ",".join(context.active_heads),
                "run_id": run_identifier(parameters),
                "status": status,
                "run_directory": str(run_dir),
            }
        )
        if status == "pending":
            pending.append(parameters)
    atomic_write_csv(experiment_dir / "grid_manifest.csv", pd.DataFrame(manifest_rows))
    print(
        f"Model run: {Path(context.model_path).parent.name}\n"
        f"Heads ({len(context.active_heads)}): {', '.join(context.active_heads)}\n"
        f"Grid: {len(configurations)} configurations | pending: {len(pending)} | "
        f"parallel workers: {args.max_parallel}\n"
        f"Screening depths: {', '.join(percentage_label(value) for value in evaluation_ratios)} | "
        f"primary: {percentage_label(args.primary_evaluation_ratio)}\n"
        f"Per-configuration plots: {'enabled' if args.save_run_plots else 'disabled'}; "
        "experiment-level scientific figures will always be saved.",
        flush=True,
    )

    failures: list[dict] = []
    if pending:
        started = time.time()
        print(
            f"Queued {len(pending)} configurations; at most "
            f"{min(args.max_parallel, len(pending))} execute simultaneously.",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=min(args.max_parallel, len(pending))) as pool:
            future_map = {}
            for parameters in pending:
                run_dir = experiment_dir / "runs" / run_identifier(parameters)
                future = pool.submit(
                    worker_run,
                    context,
                    parameters,
                    str(run_dir),
                    args.save_run_plots,
                )
                future_map[future] = parameters
            completed_count = 0
            for future in as_completed(future_map):
                parameters = future_map[future]
                completed_count += 1
                try:
                    summary = future.result()
                    print(
                        f"completed {summary['run_id']} | "
                        f"{completed_count}/{len(pending)}",
                        flush=True,
                    )
                except Exception as exc:
                    failures.append(
                        {"run_id": run_identifier(parameters), "error": repr(exc)}
                    )
                    print(
                        f"failed {run_identifier(parameters)} | {exc!r}", flush=True
                    )
        print(f"Batch runtime: {time.time() - started:.1f}s", flush=True)
    if failures:
        atomic_write_json(experiment_dir / "failed_runs.json", {"failures": failures})

    runs, metrics = collect_experiment_results(
        experiment_dir, configurations, context, cutoffs
    )
    repeats = load_rrf_repeat_results(experiment_dir, configurations, cutoffs)
    report = create_experiment_analysis(
        experiment_dir,
        runs,
        metrics,
        repeats,
        cutoffs=cutoffs,
        evaluation_ratios=evaluation_ratios,
        primary_evaluation_ratio=args.primary_evaluation_ratio,
        selection_metric_family=args.selection_metric_family,
    )
    print(
        "\n" + report["terminal_report"] + "\n\n"
        f"Experiment analysis: {experiment_dir}",
        flush=True,
    )
    if failures:
        raise SystemExit(f"{len(failures)} configurations failed; see failed_runs.json")


if __name__ == "__main__":
    main()
