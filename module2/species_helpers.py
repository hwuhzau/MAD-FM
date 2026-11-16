"""Run the established ranking grid independently for all four task datasets.

Each task reads ``<task>/prepared_data`` and writes every grid artifact under
``<task>/ranking_results/<experiment-name>``. Existing completed grid cells are
detected and skipped by the underlying evaluator, so interrupted experiments
can be resumed by running this launcher again. Version 3 exports every figure
as PNG, PDF, and SVG with Arial 10-point text, the approved five-color palette,
and capacity-normalized Recall@K.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


SCRIPT_DIR = Path(__file__).resolve().parent
RANKING_DIR = SCRIPT_DIR.parent
GRID_SCRIPT = SCRIPT_DIR / "ranking_grid.py"
if str(RANKING_DIR) not in sys.path:
    sys.path.insert(0, str(RANKING_DIR))

import ranking_core  # noqa: E402
from ranking_core import (  # noqa: E402
    EvaluationParameters,
    RankingContext,
    atomic_write_csv,
    benjamini_hochberg,
    evaluate_repeated_splits,
    l2_normalize,
    load_prepared_data,
    random_ranking_baseline,
)
from ranking_grid import (  # noqa: E402
    CAPACITY_RECALL_FAMILY,
    capacity_recall_scale,
    percentage_label,
    percentage_metric,
    random_metric_expectation,
    simulate_uniform_random_repeats,
)

# =============================================================================
# USER-EDITABLE DEFAULTS
# =============================================================================
DEFAULT_MODEL_RUN = (
    "seed42_20261005_101049_759729_shared_residual_dim512_bs32_neg49_"
    "lr0.0002_src-go-coexp-hic-string-tf_v3"
)
TASKS = {
    "arabidopsis_TARGET": "Arabidopsis thaliana",
    "homo_BRCA": "Homo sapiens",
    "homo_LUAD": "Homo sapiens",
    "zea_ZILI": "Zea mays",
}
GRID_SEED_RATIOS = (0.10, 0.30, 0.50, 0.70)
GRID_SIMILARITY_TOP_RATIOS = (0.10, 0.30, 0.50, 0.70, 1.00)
GRID_EVALUATION_RATIOS = (0.01, 0.05, 0.10)
GRID_PRIMARY_EVALUATION_RATIO = 0.01
GRID_NUM_REPEATS = 100
GRID_SPLIT_SEED = 42
GRID_RRF_K = 60.0
GRID_MAX_PARALLEL = 4
GRID_PROJECTION_BATCH_SIZE = 1024
GRID_EXPERIMENT_NAME = "ranking_hyperparameter_grid_v3"
PALE_PINK_RED = "#E8A4A4"
SKY_BLUE = "#87CEEB"
BLUE_VIOLET = "#5C6B9E"
PALE_PINK = "#E6B8B8"
PALE_CYAN = "#A8D5E5"
RRF_COLOR = BLUE_VIOLET
RAW_ESM2_COLOR = PALE_CYAN
RAW_ESM2_EDGE_COLOR = "#6B6B6B"
RANDOM_COLOR = PALE_PINK
RANDOM_EDGE_COLOR = "#6B6B6B"
GRID_COLOR = "#D9D9D9"
PLOT_FONT_FAMILY = (
    "Arial" if Path("C:/Windows/Fonts/arial.ttf").exists() else "DejaVu Sans"
)
PLOT_FONT_SIZE_PT = 10
PLOT_DPI = 300
# =============================================================================


def configure_plot_style() -> None:
    """Use one readable publication style for all v3 figures."""
    font_family = PLOT_FONT_FAMILY
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.sans-serif": [font_family],
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
    """Save matching PNG plus true-vector PDF and SVG versions."""
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
        description="Run the same trained maize model ranking grid on four datasets"
    )
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=list(TASKS))
    parser.add_argument("--model-run", default=DEFAULT_MODEL_RUN)
    parser.add_argument("--experiment-name", default=GRID_EXPERIMENT_NAME)
    parser.add_argument("--max-parallel", type=int, default=GRID_MAX_PARALLEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--projection-batch-size", type=int, default=GRID_PROJECTION_BATCH_SIZE
    )
    parser.add_argument("--num-repeats", type=int, default=GRID_NUM_REPEATS)
    parser.add_argument("--split-seed", type=int, default=GRID_SPLIT_SEED)
    parser.add_argument("--rrf-k", type=float, default=GRID_RRF_K)
    parser.add_argument(
        "--seed-ratios", nargs="+", type=float, default=list(GRID_SEED_RATIOS)
    )
    parser.add_argument(
        "--similarity-top-ratios", nargs="+", type=float,
        default=list(GRID_SIMILARITY_TOP_RATIOS),
    )
    parser.add_argument(
        "--evaluation-ratios", nargs="+", type=float,
        default=list(GRID_EVALUATION_RATIOS),
    )
    parser.add_argument(
        "--primary-evaluation-ratio", type=float,
        default=GRID_PRIMARY_EVALUATION_RATIO,
    )
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--save-run-plots", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--baseline-comparison-only",
        action="store_true",
        help=(
            "reuse a completed grid analysis and only generate figures_bc_v3 with "
            "multi-relation RRF, raw ESM-2, and uniform-random comparisons"
        ),
    )
    return parser.parse_args()


def add_list(command: list[str], flag: str, values: list[float]) -> None:
    command.append(flag)
    command.extend(f"{value:g}" for value in values)


def build_command(
    args: argparse.Namespace, task: str, allow_missing_prepared: bool = False
) -> tuple[list[str], Path]:
    task_dir = SCRIPT_DIR / task
    prepared_dir = task_dir / "prepared_data"
    output_root = task_dir / "ranking_results"
    if not prepared_dir.exists() and not allow_missing_prepared:
        raise FileNotFoundError(
            f"{prepared_dir} does not exist; first run "
            "prepare_each_species_esm2_embeddings.py"
        )
    command = [
        sys.executable,
        str(GRID_SCRIPT),
        "--model-run", args.model_run,
        "--data-dir", str(prepared_dir),
        "--output-root", str(output_root),
        "--experiment-name", args.experiment_name,
        "--max-parallel", str(args.max_parallel),
        "--device", args.device,
        "--projection-batch-size", str(args.projection_batch_size),
        "--num-repeats", str(args.num_repeats),
        "--split-seed", str(args.split_seed),
        "--rrf-k", f"{args.rrf_k:g}",
        "--primary-evaluation-ratio", f"{args.primary_evaluation_ratio:g}",
        "--allow-nonfull-data",
        "--save-run-plots" if args.save_run_plots else "--no-run-plots",
    ]
    add_list(command, "--seed-ratios", args.seed_ratios)
    add_list(command, "--similarity-top-ratios", args.similarity_top_ratios)
    add_list(command, "--evaluation-ratios", args.evaluation_ratios)
    if args.rerun_completed:
        command.append("--rerun-completed")
    return command, output_root / args.experiment_name


def run_and_tee(command: list[str], log_path: Path, task: str) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("PYTHONUTF8", "1")
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(
            f"\n{'=' * 96}\nStarted {datetime.now().isoformat(timespec='seconds')}\n"
            f"Command: {subprocess.list2cmdline(command)}\n{'=' * 96}\n"
        )
        process = subprocess.Popen(
            command,
            cwd=str(RANKING_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(f"{task} | {line}", end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
        log.write(
            f"\nFinished {datetime.now().isoformat(timespec='seconds')} | "
            f"return_code={return_code}\n"
        )
    return return_code


def _paired_greater_p_value(difference: np.ndarray) -> float:
    difference = np.asarray(difference, dtype=float)
    if not np.isfinite(difference).all():
        raise ValueError("paired comparison contains NaN or infinity")
    if np.allclose(difference, 0.0):
        return 1.0
    return float(
        wilcoxon(
            difference,
            alternative="greater",
            zero_method="wilcox",
            method="auto",
        ).pvalue
    )


def _significance_stars(q_value: float) -> str:
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return "ns"


def _add_figure_note(fig: plt.Figure, note: str) -> None:
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


def _raw_esm2_context(data_dir: Path) -> RankingContext:
    """Build the same combined gene universe using frozen raw ESM-2 vectors."""
    known_ids, known_raw, candidate_ids, candidate_raw, manifest = load_prepared_data(
        data_dir, allow_nonfull_data=True
    )
    vector_by_id = {
        gene_id: vector for gene_id, vector in zip(candidate_ids, candidate_raw)
    }
    for gene_id, vector in zip(known_ids, known_raw):
        if gene_id in vector_by_id:
            candidate_vector = l2_normalize(vector_by_id[gene_id][None])[0]
            known_vector = l2_normalize(vector[None])[0]
            cosine = float(np.dot(candidate_vector, known_vector))
            if cosine < 0.9999:
                raise ValueError(
                    f"known/candidate raw ESM-2 vector mismatch for {gene_id}: {cosine}"
                )
        vector_by_id[gene_id] = vector
    candidate_set = set(candidate_ids)
    all_ids = list(candidate_ids) + [
        gene_id for gene_id in known_ids if gene_id not in candidate_set
    ]
    all_raw = l2_normalize(
        np.stack([vector_by_id[gene_id] for gene_id in all_ids]).astype(np.float32)
    )
    all_index = {gene_id: index for index, gene_id in enumerate(all_ids)}
    known_positions = np.asarray(
        [all_index[gene_id] for gene_id in known_ids], dtype=np.int64
    )
    raw_similarity = (all_raw @ all_raw[known_positions].T).astype(
        np.float32, copy=False
    )
    return RankingContext(
        known_ids=known_ids,
        candidate_ids=candidate_ids,
        all_ids=all_ids,
        active_heads=["raw_esm2"],
        similarities={"raw_esm2": raw_similarity},
        model_path="frozen_raw_esm2_480d",
        model_sha256="not_applicable",
        model_metadata={"architecture": "raw_esm2", "input_dim": 480},
        data_dir=str(data_dir.resolve()),
        data_manifest=manifest,
    )


def _canonicalize_raw_metrics(
    frame: pd.DataFrame,
    cutoff_rows: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()
    for _, cutoff in cutoff_rows.iterrows():
        ratio = float(cutoff.evaluation_ratio)
        actual_k = int(cutoff.actual_k)
        for family in ["recall", "average_precision", "ndcg", "enrichment"]:
            source = f"{family}_at_{actual_k}"
            if source not in frame.columns:
                raise KeyError(f"raw ESM-2 metric is absent: {source}")
            frame[percentage_metric(family, ratio)] = frame[source]
        scale = capacity_recall_scale(actual_k, int(cutoff.hidden_count))
        frame[percentage_metric(CAPACITY_RECALL_FAMILY, ratio)] = np.clip(
            frame[percentage_metric("recall", ratio)] * scale,
            0.0,
            1.0,
        )
    frame["normalized_mean_rank"] = frame["mean_rank"] / frame["ranking_count"]
    frame["normalized_median_rank"] = frame["median_rank"] / frame["ranking_count"]
    return frame


def _ensure_capacity_recall_metrics(
    frame: pd.DataFrame,
    cutoff_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Add V3 capacity-normalized Recall columns to any matched repeat table."""
    frame = frame.copy()
    for _, cutoff in cutoff_rows.iterrows():
        ratio = float(cutoff.evaluation_ratio)
        recall_column = percentage_metric("recall", ratio)
        capacity_column = percentage_metric(CAPACITY_RECALL_FAMILY, ratio)
        if capacity_column not in frame.columns:
            if recall_column not in frame.columns:
                raise KeyError(f"Recall metric is absent: {recall_column}")
            scale = capacity_recall_scale(
                int(cutoff.actual_k),
                int(cutoff.hidden_count),
            )
            frame[capacity_column] = np.clip(
                frame[recall_column] * scale,
                0.0,
                1.0,
            )
    return frame


def _validate_matched_repeats(
    rrf_repeats: pd.DataFrame,
    raw_repeats: pd.DataFrame,
) -> None:
    keys = ["repeat", "seed_count", "hidden_count", "ranking_count"]
    left = rrf_repeats[keys].sort_values("repeat").reset_index(drop=True)
    right = raw_repeats[keys].sort_values("repeat").reset_index(drop=True)
    if not left.equals(right):
        raise RuntimeError(
            "raw ESM-2 and multi-relation RRF evaluations did not use matched splits"
        )


def _three_way_metric_profile(
    rrf_repeats: pd.DataFrame,
    raw_repeats: pd.DataFrame,
    random_repeats: pd.DataFrame,
    cutoff_rows: pd.DataFrame,
    evaluation_ratios: list[float],
    metric_family: str,
    figure_note: str,
    output_base: Path,
) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8.6, 5.8), constrained_layout=False)
    # Only the legend receives a dedicated band; significance annotations
    # retain the original in-plot presentation.
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.13, top=0.80)
    centers = np.arange(1, len(evaluation_ratios) + 1, dtype=float)
    offsets = (-0.25, 0.0, 0.25)
    positions = [centers + offset for offset in offsets]
    series = [rrf_repeats, raw_repeats, random_repeats]
    colors = [RRF_COLOR, RAW_ESM2_COLOR, RANDOM_COLOR]
    edge_colors = [RRF_COLOR, RAW_ESM2_EDGE_COLOR, RANDOM_EDGE_COLOR]
    grouped_values: list[list[np.ndarray]] = []
    common = {
        "widths": 0.21,
        "patch_artist": True,
        "showfliers": False,
        "medianprops": {"color": "#1A1A1A", "linewidth": 1.7},
        "whiskerprops": {"color": "#555555", "linewidth": 1.0},
        "capprops": {"color": "#555555", "linewidth": 1.0},
    }
    for method_frame, method_positions, color, edge_color in zip(
        series, positions, colors, edge_colors
    ):
        values = [
            method_frame[percentage_metric(metric_family, ratio)].to_numpy(dtype=float)
            * (100.0 if metric_family == CAPACITY_RECALL_FAMILY else 1.0)
            for ratio in evaluation_ratios
        ]
        grouped_values.append(values)
        boxes = ax.boxplot(
            values,
            positions=method_positions,
            boxprops={"edgecolor": edge_color, "linewidth": 1.1},
            **common,
        )
        for patch in boxes["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.80)

    raw_p_values = []
    random_p_values = []
    cutoff_by_ratio = cutoff_rows.set_index("evaluation_ratio")
    for ratio in evaluation_ratios:
        metric = percentage_metric(metric_family, ratio)
        test_metric = (
            percentage_metric("recall", ratio)
            if metric_family == CAPACITY_RECALL_FAMILY
            else metric
        )
        raw_p_values.append(
            _paired_greater_p_value(
                rrf_repeats[test_metric].to_numpy(dtype=float)
                - raw_repeats[test_metric].to_numpy(dtype=float)
            )
        )
        cutoff = cutoff_by_ratio.loc[ratio]
        random_expectation = random_metric_expectation(
            (
                "recall"
                if metric_family == CAPACITY_RECALL_FAMILY
                else metric_family
            ),
            int(cutoff.actual_k),
            int(cutoff.ranking_count),
            int(cutoff.hidden_count),
        )
        random_p_values.append(
            _paired_greater_p_value(
                rrf_repeats[test_metric].to_numpy(dtype=float)
                - random_expectation
            )
        )
    raw_q_values = benjamini_hochberg(raw_p_values)
    random_q_values = benjamini_hochberg(random_p_values)
    all_values = np.concatenate(
        [values for method_values in grouped_values for values in method_values]
    )
    if metric_family == CAPACITY_RECALL_FAMILY:
        lower = 0.0
        upper = 100.0
        span = 100.0
    else:
        lower = min(0.0, float(np.min(all_values)))
        upper = float(np.max(all_values))
        span = max(upper - lower, 0.10)
    annotation_upper = upper
    for index, (center, raw_q, random_q) in enumerate(
        zip(centers, raw_q_values, random_q_values)
    ):
        raw_bracket_y = max(
            float(np.max(grouped_values[0][index])),
            float(np.max(grouped_values[1][index])),
        ) + 0.055 * span
        bracket_h = 0.018 * span
        ax.plot(
            [center - 0.25, center - 0.25, center, center],
            [
                raw_bracket_y,
                raw_bracket_y + bracket_h,
                raw_bracket_y + bracket_h,
                raw_bracket_y,
            ],
            color="#333333",
            linewidth=0.9,
        )
        ax.text(
            center - 0.125,
            raw_bracket_y + 1.35 * bracket_h,
            f"RRF > raw: {_significance_stars(float(raw_q))}  q={float(raw_q):.2g}",
            ha="center",
            va="bottom",
            fontsize=PLOT_FONT_SIZE_PT,
        )
        random_bracket_y = max(
            float(np.max(grouped_values[0][index])),
            float(np.max(grouped_values[1][index])),
            float(np.max(grouped_values[2][index])),
        ) + 0.145 * span
        ax.plot(
            [center - 0.25, center - 0.25, center + 0.25, center + 0.25],
            [
                random_bracket_y,
                random_bracket_y + bracket_h,
                random_bracket_y + bracket_h,
                random_bracket_y,
            ],
            color="#333333",
            linewidth=0.9,
        )
        ax.text(
            center,
            random_bracket_y + 1.35 * bracket_h,
            f"RRF > random: {_significance_stars(float(random_q))}  q={float(random_q):.2g}",
            ha="center",
            va="bottom",
            fontsize=PLOT_FONT_SIZE_PT,
        )
        annotation_upper = max(
            annotation_upper,
            random_bracket_y + 4.0 * bracket_h,
        )

    tick_labels = [
        f"Top {ratio * 100:g}%\n(K={int(cutoff_by_ratio.loc[ratio, 'actual_k'])})"
        for ratio in evaluation_ratios
    ]
    if metric_family == CAPACITY_RECALL_FAMILY:
        display_family = "Capacity-normalized Recall"
        y_label = "Capacity-normalized Recall (%)"
    else:
        display_family = (
            "MAP" if metric_family == "average_precision" else metric_family.upper()
        )
        y_label = display_family
    ax.set_xticks(centers, tick_labels)
    ax.set_xlabel("Screening depth (% of genes actually ranked)")
    ax.set_ylabel(y_label)
    fig.suptitle(
        f"{display_family} across three ranking strategies",
        y=0.965,
        fontsize=PLOT_FONT_SIZE_PT,
        weight="semibold",
    )
    if metric_family == CAPACITY_RECALL_FAMILY:
        # Keep the metric on its natural 0–100% scale while adding headroom
        # only when a high-valued group needs space for the two comparison
        # brackets and their q-value labels.
        ax.set_ylim(0.0, max(100.0, annotation_upper + 4.0))
        ax.set_yticks(np.arange(0.0, 100.1, 20.0))
    else:
        ax.set_ylim(lower, upper + 0.34 * span)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(
        handles=[
            Patch(facecolor=RRF_COLOR, edgecolor=RRF_COLOR, label="Multi-relation RRF"),
            Patch(
                facecolor=RAW_ESM2_COLOR,
                edgecolor=RAW_ESM2_EDGE_COLOR,
                label="Raw ESM-2",
            ),
            Patch(
                facecolor=RANDOM_COLOR,
                edgecolor=RANDOM_EDGE_COLOR,
                label="Uniform-random ranking",
            ),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=3,
        columnspacing=1.4,
        handlelength=1.5,
    )
    _add_figure_note(fig, figure_note)
    save_figure(fig, output_base)


def _three_way_enrichment_profile(
    rrf_repeats: pd.DataFrame,
    raw_repeats: pd.DataFrame,
    cutoff_rows: pd.DataFrame,
    evaluation_ratios: list[float],
    figure_note: str,
    output_base: Path,
) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(7.8, 5.4), constrained_layout=True)
    x = np.arange(len(evaluation_ratios), dtype=float)
    width = 0.30
    random_q_values = benjamini_hochberg(
        [
            _paired_greater_p_value(
                rrf_repeats[
                    percentage_metric("enrichment", ratio)
                ].to_numpy(dtype=float)
                - 1.0
            )
            for ratio in evaluation_ratios
        ]
    )
    plotted_upper = 1.0
    for method_index, (frame, offset, color, edge_color, label) in enumerate([
        (
            rrf_repeats,
            -width / 2,
            RRF_COLOR,
            "#6B6B6B",
            "Multi-relation RRF",
        ),
        (
            raw_repeats,
            width / 2,
            RAW_ESM2_COLOR,
            RAW_ESM2_EDGE_COLOR,
            "Raw ESM-2",
        ),
    ]):
        means, errors = [], []
        for ratio in evaluation_ratios:
            values = frame[percentage_metric("enrichment", ratio)].to_numpy(dtype=float)
            means.append(float(np.mean(values)))
            sem = float(np.std(values, ddof=1) / math.sqrt(len(values)))
            errors.append(1.96 * sem)
        bars = ax.bar(
            x + offset,
            means,
            yerr=errors,
            width=width,
            color=color,
            edgecolor=edge_color,
            linewidth=0.9,
            label=label,
            error_kw={"ecolor": "#333333", "elinewidth": 1.0, "capsize": 3},
        )
        plotted_upper = max(
            plotted_upper,
            max(value + error for value, error in zip(means, errors)),
        )
        for item_index, (bar, value) in enumerate(zip(bars, means)):
            annotation = f"{value:.1f}×"
            if method_index == 0:
                q_value = float(random_q_values[item_index])
                annotation += (
                    f"\n{_significance_stars(q_value)}  q={q_value:.2g}"
                )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                annotation,
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
    cutoff_by_ratio = cutoff_rows.set_index("evaluation_ratio")
    labels = [
        f"{ratio * 100:g}%\n(K={int(cutoff_by_ratio.loc[ratio, 'actual_k'])})"
        for ratio in evaluation_ratios
    ]
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.65, len(evaluation_ratios) - 0.45)
    ax.set_xlabel("Screening depth (% of genes actually ranked)")
    ax.set_ylabel("Recall enrichment over random (fold)")
    ax.set_title("Validation-gene enrichment across three ranking strategies", pad=10, weight="semibold")
    ax.set_ylim(0.0, plotted_upper * 1.20)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    _add_figure_note(fig, figure_note)
    save_figure(fig, output_base)


def _three_way_rank_boxplot(
    rrf_repeats: pd.DataFrame,
    raw_repeats: pd.DataFrame,
    random_repeats: pd.DataFrame,
    rank_metric: str,
    raw_q_value: float,
    random_q_value: float,
    figure_note: str,
    output_base: Path,
) -> None:
    rrf_values = rrf_repeats[rank_metric].to_numpy(dtype=float)
    raw_values = raw_repeats[rank_metric].to_numpy(dtype=float)
    random_values = random_repeats[rank_metric].to_numpy(dtype=float)
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    positions = [0.8, 1.6, 2.4]
    boxes = ax.boxplot(
        [rrf_values, raw_values, random_values],
        positions=positions,
        widths=0.36,
        patch_artist=True,
        showfliers=False,
        boxprops={"edgecolor": "#333333", "linewidth": 1.1},
        medianprops={"color": "#111111", "linewidth": 1.7},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
    )
    for patch, color in zip(
        boxes["boxes"], [RRF_COLOR, RAW_ESM2_COLOR, RANDOM_COLOR]
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)
    for position, sample in zip(positions, [rrf_values, raw_values, random_values]):
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
    ymax = max(float(rrf_values.max()), float(raw_values.max()), float(random_values.max()))
    ymin = min(float(rrf_values.min()), float(raw_values.min()), float(random_values.min()), 0.0)
    span = max(ymax - ymin, 0.1)
    raw_bracket_y = max(float(rrf_values.max()), float(raw_values.max())) + 0.06 * span
    bracket_h = 0.02 * span
    ax.plot(
        [positions[0], positions[0], positions[1], positions[1]],
        [
            raw_bracket_y,
            raw_bracket_y + bracket_h,
            raw_bracket_y + bracket_h,
            raw_bracket_y,
        ],
        color="#333333",
        linewidth=0.9,
    )
    ax.text(
        float(np.mean(positions[:2])),
        raw_bracket_y + 1.35 * bracket_h,
        f"RRF < raw: {_significance_stars(raw_q_value)}  q={raw_q_value:.2g}",
        ha="center",
        va="bottom",
        fontsize=PLOT_FONT_SIZE_PT,
    )
    random_bracket_y = ymax + 0.15 * span
    ax.plot(
        [positions[0], positions[0], positions[2], positions[2]],
        [
            random_bracket_y,
            random_bracket_y + bracket_h,
            random_bracket_y + bracket_h,
            random_bracket_y,
        ],
        color="#333333",
        linewidth=0.9,
    )
    ax.text(
        float(np.mean([positions[0], positions[2]])),
        random_bracket_y + 1.35 * bracket_h,
        f"RRF < random: {_significance_stars(random_q_value)}  q={random_q_value:.2g}",
        ha="center",
        va="bottom",
        fontsize=PLOT_FONT_SIZE_PT,
    )
    ax.set_xticks(
        positions,
        ["Validation genes\n(Multi-relation RRF)", "Validation genes\n(Raw ESM-2)", "Uniform-random\nranking"],
    )
    ax.set_xlim(0.35, 2.85)
    display = "Mean" if rank_metric == "normalized_mean_rank" else "Median"
    ax.set_ylabel(f"Normalized {display.lower()} validation-gene rank")
    ax.set_title(
        f"Normalized validation-gene {display.lower()} rank",
        pad=10,
        weight="semibold",
    )
    ax.set_ylim(ymin, random_bracket_y + 0.20 * span)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    _add_figure_note(fig, figure_note)
    save_figure(fig, output_base)


def generate_baseline_comparison_figures(
    task: str,
    experiment_dir: Path,
) -> Path:
    """Generate enhanced 04-09 plots with a matched raw ESM-2 baseline."""
    analysis_dir = experiment_dir / "analysis"
    tables_dir = analysis_dir / "tables"
    report_path = analysis_dir / "analysis_report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"completed analysis report is required before figures_bc_v3: {report_path}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = report.get("selected_aggregation", {})
    selected_seed_ratio = float(selected["seed_ratio"])
    selected_top_ratio = float(selected["similarity_top_ratio"])
    evaluation_ratios = sorted(float(value) for value in report["evaluation_ratios"])
    selected_run_id = str(selected["run_id"])
    selected_run_summary_path = (
        experiment_dir / "runs" / selected_run_id / "run_summary.json"
    )
    if not selected_run_summary_path.exists():
        raise FileNotFoundError(
            f"selected run summary is required for matched splits: {selected_run_summary_path}"
        )
    selected_run_summary = json.loads(
        selected_run_summary_path.read_text(encoding="utf-8")
    )
    selected_parameters = selected_run_summary["parameters"]
    split_seed = int(selected_parameters["split_seed"])
    rrf_k = float(selected_parameters["rrf_k"])
    num_repeats = int(selected_parameters["num_repeats"])

    all_rrf_repeats = pd.read_csv(tables_dir / "rrf_metrics_by_repeat_percentage.csv")
    rrf_repeats = all_rrf_repeats[
        np.isclose(all_rrf_repeats.seed_ratio, selected_seed_ratio)
        & np.isclose(all_rrf_repeats.similarity_top_ratio, selected_top_ratio)
    ].copy()
    if "method" in rrf_repeats.columns:
        rrf_repeats = rrf_repeats[rrf_repeats.method == "rrf"].copy()
    rrf_repeats = rrf_repeats.sort_values("repeat").reset_index(drop=True)
    if rrf_repeats.empty:
        raise RuntimeError("selected multi-relation RRF repeat metrics are absent")
    if int(rrf_repeats.repeat.nunique()) != num_repeats:
        raise RuntimeError(
            "selected RRF repeat table does not match its recorded run parameters"
        )

    cutoff_table = pd.read_csv(tables_dir / "percentage_cutoff_definitions.csv")
    cutoff_rows = cutoff_table[
        np.isclose(cutoff_table.seed_ratio, selected_seed_ratio)
    ].copy()
    cutoff_rows = cutoff_rows[
        cutoff_rows.evaluation_ratio.apply(
            lambda value: any(np.isclose(float(value), ratio) for ratio in evaluation_ratios)
        )
    ].sort_values("evaluation_ratio")
    if len(cutoff_rows) != len(evaluation_ratios):
        raise RuntimeError("selected percentage cutoff definitions are incomplete")
    ks = tuple(int(value) for value in cutoff_rows.actual_k)
    rrf_repeats = _ensure_capacity_recall_metrics(rrf_repeats, cutoff_rows)

    raw_context = _raw_esm2_context(SCRIPT_DIR / task / "prepared_data")
    raw_metrics, _, _ = evaluate_repeated_splits(
        raw_context,
        EvaluationParameters(
            seed_ratio=selected_seed_ratio,
            similarity_top_ratio=selected_top_ratio,
            num_repeats=num_repeats,
            split_seed=split_seed,
            rrf_k=rrf_k,
            ks=ks,
        ),
    )
    raw_repeats = raw_metrics[raw_metrics.method == "raw_esm2"].copy()
    raw_repeats = _canonicalize_raw_metrics(raw_repeats, cutoff_rows)
    raw_repeats = _ensure_capacity_recall_metrics(raw_repeats, cutoff_rows)
    raw_repeats = raw_repeats.sort_values("repeat").reset_index(drop=True)
    _validate_matched_repeats(rrf_repeats, raw_repeats)

    random_path = tables_dir / "matched_uniform_random_repeat_metrics.csv"
    if random_path.exists():
        random_repeats = pd.read_csv(random_path).sort_values("repeat").reset_index(drop=True)
    else:
        random_repeats = simulate_uniform_random_repeats(
            rrf_repeats, evaluation_ratios
        )
    random_repeats = _ensure_capacity_recall_metrics(
        random_repeats,
        cutoff_rows,
    )
    _validate_matched_repeats(rrf_repeats, random_repeats)

    raw_output = raw_repeats.copy()
    raw_output.insert(0, "seed_ratio", selected_seed_ratio)
    raw_output.insert(1, "similarity_top_ratio", selected_top_ratio)
    atomic_write_csv(
        tables_dir / "raw_esm2_repeat_metrics_selected_configuration.csv",
        raw_output,
    )

    figures_dir = analysis_dir / "figures_bc_v3"
    figures_dir.mkdir(parents=True, exist_ok=True)
    first = cutoff_rows.iloc[0]
    figure_note = (
        f"Ranked genes N={int(first.ranking_count):,} | reference genes={int(first.seed_count)} | "
        f"validation genes={int(first.hidden_count)} | selected aggregation={selected_top_ratio:g} | "
        f"matched repeated splits={len(rrf_repeats)}"
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
        _three_way_metric_profile(
            rrf_repeats,
            raw_repeats,
            random_repeats,
            cutoff_rows,
            evaluation_ratios,
            family,
            figure_note,
            figures_dir / f"{figure_index:02d}_{file_label}",
        )
    _three_way_enrichment_profile(
        rrf_repeats,
        raw_repeats,
        cutoff_rows,
        evaluation_ratios,
        figure_note,
        figures_dir / "07_recall_enrichment_over_random",
    )
    rank_metrics = ["normalized_mean_rank", "normalized_median_rank"]
    raw_rank_q_values = benjamini_hochberg(
        [
            _paired_greater_p_value(
                raw_repeats[metric].to_numpy(dtype=float)
                - rrf_repeats[metric].to_numpy(dtype=float)
            )
            for metric in rank_metrics
        ]
    )
    random_rank_q_values = benjamini_hochberg(
        [
            _paired_greater_p_value(
                0.5 - rrf_repeats[metric].to_numpy(dtype=float)
            )
            for metric in rank_metrics
        ]
    )
    _three_way_rank_boxplot(
        rrf_repeats,
        raw_repeats,
        random_repeats,
        "normalized_mean_rank",
        float(raw_rank_q_values[0]),
        float(random_rank_q_values[0]),
        figure_note,
        figures_dir / "08_normalized_mean_rank",
    )
    _three_way_rank_boxplot(
        rrf_repeats,
        raw_repeats,
        random_repeats,
        "normalized_median_rank",
        float(raw_rank_q_values[1]),
        float(random_rank_q_values[1]),
        figure_note,
        figures_dir / "09_normalized_median_rank",
    )
    (figures_dir / "README_三类排序策略比较.txt").write_text(
        "\n".join(
            [
                "三类排序策略比较（04-09）",
                "=" * 64,
                "Multi-relation RRF：训练后的关系特异表征经等权RRF融合。",
                "Raw ESM-2：冻结ESM-2的480维原始表征，不经过关系投影头训练。",
                "Uniform-random ranking：与每次留出划分匹配的均匀随机排序。",
                "配色：蓝紫色=Multi-relation RRF；浅青色=Raw ESM-2；淡粉色=随机排序。",
                "每张图均输出PNG、PDF和SVG；字体统一为Arial 10pt。",
                "三种策略使用相同参考基因划分、候选集合、相似性聚合比例和评价深度。",
                "04使用Capacity-normalized Recall=Hits@K/min(K,验证基因数)，以0–100%显示；",
                "原始Recall继续保留在CSV结果表中。04-06同时展示RRF相对Raw ESM-2",
                "及相对精确随机期望的单侧Wilcoxon检验，",
                "并分别在三个评价深度内进行BH-FDR校正；07标注RRF相对随机水平的检验。",
                "08-09同时展示RRF相对Raw ESM-2及相对随机水平的排名检验，并在两个排名指标内校正。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return figures_dir


def collect_task_summary(
    task: str, experiment_dir: Path, model_run: str
) -> dict:
    manifest_path = SCRIPT_DIR / task / "prepared_data" / "dataset_manifest.json"
    report_path = experiment_dir / "analysis" / "analysis_report.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = {
        "task": task,
        "species": TASKS[task],
        "model_run": model_run,
        "known_retained": manifest["counts"]["known_retained"],
        "candidate_retained": manifest["counts"]["candidate_retained"],
        "known_candidate_overlap": manifest["counts"]["known_candidate_overlap_retained"],
        "final_unknown_candidates": manifest["counts"]["final_unknown_candidates"],
        "experiment_directory": str(experiment_dir.resolve()),
        "status": "missing_analysis_report",
    }
    if not report_path.exists():
        return row
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = report.get("selected_aggregation", {})
    protocol = report.get("primary_protocol", {})
    row.update(
        {
            "status": report.get("status"),
            "completed_configurations": report.get("completed_configurations"),
            "selected_seed_ratio": selected.get("seed_ratio"),
            "selected_similarity_top_ratio": selected.get("similarity_top_ratio"),
            "selection_metric": protocol.get("selection_metric"),
            "selected_score": selected.get("score"),
            "selected_run_id": selected.get("run_id"),
        }
    )
    return row


def main() -> None:
    args = parse_args()
    if args.max_parallel < 1 or args.num_repeats < 2:
        raise ValueError("max_parallel must be positive and num_repeats at least 2")
    if not GRID_SCRIPT.exists():
        raise FileNotFoundError(GRID_SCRIPT)
    selected_tasks = list(dict.fromkeys(args.tasks))
    failures = []
    summaries = []
    print(
        f"Model: {args.model_run}\nTasks: {', '.join(selected_tasks)}\n"
        f"Each task: {len(set(args.seed_ratios)) * len(set(args.similarity_top_ratios))} "
        f"grid configurations; {args.num_repeats} repeats/configuration; "
        f"{args.max_parallel} internal workers\n",
        flush=True,
    )
    for index, task in enumerate(selected_tasks, start=1):
        command, experiment_dir = build_command(
            args, task, allow_missing_prepared=args.dry_run
        )
        print(f"\n[{index}/{len(selected_tasks)}] {task}\nOutput: {experiment_dir}", flush=True)
        if args.dry_run:
            print(subprocess.list2cmdline(command), flush=True)
            continue
        return_code = 0
        if not args.baseline_comparison_only:
            return_code = run_and_tee(
                command, experiment_dir / "launcher.log", task
            )
        if return_code != 0:
            failures.append({"task": task, "return_code": return_code})
            print(f"FAILED: {task} (return code {return_code})", flush=True)
            if args.stop_on_error:
                break
        else:
            try:
                figures_bc = generate_baseline_comparison_figures(
                    task,
                    experiment_dir,
                )
                summary = collect_task_summary(task, experiment_dir, args.model_run)
                summary["baseline_comparison_figures"] = str(figures_bc.resolve())
                summaries.append(summary)
                print(
                    f"COMPLETED: {task}\nThree-way figures: {figures_bc}",
                    flush=True,
                )
            except Exception as exc:
                failures.append(
                    {
                        "task": task,
                        "stage": "figures_bc",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"FAILED: {task} figures_bc ({type(exc).__name__}: {exc})", flush=True)
                if args.stop_on_error:
                    break

    if not args.dry_run and summaries:
        summary_path = SCRIPT_DIR / "each_species_grid_summary.csv"
        pd.DataFrame(summaries).to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\nCross-task summary: {summary_path}", flush=True)
    if failures:
        failure_path = SCRIPT_DIR / "each_species_grid_failures.json"
        failure_path.write_text(
            json.dumps({"failures": failures}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(f"{len(failures)} task(s) failed; see {failure_path}")


if __name__ == "__main__":
    main()
