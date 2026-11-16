"""V3 comparison of self-fine-tuned, maize-transfer, raw ESM-2 and random rankings.

The task-local ``03_2`` launchers generate only a compact comparison analysis,
not another hyperparameter grid.  All four strategies use exactly the same
reference/validation splits and the configuration selected by the preceding
``ranking_grid.py`` experiment.

V3 writes PNG, PDF and SVG figures, uses Arial 10 pt throughout, applies the
approved five-color palette, and plots capacity-normalized Recall@K.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RANKING_DIR = SCRIPT_DIR.parent
if str(RANKING_DIR) not in sys.path:
    sys.path.insert(0, str(RANKING_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ranking_core import (  # noqa: E402
    EvaluationParameters,
    atomic_write_csv,
    benjamini_hochberg,
    evaluate_repeated_splits,
    prepare_ranking_context,
)
from ranking_grid import (  # noqa: E402
    CAPACITY_RECALL_FAMILY,
    percentage_metric,
    simulate_uniform_random_repeats,
)
from species_helpers import (  # noqa: E402
    _canonicalize_raw_metrics,
    _ensure_capacity_recall_metrics,
    _paired_greater_p_value,
    _raw_esm2_context,
    _significance_stars,
    _validate_matched_repeats,
)


TASKS = {
    "arabidopsis_TARGET": "Arabidopsis thaliana",
    "homo_BRCA": "Homo sapiens (BRCA)",
    "homo_LUAD": "Homo sapiens (LUAD)",
}
DEFAULT_SOURCE_EXPERIMENT = "ranking_hyperparameter_grid_v3"
DEFAULT_OUTPUT_NAME = "four_strategy_comparison_v3"
METHODS = (
    "self_finetuned_rrf",
    "maize_transfer_rrf",
    "raw_esm2",
    "uniform_random",
)
METHOD_LABELS = {
    "self_finetuned_rrf": "Self-fine-tuned RRF",
    "maize_transfer_rrf": "Maize-transfer RRF",
    "raw_esm2": "Raw ESM-2",
    "uniform_random": "Uniform-random ranking",
}
COLORS = {
    "self_finetuned_rrf": "#5C6B9E",
    "maize_transfer_rrf": "#87CEEB",
    "raw_esm2": "#A8D5E5",
    "uniform_random": "#E6B8B8",
}
EDGES = {
    "self_finetuned_rrf": "#6B6B6B",
    "maize_transfer_rrf": "#6B6B6B",
    "raw_esm2": "#6B6B6B",
    "uniform_random": "#6B6B6B",
}
GRID_COLOR = "#D9D9D9"
PLOT_FONT_FAMILY = (
    "Arial" if Path("C:/Windows/Fonts/arial.ttf").exists() else "DejaVu Sans"
)
PLOT_FONT_SIZE_PT = 10
PLOT_DPI = 300


def configure_plot_style() -> None:
    """Apply the requested publication style to every v3 figure."""
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
    """Save matching high-resolution PNG plus vector PDF and SVG files."""
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


def _add_figure_note(fig: plt.Figure, note: str) -> None:
    """Add a readable, wrapped 10 pt note below the plot."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matched four-strategy ranking comparison for one species/task"
    )
    parser.add_argument("--task", choices=sorted(TASKS))
    parser.add_argument(
        "--relation-set",
        choices=["two", "five"],
        default="five",
        help="select model/four_strategy_comparison_v3 (two) or *_five_v3 outputs",
    )
    parser.add_argument("--self-model", type=Path)
    parser.add_argument("--maize-model", type=Path)
    parser.add_argument(
        "--task-dir", type=Path,
        help="task workspace containing ranking_results; defaults to the legacy layout",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        help="prepared ranking dataset; defaults to <task-dir>/prepared_data",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="explicit comparison output directory",
    )
    parser.add_argument("--source-experiment", default=DEFAULT_SOURCE_EXPERIMENT)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--projection-batch-size", type=int, default=1024)
    parser.add_argument("--num-repeats", type=int)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--rrf-k", type=float)
    parser.add_argument("--seed-ratio", type=float)
    parser.add_argument("--similarity-top-ratio", type=float)
    return parser.parse_args()


def newest_completed_self_model(task_dir: Path, relation_set: str) -> Path:
    model_folder = "model_five" if relation_set == "five" else "model"
    candidates = [
        path for path in (task_dir / model_folder / "all_output").glob(
            "*/protein_multihead.pt"
        )
        if (path.parent / "test_metrics.csv").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no completed self-fine-tuned model under "
            f"{task_dir / model_folder / 'all_output'}; run the matching "
            f"{'04_1' if relation_set == 'five' else '03_1'} fine-tuning script first"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def load_protocol(args: argparse.Namespace, task_dir: Path) -> dict:
    experiment_dir = task_dir / "ranking_results" / args.source_experiment
    config_path = experiment_dir / "metadata" / "experiment_config.json"
    report_path = experiment_dir / "analysis" / "analysis_report.json"
    cutoff_path = experiment_dir / "analysis" / "tables" / "percentage_cutoff_definitions.csv"
    if not config_path.is_file() or not report_path.is_file() or not cutoff_path.is_file():
        raise FileNotFoundError(
            f"completed 02 grid analysis is required under {experiment_dir}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = report["selected_aggregation"]
    seed_ratio = float(args.seed_ratio if args.seed_ratio is not None else selected["seed_ratio"])
    top_ratio = float(
        args.similarity_top_ratio
        if args.similarity_top_ratio is not None
        else selected["similarity_top_ratio"]
    )
    evaluation_ratios = sorted(float(value) for value in report["evaluation_ratios"])
    cutoffs = pd.read_csv(cutoff_path)
    cutoff_rows = cutoffs[np.isclose(cutoffs.seed_ratio, seed_ratio)].copy()
    cutoff_rows = cutoff_rows[
        cutoff_rows.evaluation_ratio.apply(
            lambda value: any(np.isclose(float(value), ratio) for ratio in evaluation_ratios)
        )
    ].sort_values("evaluation_ratio")
    if len(cutoff_rows) != len(evaluation_ratios):
        raise RuntimeError(
            f"no complete cutoff definition for reference-gene ratio {seed_ratio:g}"
        )
    maize_model = Path(args.maize_model or config["model_path"]).expanduser().resolve()
    if not maize_model.is_file():
        raise FileNotFoundError(f"maize-transfer model not found: {maize_model}")
    return {
        "experiment_dir": experiment_dir,
        "seed_ratio": seed_ratio,
        "top_ratio": top_ratio,
        "evaluation_ratios": evaluation_ratios,
        "cutoff_rows": cutoff_rows,
        "ks": tuple(int(value) for value in cutoff_rows.actual_k),
        "num_repeats": int(
            args.num_repeats if args.num_repeats is not None else config["num_repeats"]
        ),
        "split_seed": int(
            args.split_seed if args.split_seed is not None else config["split_seed"]
        ),
        "rrf_k": float(args.rrf_k if args.rrf_k is not None else config["rrf_k"]),
        "maize_model": maize_model,
    }


def rrf_metrics(context, parameters: EvaluationParameters, cutoffs: pd.DataFrame) -> pd.DataFrame:
    metrics, _, _ = evaluate_repeated_splits(context, parameters)
    frame = metrics[metrics.method == "rrf"].copy()
    frame = _canonicalize_raw_metrics(frame, cutoffs)
    return frame.sort_values("repeat").reset_index(drop=True)


def comparison_q_values(
    frames: dict[str, pd.DataFrame], metric: str, higher_is_better: bool
) -> list[float]:
    test_metric = metric
    capacity_prefix = f"{CAPACITY_RECALL_FAMILY}_at_"
    if metric.startswith(capacity_prefix):
        test_metric = metric.replace(capacity_prefix, "recall_at_", 1)
    self_values = frames["self_finetuned_rrf"][test_metric].to_numpy(dtype=float)
    p_values = []
    for baseline in METHODS[1:]:
        baseline_values = frames[baseline][test_metric].to_numpy(dtype=float)
        difference = (
            self_values - baseline_values
            if higher_is_better
            else baseline_values - self_values
        )
        p_values.append(_paired_greater_p_value(difference))
    return [float(value) for value in benjamini_hochberg(p_values)]


def plot_depth_profile(
    frames: dict[str, pd.DataFrame],
    cutoff_rows: pd.DataFrame,
    evaluation_ratios: list[float],
    metric_family: str,
    figure_note: str,
    output_base: Path,
) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(11.2, 6.6), constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.13, top=0.80)
    centers = np.arange(1, len(evaluation_ratios) + 1, dtype=float)
    offsets = np.linspace(-0.30, 0.30, len(METHODS))
    width = 0.17
    all_values = []
    grouped: dict[str, list[np.ndarray]] = {}
    for method, offset in zip(METHODS, offsets):
        values = [
            frames[method][percentage_metric(metric_family, ratio)].to_numpy(dtype=float)
            * (100.0 if metric_family == CAPACITY_RECALL_FAMILY else 1.0)
            for ratio in evaluation_ratios
        ]
        grouped[method] = values
        all_values.extend(values)
        boxes = ax.boxplot(
            values,
            positions=centers + offset,
            widths=width,
            patch_artist=True,
            showfliers=False,
            boxprops={"edgecolor": EDGES[method], "linewidth": 1.0},
            medianprops={"color": "#1A1A1A", "linewidth": 1.5},
            whiskerprops={"color": "#555555", "linewidth": 0.9},
            capprops={"color": "#555555", "linewidth": 0.9},
        )
        for patch in boxes["boxes"]:
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.82)

    if metric_family == CAPACITY_RECALL_FAMILY:
        lower = 0.0
        upper = 100.0
        span = 100.0
    else:
        lower = min(0.0, float(np.min(np.concatenate(all_values))))
        upper = float(np.max(np.concatenate(all_values)))
        span = max(upper - lower, 0.10)
    baseline_short = ["maize", "raw", "random"]
    for index, ratio in enumerate(evaluation_ratios):
        metric = percentage_metric(metric_family, ratio)
        q_values = comparison_q_values(frames, metric, higher_is_better=True)
        comparison_parts = [
            f"{name}: {_significance_stars(q)} q={q:.2g}"
            for name, q in zip(baseline_short, q_values)
        ]
        annotation = (
            f"Self > {comparison_parts[0]}\n"
            f"{comparison_parts[1]} | {comparison_parts[2]}"
        )
        if metric_family == CAPACITY_RECALL_FAMILY:
            ax.text(
                centers[index],
                0.985,
                annotation,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=PLOT_FONT_SIZE_PT,
                linespacing=1.15,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "#D9D9D9",
                    "alpha": 0.9,
                    "pad": 2,
                },
            )
        else:
            y = (
                max(float(values[index].max()) for values in grouped.values())
                + 0.07 * span
            )
            ax.text(
                centers[index],
                y,
                annotation,
                ha="center",
                va="bottom",
                fontsize=PLOT_FONT_SIZE_PT,
                linespacing=1.15,
            )

    cutoff_by_ratio = cutoff_rows.set_index("evaluation_ratio")
    ax.set_xticks(
        centers,
        [
            f"Top {ratio * 100:g}%\n(K={int(cutoff_by_ratio.loc[ratio, 'actual_k'])})"
            for ratio in evaluation_ratios
        ],
    )
    if metric_family == CAPACITY_RECALL_FAMILY:
        display = "Capacity-normalized Recall"
        y_label = "Capacity-normalized Recall (%)"
    else:
        display = "MAP" if metric_family == "average_precision" else metric_family.upper()
        y_label = display
    ax.set_xlabel("Screening depth (% of genes actually ranked)")
    ax.set_ylabel(y_label)
    fig.suptitle(
        f"{display} across four ranking strategies",
        y=0.965,
        fontsize=PLOT_FONT_SIZE_PT,
        weight="semibold",
    )
    if metric_family == CAPACITY_RECALL_FAMILY:
        ax.set_ylim(0.0, 100.0)
    else:
        ax.set_ylim(lower, upper + 0.34 * span)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(
        handles=[
            Patch(facecolor=COLORS[m], edgecolor=EDGES[m], label=METHOD_LABELS[m])
            for m in METHODS
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=4,
        fontsize=PLOT_FONT_SIZE_PT,
    )
    _add_figure_note(fig, figure_note)
    save_figure(fig, output_base)


def plot_rank_profile(
    frames: dict[str, pd.DataFrame],
    metric: str,
    figure_note: str,
    output_base: Path,
) -> None:
    values = [frames[method][metric].to_numpy(dtype=float) for method in METHODS]
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    positions = np.arange(1, len(METHODS) + 1, dtype=float)
    boxes = ax.boxplot(
        values,
        positions=positions,
        widths=0.50,
        patch_artist=True,
        showfliers=False,
        boxprops={"edgecolor": "#333333", "linewidth": 1.0},
        medianprops={"color": "#111111", "linewidth": 1.6},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
    )
    for patch, method in zip(boxes["boxes"], METHODS):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.82)
    for position, sample in zip(positions, values):
        ax.scatter(
            position, np.mean(sample), marker="D", s=36,
            facecolor="white", edgecolor="#111111", linewidth=0.8, zorder=4,
        )
    q_values = comparison_q_values(frames, metric, higher_is_better=False)
    ymax = max(float(sample.max()) for sample in values)
    ymin = min(0.0, min(float(sample.min()) for sample in values))
    span = max(ymax - ymin, 0.10)
    comparison_parts = [
        f"{name}: {_significance_stars(q)} q={q:.2g}"
        for name, q in zip(["maize", "raw", "random"], q_values)
    ]
    annotation = (
        f"Self < {comparison_parts[0]} | {comparison_parts[1]}\n"
        f"{comparison_parts[2]}"
    )
    ax.text(
        2.5,
        ymax + 0.09 * span,
        annotation,
        ha="center",
        va="bottom",
        fontsize=PLOT_FONT_SIZE_PT,
        linespacing=1.15,
    )
    display = "Mean" if metric == "normalized_mean_rank" else "Median"
    ax.set_xticks(
        positions,
        [
            "Self-fine-tuned\nRRF",
            "Maize-transfer\nRRF",
            "Raw ESM-2",
            "Uniform-random\nranking",
        ],
    )
    ax.set_ylabel(f"Normalized {display.lower()} validation-gene rank")
    ax.set_title(
        f"Normalized validation-gene {display.lower()} rank across four strategies",
        pad=22,
        weight="semibold",
    )
    ax.set_ylim(ymin, ymax + 0.23 * span)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    _add_figure_note(fig, figure_note)
    save_figure(fig, output_base)


def main() -> None:
    task_supplied = any(
        argument == "--task" or argument.startswith("--task=")
        for argument in sys.argv[1:]
    )
    help_requested = any(argument in {"-h", "--help"} for argument in sys.argv[1:])
    if not task_supplied and not help_requested:
        print(
            "No --task supplied; running all three five-relation comparisons in sequence.",
            flush=True,
        )
        for task in TASKS:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--task",
                task,
                *sys.argv[1:],
            ]
            print(f"\nStarting comparison task: {task}\n", flush=True)
            return_code = subprocess.call(command, cwd=str(SCRIPT_DIR.parents[1]))
            if return_code != 0:
                raise SystemExit(return_code)
        print("\nAll three four-strategy comparisons completed.\n", flush=True)
        return

    args = parse_args()
    if args.relation_set == "five":
        METHOD_LABELS["self_finetuned_rrf"] = "Self-fine-tuned five-head RRF"
    task_dir = (
        args.task_dir.expanduser().resolve()
        if args.task_dir is not None
        else SCRIPT_DIR / args.task
    )
    data_dir = (
        args.data_dir.expanduser().resolve()
        if args.data_dir is not None
        else task_dir / "prepared_data"
    )
    protocol = load_protocol(args, task_dir)
    self_model = (
        args.self_model.expanduser().resolve()
        if args.self_model is not None
        else newest_completed_self_model(task_dir, args.relation_set)
    )
    if not self_model.is_file():
        raise FileNotFoundError(f"self-fine-tuned model not found: {self_model}")
    parameters = EvaluationParameters(
        seed_ratio=protocol["seed_ratio"],
        similarity_top_ratio=protocol["top_ratio"],
        num_repeats=protocol["num_repeats"],
        split_seed=protocol["split_seed"],
        rrf_k=protocol["rrf_k"],
        ks=protocol["ks"],
    )
    print(
        f"\n{args.task}: matched four-strategy comparison\n"
        f"Self model: {self_model}\nMaize model: {protocol['maize_model']}\n"
        f"Reference ratio={protocol['seed_ratio']:g}, aggregation ratio={protocol['top_ratio']:g}, "
        f"repeats={protocol['num_repeats']}",
        flush=True,
    )
    self_context = prepare_ranking_context(
        model_path=self_model,
        data_dir=data_dir,
        requested_heads=None,
        device_name=args.device,
        projection_batch_size=args.projection_batch_size,
        allow_nonfull_data=True,
    )
    maize_context = prepare_ranking_context(
        model_path=protocol["maize_model"],
        data_dir=data_dir,
        requested_heads=None,
        device_name=args.device,
        projection_batch_size=args.projection_batch_size,
        allow_nonfull_data=True,
    )
    raw_context = _raw_esm2_context(data_dir)
    frames = {
        "self_finetuned_rrf": rrf_metrics(
            self_context, parameters, protocol["cutoff_rows"]
        ),
        "maize_transfer_rrf": rrf_metrics(
            maize_context, parameters, protocol["cutoff_rows"]
        ),
    }
    raw_metrics, _, _ = evaluate_repeated_splits(raw_context, parameters)
    frames["raw_esm2"] = _canonicalize_raw_metrics(
        raw_metrics[raw_metrics.method == "raw_esm2"].copy(),
        protocol["cutoff_rows"],
    ).sort_values("repeat").reset_index(drop=True)
    frames["uniform_random"] = simulate_uniform_random_repeats(
        frames["self_finetuned_rrf"], protocol["evaluation_ratios"]
    ).sort_values("repeat").reset_index(drop=True)
    for method in METHODS:
        frames[method] = _ensure_capacity_recall_metrics(
            frames[method],
            protocol["cutoff_rows"],
        )
    for method in METHODS[1:]:
        _validate_matched_repeats(frames["self_finetuned_rrf"], frames[method])

    output_name = args.output_name
    if args.relation_set == "five" and output_name == DEFAULT_OUTPUT_NAME:
        output_name = "four_strategy_comparison_five_v3"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else task_dir / "ranking_results" / output_name
    )
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(
        [frame.assign(strategy=method) for method, frame in frames.items()],
        ignore_index=True,
    )
    atomic_write_csv(tables_dir / "matched_four_strategy_metrics_by_repeat.csv", combined)
    first = protocol["cutoff_rows"].iloc[0]
    figure_note = (
        f"Ranked genes N={int(first.ranking_count):,} | reference genes={int(first.seed_count)} | "
        f"validation genes={int(first.hidden_count)} | aggregation={protocol['top_ratio']:g} | "
        f"matched repeated splits={protocol['num_repeats']} | equal-head RRF k={protocol['rrf_k']:g}"
    )
    for index, (family, label) in enumerate(
        [
            (
                CAPACITY_RECALL_FAMILY,
                "capacity_normalized_recall_profile",
            ),
            ("average_precision", "map_depth_profile"),
            ("ndcg", "ndcg_depth_profile"),
            ("enrichment", "recall_enrichment_over_random"),
        ],
        start=4,
    ):
        plot_depth_profile(
            frames,
            protocol["cutoff_rows"],
            protocol["evaluation_ratios"],
            family,
            figure_note,
            figures_dir / f"{index:02d}_{label}",
        )
    plot_rank_profile(
        frames, "normalized_mean_rank", figure_note,
        figures_dir / "08_normalized_mean_rank",
    )
    plot_rank_profile(
        frames, "normalized_median_rank", figure_note,
        figures_dir / "09_normalized_median_rank",
    )
    manifest = {
        "task": args.task,
        "species": TASKS[args.task],
        "relation_set": args.relation_set,
        "comparison": list(METHODS),
        "self_model": str(self_model),
        "self_heads": self_context.active_heads,
        "maize_model": str(protocol["maize_model"]),
        "maize_heads": maize_context.active_heads,
        "reference_gene_ratio": protocol["seed_ratio"],
        "similarity_aggregation_ratio": protocol["top_ratio"],
        "evaluation_ratios": protocol["evaluation_ratios"],
        "num_repeats": protocol["num_repeats"],
        "split_seed": protocol["split_seed"],
        "rrf_k": protocol["rrf_k"],
        "figure_formats": ["png", "pdf", "svg"],
        "png_dpi": PLOT_DPI,
        "font_family": PLOT_FONT_FAMILY,
        "font_size_pt": PLOT_FONT_SIZE_PT,
        "capacity_normalized_recall_definition": (
            "Hits@K / min(K, hidden validation-gene count); stored on 0-1 "
            "scale and plotted on 0-100% scale"
        ),
        "approved_palette": [
            "#E8A4A4",
            "#87CEEB",
            "#5C6B9E",
            "#E6B8B8",
            "#A8D5E5",
        ],
        "raw_esm2_fill_color": COLORS["raw_esm2"],
        "raw_esm2_edge_color": EDGES["raw_esm2"],
        "fairness_constraint": (
            "all strategies use identical reference/validation splits, candidate universe, "
            "aggregation ratio, screening depths and RRF constant"
        ),
    }
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nCompleted. V3 four-strategy figures: {figures_dir.resolve()}")


if __name__ == "__main__":
    main()
