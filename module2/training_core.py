"""Train the five-head protein-only multi-relation projection model (v3).

The ESM2 embeddings are fixed input features. Each epoch traverses every
training edge exactly once and updates only its matching relation head.
Validation selects the best epoch; the test split is evaluated exactly once
after restoring the best active-head model. Publication figures use Arial
10-point text and are exported as PNG, PDF, SVG, and TIFF files.
"""

import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

torch.backends.cudnn.benchmark = False

REPO = Path(__file__).resolve().parents[1]
PAIRS = REPO / "data/processed/pairs_train_v2.jsonl"
GENES = REPO / "data/processed/genes.parquet"
PROT_EMB = REPO / "data/processed/protein_emb.npy"
AVAILABLE_SOURCES = ("go", "coexp", "hic", "string", "tf")
DEFAULT_TRAINING_SOURCES = AVAILABLE_SOURCES
SOURCE_DISPLAY_NAMES = {
    "go": "GO",
    "coexp": "Coexp",
    "hic": "Hi-C",
    "string": "STRING",
    "tf": "TF",
}
INPUT_DIM = 480
MNRL_SCALE = 20.0
SAFE_NEGATIVE_TAU = 0.07
DEFAULT_ARCHITECTURE = "shared_residual"
DEFAULT_BATCH_SIZE = 32
DEFAULT_NEGATIVES_PER_ANCHOR = 49
DEFAULT_OUTPUT_DIM = 512
TASK_SAMPLING_STRATEGIES = (
    "proportional", "sqrt_frequency", "uniform_task")

PLOT_COLORS = {
    "blue": "#5C6B9E",
    "orange": "#E8A4A4",
    "green": "#A8D5E5",
    "vermillion": "#E6B8B8",
    "purple": "#87CEEB",
    "gray": "#4C566A",
    "axis": "#20242B",
    "muted": "#4F5968",
    "grid": "#D5DAE2",
    "light_grid": "#ECEFF3",
}
PLOT_DPI = 300
PLOT_FONT_FAMILY = (
    "Arial" if Path("C:/Windows/Fonts/arial.ttf").exists() else "DejaVu Sans"
)
PLOT_FONT_SIZE_PT = 10
PLOT_FONT_SIZE_PX = round(PLOT_FONT_SIZE_PT * PLOT_DPI / 72)
PLOT_FIGURE_SIZE_INCHES = (10.0, 20.0 / 3.0)

plt.rcParams.update({
    "font.family": PLOT_FONT_FAMILY,
    "font.sans-serif": [PLOT_FONT_FAMILY],
    "font.size": PLOT_FONT_SIZE_PT,
    "axes.titlesize": PLOT_FONT_SIZE_PT,
    "axes.labelsize": PLOT_FONT_SIZE_PT,
    "xtick.labelsize": PLOT_FONT_SIZE_PT,
    "ytick.labelsize": PLOT_FONT_SIZE_PT,
    "legend.fontsize": PLOT_FONT_SIZE_PT,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@contextmanager
def exclusive_file_lock(lock_path, poll_seconds=0.10):
    """Cross-process lock used when parallel runs update shared CSV files."""
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        lock_handle.seek(0, os.SEEK_END)
        if lock_handle.tell() == 0:
            lock_handle.write(b"0")
            lock_handle.flush()
        lock_handle.seek(0)
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(poll_seconds)
            try:
                yield
            finally:
                lock_handle.seek(0)
                msvcrt.locking(lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def load_plot_font(_size=None, bold=False):
    """Return a portable 10 pt font for a 300-DPI publication figure."""
    file_name = "arialbd.ttf" if bold else "arial.ttf"
    font_path = Path("C:/Windows/Fonts") / file_name
    if font_path.exists():
        return ImageFont.truetype(str(font_path), PLOT_FONT_SIZE_PX)
    fallback = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(fallback, PLOT_FONT_SIZE_PX)


def text_bounds(draw, text, font):
    if hasattr(draw, "textbbox"):
        return draw.textbbox((0, 0), text, font=font)
    if hasattr(font, "getbbox"):
        return font.getbbox(text)
    width, height = draw.textsize(text, font=font)
    return 0, 0, width, height


def text_width(draw, text, font):
    if hasattr(draw, "textlength"):
        return draw.textlength(text, font=font)
    bounds = text_bounds(draw, text, font)
    return bounds[2] - bounds[0]


def centered_text(draw, center_x, y, text, font, fill):
    bounds = text_bounds(draw, text, font)
    draw.text(
        (center_x - (bounds[2] - bounds[0]) / 2, y),
        text, fill=fill, font=font)


def vertical_text(draw, center_x, center_y, text, font, fill):
    bounds = text_bounds(draw, text, font)
    width = bounds[2] - bounds[0] + 12
    height = bounds[3] - bounds[1] + 12
    layer = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text((6, 6 - bounds[1]), text, fill=fill, font=font)
    resampling = getattr(Image, "Resampling", Image)
    rotated = layer.rotate(90, expand=True, resample=resampling.BICUBIC)
    position = (
        int(center_x - rotated.width / 2),
        int(center_y - rotated.height / 2),
    )
    draw.bitmap(position, rotated)


def draw_dashed_line(draw, points, fill, width=3, dash=14, gap=9):
    for start, end in zip(points[:-1], points[1:]):
        x1, y1 = start
        x2, y2 = end
        distance = max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1e-9)
        position = 0.0
        while position < distance:
            segment_end = min(position + dash, distance)
            ratio_start = position / distance
            ratio_end = segment_end / distance
            draw.line(
                (
                    x1 + (x2 - x1) * ratio_start,
                    y1 + (y2 - y1) * ratio_start,
                    x1 + (x2 - x1) * ratio_end,
                    y1 + (y2 - y1) * ratio_end,
                ),
                fill=fill,
                width=width,
            )
            position += dash + gap


def padded_limits(values, include_zero=False, padding=0.10):
    lower, upper = min(values), max(values)
    if lower == upper:
        delta = max(abs(lower) * 0.10, 0.1)
    else:
        delta = (upper - lower) * padding
    lower -= delta
    upper += delta
    if include_zero:
        lower = min(0.0, lower)
    return lower, upper


def draw_panel_header(draw, left, top, panel_label, title):
    draw.text(
        (left, top), panel_label, fill=PLOT_COLORS["axis"],
        font=load_plot_font(31, bold=True))
    draw.text(
        (left + 48, top), title, fill=PLOT_COLORS["axis"],
        font=load_plot_font(31, bold=True))


def draw_axes(draw, plot_box, y_min, y_max, y_label, x_labels=None):
    left, top, right, bottom = plot_box
    tick_font = load_plot_font(20)
    axis_font = load_plot_font(22, bold=True)
    for index in range(6):
        ratio = index / 5
        y = bottom - ratio * (bottom - top)
        value = y_min + ratio * (y_max - y_min)
        draw.line((left, y, right, y), fill=PLOT_COLORS["grid"], width=2)
        label = f"{value:.2f}" if abs(y_max) <= 10 else f"{value:.1f}"
        bounds = text_bounds(draw, label, tick_font)
        draw.text(
            (
                left - (bounds[2] - bounds[0]) - 18,
                y - (bounds[3] - bounds[1]) / 2 - bounds[1],
            ),
            label,
            fill=PLOT_COLORS["muted"], font=tick_font)
    draw.line((left, top, left, bottom), fill=PLOT_COLORS["axis"], width=3)
    draw.line((left, bottom, right, bottom), fill=PLOT_COLORS["axis"], width=3)
    vertical_text(
        draw,
        left - 78,
        (top + bottom) / 2,
        y_label,
        axis_font,
        PLOT_COLORS["muted"],
    )
    if x_labels:
        slot = (right - left) / len(x_labels)
        for index, label in enumerate(x_labels):
            centered_text(
                draw, left + (index + 0.5) * slot, bottom + 16, label,
                tick_font, PLOT_COLORS["muted"])


def draw_line_panel(
    draw, box, panel_label, title, x_values, series, y_min, y_max,
    y_label, best_epoch=None,
):
    left, top, right, bottom = box
    draw_panel_header(draw, left, top, panel_label, title)
    plot_box = (left + 112, top + 112, right - 22, bottom - 125)
    plot_left, plot_top, plot_right, plot_bottom = plot_box
    draw_axes(draw, plot_box, y_min, y_max, y_label)

    legend_font = load_plot_font(20)
    legend_x = plot_left
    for name, specification in series.items():
        values, color, dashed = specification
        sample_y = top + 69
        sample_points = [(legend_x, sample_y + 9), (legend_x + 40, sample_y + 9)]
        if dashed:
            draw_dashed_line(draw, sample_points, color, width=4, dash=10, gap=6)
        else:
            draw.line(sample_points, fill=color, width=4)
        draw.text(
            (legend_x + 50, sample_y - 4), name,
            fill=PLOT_COLORS["axis"], font=legend_font)
        legend_x += 68 + text_width(draw, name, legend_font)

    x_min, x_max = min(x_values), max(x_values)
    if x_min == x_max:
        x_max = x_min + 1
    tick_indices = sorted(set(
        int(round(value))
        for value in np.linspace(0, len(x_values) - 1, min(6, len(x_values)))
    ))
    tick_font = load_plot_font(20)
    for index in tick_indices:
        epoch = x_values[index]
        x = plot_left + (epoch - x_min) / (x_max - x_min) * (
            plot_right - plot_left)
        draw.line(
            (x, plot_top, x, plot_bottom),
            fill=PLOT_COLORS["light_grid"], width=2)
        centered_text(
            draw, x, plot_bottom + 15, str(epoch), tick_font,
            PLOT_COLORS["muted"])
    centered_text(
        draw, (plot_left + plot_right) / 2, bottom - 48, "Epoch",
        load_plot_font(22, bold=True), PLOT_COLORS["muted"])

    if best_epoch is not None and x_min <= best_epoch <= x_max:
        best_x = plot_left + (best_epoch - x_min) / (x_max - x_min) * (
            plot_right - plot_left)
        draw_dashed_line(
            draw, [(best_x, plot_top), (best_x, plot_bottom)],
            PLOT_COLORS["gray"], width=3, dash=12, gap=8)
        draw.text(
            (best_x + 8, plot_top + 7), f"best {best_epoch}",
            fill=PLOT_COLORS["gray"], font=load_plot_font(18, bold=True))

    marker_interval = max(1, len(x_values) // 12)
    for values, color, dashed in series.values():
        points = []
        for x_value, y_value in zip(x_values, values):
            x = plot_left + (x_value - x_min) / (x_max - x_min) * (
                plot_right - plot_left)
            clipped = min(y_max, max(y_min, y_value))
            y = plot_bottom - (clipped - y_min) / max(
                y_max - y_min, 1e-9) * (plot_bottom - plot_top)
            points.append((x, y))
        if dashed:
            draw_dashed_line(draw, points, color, width=4)
        elif len(points) > 1:
            draw.line(points, fill=color, width=4)
        for index in range(0, len(points), marker_interval):
            x, y = points[index]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        if points:
            x, y = points[-1]
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)


def save_matplotlib_figure(figure, path, export_tiff=False):
    """Save PNG previews and, when requested, publication-ready formats."""
    path = Path(path)
    figure.savefig(
        str(path), format="png", dpi=PLOT_DPI, facecolor="white")
    if export_tiff:
        figure.savefig(
            str(path.with_suffix(".pdf")), format="pdf", facecolor="white")
        figure.savefig(
            str(path.with_suffix(".svg")), format="svg", facecolor="white")
        figure.savefig(
            str(path.with_suffix(".tiff")), format="tiff",
            dpi=PLOT_DPI, facecolor="white")
    plt.close(figure)


def save_training_plot(
    history, path, sources, training_negative_count,
    evaluation_negative_count=None, export_tiff=False,
):
    if not history:
        return
    evaluation_negative_count = (
        training_negative_count if evaluation_negative_count is None
        else evaluation_negative_count)
    if training_negative_count == evaluation_negative_count:
        sampling_text = (
            f"1 positive + {training_negative_count} filtered negatives "
            "per query")
    else:
        sampling_text = (
            f"train: 1 + {training_negative_count} candidates; "
            f"validation: 1 + {evaluation_negative_count} candidates")

    epochs = np.asarray([row["epoch"] for row in history])
    best_epoch = min(history, key=lambda row: row["val_loss"])["epoch"]
    loss_values = [
        value for row in history
        for value in (row["train_loss"], row["val_loss"])
    ]
    loss_min, loss_max = padded_limits(loss_values)
    rank_values = [
        value for row in history
        for value in (row["val_mean_rank"], row["val_median_rank"])
    ]
    rank_min, rank_max = padded_limits(rank_values, include_zero=True)

    figure, axes = plt.subplots(
        2, 2, figsize=PLOT_FIGURE_SIZE_INCHES, dpi=PLOT_DPI)
    figure.subplots_adjust(
        left=0.09, right=0.98, bottom=0.13, top=0.80,
        hspace=0.48, wspace=0.30)
    figure.suptitle(
        "Training and validation trajectories",
        x=0.09, y=0.965, ha="left", fontweight="bold")
    figure.text(
        0.09, 0.915,
        f"Macro average across {len(sources)} active relation heads; "
        f"{sampling_text}",
        ha="left", va="top", color=PLOT_COLORS["muted"])

    def style_axis(axis, panel_label, title, y_label, y_limits):
        axis.set_title(
            f"{panel_label}   {title}", loc="left", fontweight="bold", pad=9)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(y_label)
        axis.set_ylim(*y_limits)
        axis.grid(
            axis="y", color=PLOT_COLORS["grid"], linewidth=0.7, alpha=0.9)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.axvline(
            best_epoch, color=PLOT_COLORS["gray"], linestyle="--",
            linewidth=1.0)
        axis.text(
            best_epoch, 0.98, f"best {best_epoch}",
            transform=axis.get_xaxis_transform(), ha="left", va="top",
            color=PLOT_COLORS["gray"], fontweight="bold")
        tick_indices = np.unique(np.linspace(
            0, len(epochs) - 1, min(6, len(epochs))).round().astype(int))
        axis.set_xticks(epochs[tick_indices])

    marker_interval = max(1, len(epochs) // 12)
    axis = axes[0, 0]
    axis.plot(
        epochs, [row["train_loss"] for row in history],
        color=PLOT_COLORS["blue"], linestyle="--", marker="o",
        markevery=marker_interval, markersize=3, linewidth=1.4, label="Train")
    axis.plot(
        epochs, [row["val_loss"] for row in history],
        color=PLOT_COLORS["vermillion"], marker="o",
        markevery=marker_interval, markersize=3, linewidth=1.4,
        label="Validation")
    style_axis(
        axis, "A", "Macro contrastive loss", "Cross-entropy loss",
        (loss_min, loss_max))
    axis.legend(frameon=False, ncol=2)

    axis = axes[0, 1]
    for label, key, color in (
        ("Hits@1", "val_hits1", PLOT_COLORS["blue"]),
        ("Hits@5", "val_hits5", PLOT_COLORS["green"]),
        ("Hits@10", "val_hits10", PLOT_COLORS["orange"]),
    ):
        axis.plot(
            epochs, [row[key] for row in history], color=color, marker="o",
            markevery=marker_interval, markersize=3, linewidth=1.4, label=label)
    style_axis(
        axis, "B", "Validation Hits@K", "Fraction of queries", (0.0, 1.0))
    axis.legend(frameon=False, ncol=3)

    axis = axes[1, 0]
    for label, key, color, linestyle in (
        ("Train MRR", "train_mrr", PLOT_COLORS["blue"], "--"),
        ("Valid MRR", "val_mrr", PLOT_COLORS["blue"], "-"),
        ("Train AUROC", "train_auroc", PLOT_COLORS["orange"], "--"),
        ("Valid AUROC", "val_auroc", PLOT_COLORS["orange"], "-"),
    ):
        axis.plot(
            epochs, [row[key] for row in history], color=color,
            linestyle=linestyle, marker="o", markevery=marker_interval,
            markersize=3, linewidth=1.4, label=label)
    style_axis(axis, "C", "Macro ranking quality", "Score", (0.0, 1.0))
    axis.legend(frameon=False, ncol=2)

    axis = axes[1, 1]
    for label, key, color in (
        ("Mean rank", "val_mean_rank", PLOT_COLORS["purple"]),
        ("Median rank", "val_median_rank", PLOT_COLORS["vermillion"]),
    ):
        axis.plot(
            epochs, [row[key] for row in history], color=color, marker="o",
            markevery=marker_interval, markersize=3, linewidth=1.4, label=label)
    style_axis(axis, "D", "Validation rank", "Rank", (rank_min, rank_max))
    axis.legend(frameon=False, ncol=2)

    figure.text(
        0.09, 0.035,
        "Dashed lines denote training estimates. AUPRC is omitted because, "
        "with one positive per query, it is mathematically identical to MRR.",
        ha="left", va="bottom", color=PLOT_COLORS["muted"])
    save_matplotlib_figure(figure, path, export_tiff=export_tiff)


def draw_grouped_bar_panel(
    draw, box, panel_label, title, labels, series, y_min, y_max, y_label,
    reference=None,
):
    left, top, right, bottom = box
    draw_panel_header(draw, left, top, panel_label, title)
    plot_box = (left + 112, top + 112, right - 22, bottom - 92)
    plot_left, plot_top, plot_right, plot_bottom = plot_box
    draw_axes(draw, plot_box, y_min, y_max, y_label, x_labels=labels)
    legend_font = load_plot_font(20)
    legend_x = plot_left
    for name, (_, color) in series.items():
        draw.rectangle(
            (legend_x, top + 70, legend_x + 27, top + 90), fill=color)
        draw.text(
            (legend_x + 38, top + 64), name,
            fill=PLOT_COLORS["axis"], font=legend_font)
        legend_x += 72 + text_width(draw, name, legend_font)
    if reference is not None:
        _, reference_label = reference
        draw_dashed_line(
            draw,
            [(legend_x, top + 80), (legend_x + 42, top + 80)],
            PLOT_COLORS["gray"],
            width=3,
            dash=10,
            gap=6,
        )
        draw.text(
            (legend_x + 52, top + 64), reference_label,
            fill=PLOT_COLORS["axis"], font=legend_font)

    if reference is not None:
        reference_value, _ = reference
        if y_min <= reference_value <= y_max:
            y = plot_bottom - (reference_value - y_min) / (y_max - y_min) * (
                plot_bottom - plot_top)
            draw_dashed_line(
                draw, [(plot_left, y), (plot_right, y)],
                PLOT_COLORS["gray"], width=3, dash=12, gap=8)

    names = list(series)
    slot = (plot_right - plot_left) / len(labels)
    group_width = slot * 0.72
    bar_width = group_width / len(names)
    for label_index in range(len(labels)):
        group_left = plot_left + label_index * slot + (slot - group_width) / 2
        for series_index, name in enumerate(names):
            values, color = series[name]
            value = values[label_index]
            bar_left = group_left + series_index * bar_width
            bar_right = bar_left + max(bar_width - 5, 2)
            clipped = min(y_max, max(y_min, value))
            bar_top = plot_bottom - (clipped - y_min) / max(
                y_max - y_min, 1e-9) * (plot_bottom - plot_top)
            draw.rectangle(
                (bar_left, bar_top, bar_right, plot_bottom), fill=color)


def save_test_plot(
    metrics, path, sources, negative_count, export_tiff=True,
):
    labels = [SOURCE_DISPLAY_NAMES[source] for source in sources] + ["Macro"]

    def values(metric):
        return [metrics["by_source"][source][metric] for source in sources] + [
            metrics["macro"][metric]
        ]

    figure, axes = plt.subplots(
        2, 2, figsize=PLOT_FIGURE_SIZE_INCHES, dpi=PLOT_DPI)
    figure.subplots_adjust(
        left=0.09, right=0.98, bottom=0.13, top=0.80,
        hspace=0.48, wspace=0.30)
    figure.suptitle(
        "Held-out test performance",
        x=0.09, y=0.965, ha="left", fontweight="bold")
    figure.text(
        0.09, 0.915,
        f"Best validation checkpoint; 1 positive + {negative_count} filtered "
        f"negatives per query ({negative_count + 1} candidates)",
        ha="left", va="top", color=PLOT_COLORS["muted"])

    x_positions = np.arange(len(labels))

    def grouped_bars(
        axis, panel_label, title, series, y_limits, y_label, reference=None,
    ):
        bar_width = 0.78 / len(series)
        group_start = x_positions - 0.39 + bar_width / 2
        for series_index, (label, series_values, color) in enumerate(series):
            axis.bar(
                group_start + series_index * bar_width,
                series_values,
                width=bar_width * 0.90,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                label=label,
            )
        axis.set_title(
            f"{panel_label}   {title}", loc="left", fontweight="bold", pad=9)
        axis.set_ylabel(y_label)
        axis.set_ylim(*y_limits)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(labels)
        axis.grid(
            axis="y", color=PLOT_COLORS["grid"], linewidth=0.7, alpha=0.9)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if reference is not None:
            reference_value, reference_label = reference
            axis.axhline(
                reference_value, color=PLOT_COLORS["gray"],
                linestyle="--", linewidth=1.0, label=reference_label)
        axis.legend(frameon=False, ncol=min(3, len(series) + int(
            reference is not None)))

    grouped_bars(
        axes[0, 0], "A", "Sampled Hits@K",
        (
            ("Hits@1", values("hits1"), PLOT_COLORS["blue"]),
            ("Hits@5", values("hits5"), PLOT_COLORS["green"]),
            ("Hits@10", values("hits10"), PLOT_COLORS["orange"]),
        ),
        (0.0, 1.0), "Fraction of queries")
    grouped_bars(
        axes[0, 1], "B", "Ranking discrimination",
        (
            ("MRR", values("mrr"), PLOT_COLORS["blue"]),
            ("AUROC", values("auroc"), PLOT_COLORS["orange"]),
        ),
        (0.0, 1.0), "Score",
        reference=(0.5, "random AUROC = 0.50"))
    rank_max = max(values("mean_rank") + values("median_rank")) * 1.16
    grouped_bars(
        axes[1, 0], "C", "Sampled positive rank",
        (
            ("Mean rank", values("mean_rank"), PLOT_COLORS["purple"]),
            (
                "Median rank", values("median_rank"),
                PLOT_COLORS["vermillion"],
            ),
        ),
        (0.0, rank_max), "Rank")
    losses = values("loss")
    random_loss = float(np.log(negative_count + 1))
    loss_max = max(max(losses), random_loss) * 1.16
    grouped_bars(
        axes[1, 1], "D", "Sampled cross-entropy loss",
        (("Loss", losses, PLOT_COLORS["vermillion"]),),
        (0.0, loss_max), "Cross-entropy loss",
        reference=(random_loss, f"random = ln({negative_count + 1})"))

    figure.text(
        0.09, 0.035,
        "Macro is the unweighted mean across active heads. AUPRC is omitted "
        "because it equals MRR under the one-positive query protocol.",
        ha="left", va="bottom", color=PLOT_COLORS["muted"])
    save_matplotlib_figure(figure, path, export_tiff=export_tiff)


class LinearProjection(nn.Module):
    """An independent linear projection for one relation."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, features):
        return F.normalize(self.projection(features), p=2, dim=-1)


class ResidualProjection(nn.Module):
    """An independent base and residual adapter for one relation."""

    def __init__(self, in_dim, out_dim, bottleneck, dropout):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.adapter = nn.Sequential(
            nn.Linear(out_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, out_dim),
            nn.Dropout(dropout),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features):
        base = self.base(features)
        output = base + self.residual_scale * self.adapter(base)
        return F.normalize(output, p=2, dim=-1)


class SharedBackbone(nn.Module):
    """The shared base transformation for shared_residual."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, features):
        return self.projection(features)


class SharedBackboneResidualHead(nn.Module):
    """A shared base followed by an independent relation adapter."""

    def __init__(self, shared_backbone, out_dim, bottleneck, dropout):
        super().__init__()
        self.shared_backbone = shared_backbone
        self.adapter = nn.Sequential(
            nn.Linear(out_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, out_dim),
            nn.Dropout(dropout),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features):
        shared = self.shared_backbone(features)
        output = shared + self.residual_scale * self.adapter(shared)
        return F.normalize(output, p=2, dim=-1)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def split_pairs_by_source(
    pairs_by_source, validation_ratio, test_ratio, seed, sources
):
    rng = random.Random(seed)
    train, validation, test = {}, {}, {}
    for source in sources:
        pairs = list(pairs_by_source[source])
        rng.shuffle(pairs)
        validation_count = max(1, int(round(len(pairs) * validation_ratio)))
        test_count = max(1, int(round(len(pairs) * test_ratio)))
        if validation_count + test_count >= len(pairs):
            raise ValueError(f"source '{source}' is too small for this split")
        test[source] = pairs[:test_count]
        validation[source] = pairs[
            test_count:test_count + validation_count]
        train[source] = pairs[test_count + validation_count:]
    return train, validation, test


def build_positive_neighbors(pairs_by_source):
    neighbors = defaultdict(set)
    for pairs in pairs_by_source.values():
        for gene_a, gene_b in pairs:
            neighbors[gene_a].add(gene_b)
            neighbors[gene_b].add(gene_a)
    return neighbors


def sample_safe_negative_matrix(
    anchors, negative_count, gene_count, positive_neighbors, rng
):
    """Sample a separate known-positive-filtered set for every anchor."""
    matrix = []
    for anchor in anchors:
        forbidden = positive_neighbors.get(anchor, set())
        selected = set()
        negatives = []
        while len(negatives) < negative_count:
            candidate = rng.randrange(gene_count)
            if (
                candidate != anchor
                and candidate not in forbidden
                and candidate not in selected
            ):
                selected.add(candidate)
                negatives.append(candidate)
        matrix.append(negatives)
    return matrix


def chunk_pairs(pairs, batch_size):
    chunks = [
        pairs[start:start + batch_size]
        for start in range(0, len(pairs), batch_size)
    ]
    if len(chunks) > 1 and len(chunks[-1]) == 1:
        chunks[-2].extend(chunks.pop())
    return chunks


def allocate_batch_counts(total_batches, weights, sources):
    weight_sum = sum(weights[source] for source in sources)
    raw = {
        source: total_batches * weights[source] / weight_sum
        for source in sources
    }
    counts = {source: int(raw[source]) for source in sources}
    remaining = total_batches - sum(counts.values())
    priority = sorted(
        sources,
        key=lambda source: (raw[source] - counts[source], -sources.index(source)),
        reverse=True,
    )
    for source in priority[:remaining]:
        counts[source] += 1
    return counts


def draw_cyclic_batch(pairs, size, state, rng):
    """Draw a batch without replacement until a shuffled pool is exhausted."""
    output = []
    while len(output) < size:
        if state["cursor"] >= len(state["pool"]):
            state["pool"] = list(pairs)
            rng.shuffle(state["pool"])
            state["cursor"] = 0
        take = min(size - len(output), len(state["pool"]) - state["cursor"])
        output.extend(state["pool"][state["cursor"]:state["cursor"] + take])
        state["cursor"] += take
    return output


def make_epoch_batches(
    train_pairs, batch_size, rng, sources, task_sampling="proportional"
):
    """Build one fixed-budget epoch under the requested task sampling rule."""
    schedule = []
    source_chunks = {
        source: chunk_pairs(list(train_pairs[source]), batch_size)
        for source in sources
    }
    if task_sampling == "proportional":
        for source in sources:
            pairs = list(train_pairs[source])
            rng.shuffle(pairs)
            schedule.extend(
                (source, batch) for batch in chunk_pairs(pairs, batch_size))
        rng.shuffle(schedule)
        return schedule

    batch_sizes = [
        len(batch) for source in sources for batch in source_chunks[source]
    ]
    total_batches = len(batch_sizes)
    if task_sampling == "sqrt_frequency":
        weights = {
            source: math.sqrt(len(train_pairs[source])) for source in sources
        }
    elif task_sampling == "uniform_task":
        weights = {source: 1.0 for source in sources}
    else:
        raise ValueError(f"unknown task sampling strategy: {task_sampling}")
    counts = allocate_batch_counts(total_batches, weights, list(sources))
    source_schedule = [
        source for source in sources for _ in range(counts[source])
    ]
    rng.shuffle(source_schedule)
    rng.shuffle(batch_sizes)
    states = {}
    for source in sources:
        pool = list(train_pairs[source])
        rng.shuffle(pool)
        states[source] = {"pool": pool, "cursor": 0}
    for source, current_batch_size in zip(source_schedule, batch_sizes):
        schedule.append((
            source,
            draw_cyclic_batch(
                train_pairs[source], current_batch_size,
                states[source], rng),
        ))
    rng.shuffle(schedule)
    return schedule


SAMPLED_METRICS = (
    "loss", "hits1", "hits5", "hits10", "mrr",
    "mean_rank", "median_rank", "auroc", "auprc",
)


class SampledMetricAccumulator:
    """Accumulate query-level metrics for one-positive sampled ranking."""

    def __init__(self):
        self.loss_sum = 0.0
        self.example_count = 0
        self.ranks = []
        self.auroc_sum = 0.0

    def update(self, loss, ranks, query_aurocs):
        ranks = np.asarray(ranks, dtype=np.float64)
        query_aurocs = np.asarray(query_aurocs, dtype=np.float64)
        count = int(ranks.size)
        self.loss_sum += float(loss) * count
        self.example_count += count
        self.ranks.extend(ranks.tolist())
        self.auroc_sum += float(query_aurocs.sum())

    def compute(self):
        if self.example_count == 0:
            raise RuntimeError("cannot compute sampled metrics without examples")
        ranks = np.asarray(self.ranks, dtype=np.float64)
        reciprocal = 1.0 / ranks
        mrr = float(reciprocal.mean())
        return {
            "loss": self.loss_sum / self.example_count,
            "hits1": float(np.mean(ranks <= 1)),
            "hits5": float(np.mean(ranks <= 5)),
            "hits10": float(np.mean(ranks <= 10)),
            "mrr": mrr,
            "mean_rank": float(ranks.mean()),
            "median_rank": float(np.median(ranks)),
            "auroc": self.auroc_sum / self.example_count,
            # With one relevant item per query, query-macro AP is 1/rank.
            "auprc": mrr,
            "examples": self.example_count,
        }


def macro_average(source_metrics, sources):
    return {
        name: float(np.mean([
            source_metrics[source][name] for source in sources
        ]))
        for name in SAMPLED_METRICS
    } | {
        "examples": int(sum(
            source_metrics[source]["examples"] for source in sources))
    }


def main():
    parser = argparse.ArgumentParser(
        description="Protein-only five-head contrastive training (v3)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=(
            "positive relation pairs per optimizer step "
            f"(default: {DEFAULT_BATCH_SIZE})"))
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--arch", choices=["linear", "residual", "shared_residual"],
        default=DEFAULT_ARCHITECTURE)
    parser.add_argument(
        "--sources", nargs="+", choices=AVAILABLE_SOURCES,
        default=list(DEFAULT_TRAINING_SOURCES),
        help=(
            "relation heads to train; default: "
            + " ".join(DEFAULT_TRAINING_SOURCES)))
    parser.add_argument("--output-dim", type=int, default=DEFAULT_OUTPUT_DIM)
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    negative_group = parser.add_mutually_exclusive_group()
    negative_group.add_argument(
        "--safe-negatives", dest="safe_negatives", action="store_true")
    negative_group.add_argument(
        "--in-batch-negatives", dest="safe_negatives", action="store_false")
    parser.set_defaults(safe_negatives=True)
    parser.add_argument(
        "--negatives-per-anchor", type=int,
        default=DEFAULT_NEGATIVES_PER_ANCHOR,
        help=(
            "independently sampled negatives per positive pair; independent "
            "of batch size (default: 49)"))
    parser.add_argument(
        "--evaluation-negatives-per-anchor", type=int, default=None,
        help=(
            "fixed negatives per validation/test query; default matches "
            "--negatives-per-anchor"))
    parser.add_argument(
        "--task-sampling", choices=TASK_SAMPLING_STRATEGIES,
        default="proportional",
        help=(
            "relation-task schedule; non-proportional strategies preserve "
            "the baseline optimizer-step and positive-presentation budgets"))

    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--patience", type=int, default=5,
        help="epochs without improvement; 0 disables early stopping")
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    naming_group = parser.add_mutually_exclusive_group()
    naming_group.add_argument(
        "--run-name", type=str, default=None,
        help=(
            "optional suffix appended to the automatic seed/time/settings "
            "run folder name"))
    naming_group.add_argument(
        "--run-id", type=str, default=None,
        help=(
            "exact non-timestamped run folder name; intended for controlled "
            "comparison experiments"))
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="existing run directory containing checkpoint.pt")
    parser.add_argument("--out", type=str,
                        default=str(REPO / "models/protein_multihead"))
    parser.add_argument("--pairs", type=str, default=str(PAIRS))
    parser.add_argument("--genes", type=str, default=str(GENES))
    parser.add_argument("--protein-embeddings", type=str, default=str(PROT_EMB))
    args = parser.parse_args()

    active_sources = tuple(dict.fromkeys(args.sources))
    if not active_sources:
        raise SystemExit("--sources must contain at least one relation")

    if args.epochs <= 0 or args.batch_size <= 1:
        raise SystemExit("--epochs must be positive and --batch-size must exceed 1")
    if args.negatives_per_anchor <= 0:
        raise SystemExit("--negatives-per-anchor must be positive")
    if (
        args.evaluation_negatives_per_anchor is not None
        and args.evaluation_negatives_per_anchor <= 0
    ):
        raise SystemExit("--evaluation-negatives-per-anchor must be positive")
    if args.output_dim <= 0 or args.bottleneck <= 0:
        raise SystemExit("--output-dim and --bottleneck must be positive")
    if args.val_ratio <= 0 or args.test_ratio <= 0:
        raise SystemExit("--val-ratio and --test-ratio must be positive")
    if args.val_ratio + args.test_ratio >= 1:
        raise SystemExit("validation and test ratios must sum to less than 1")
    if args.patience < 0 or args.min_delta < 0:
        raise SystemExit("--patience and --min-delta cannot be negative")

    negatives_per_anchor = args.negatives_per_anchor
    evaluation_negatives_per_anchor = (
        args.negatives_per_anchor
        if args.evaluation_negatives_per_anchor is None
        else args.evaluation_negatives_per_anchor)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_root = Path(args.out).expanduser().resolve()
    all_output_dir = output_root / "all_output"
    if args.resume_from:
        output_dir = Path(args.resume_from).expanduser().resolve()
        if not output_dir.is_dir():
            raise FileNotFoundError(
                f"resume run directory not found: {output_dir}")
        if output_dir.parent.name == "all_output":
            all_output_dir = output_dir.parent
            output_root = all_output_dir.parent
    else:
        requested_name = args.run_id or args.run_name
        if requested_name:
            invalid_name = any(
                character in '<>:"/\\|?*' for character in requested_name)
            if (
                invalid_name
                or Path(requested_name).name != requested_name
                or requested_name in {".", ".."}
            ):
                raise SystemExit(
                    "--run-name/--run-id must be one filename-safe folder name")
        if args.run_id:
            run_name = args.run_id
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            source_tag = "-".join(active_sources)
            run_name = (
                f"seed{args.seed}_{timestamp}_{args.arch}_dim{args.output_dim}_"
                f"bs{args.batch_size}_neg{negatives_per_anchor}_"
                f"lr{args.lr:g}_src-{source_tag}_v3")
            if args.run_name:
                run_name = f"{run_name}_{args.run_name}"
        all_output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = all_output_dir / run_name
        output_dir.mkdir(parents=False, exist_ok=False)

    log_path = output_dir / "training.log"
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)

    def emit(message=""):
        print(message, flush=True)
        print(message, file=log_handle, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs_path = resolve_path(args.pairs)
    genes_path = resolve_path(args.genes)
    embeddings_path = resolve_path(args.protein_embeddings)

    genes = pd.read_parquet(genes_path)
    gene_ids = genes["id"].astype(str).tolist()
    if len(gene_ids) != len(set(gene_ids)):
        raise ValueError("genes.parquet contains duplicate gene IDs")
    gene_to_index = {gene_id: index for index, gene_id in enumerate(gene_ids)}

    protein_array = np.load(embeddings_path).astype("float32")
    expected_shape = (len(gene_ids), INPUT_DIM)
    if protein_array.shape != expected_shape:
        raise ValueError(
            f"protein embedding shape mismatch: expected {expected_shape}, "
            f"got {tuple(protein_array.shape)}")
    norms = np.linalg.norm(protein_array, axis=1, keepdims=True)
    zero_vector_count = int((norms.squeeze(1) <= 1e-9).sum())
    protein_array /= np.clip(norms, 1e-9, None)
    protein_embeddings = torch.from_numpy(protein_array).to(device)

    all_pairs = defaultdict(list)
    skipped_records = 0
    with pairs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            source = record.get("source")
            anchor = record.get("anchor")
            positive = record.get("positive")
            if (
                source not in AVAILABLE_SOURCES
                or anchor not in gene_to_index
                or positive not in gene_to_index
            ):
                skipped_records += 1
                continue
            all_pairs[source].append(
                (gene_to_index[anchor], gene_to_index[positive]))
    for source in active_sources:
        if len(all_pairs[source]) < 3:
            raise ValueError(f"source '{source}' needs at least three usable edges")

    # Only enabled relation heads need train/validation/test splits.  Older
    # runs enabled every available source, so this is behaviour-preserving for
    # the maize five-head setting while also allowing species-specific runs
    # whose JSONL contains only a subset (for example GO and coexpression).
    present_sources = tuple(
        source for source in AVAILABLE_SOURCES if len(all_pairs[source]) >= 3
    )

    all_train_pairs, all_validation_pairs, all_test_pairs = (
        split_pairs_by_source(
            all_pairs,
            args.val_ratio,
            args.test_ratio,
            args.split_seed,
            present_sources,
        )
    )
    train_pairs = {
        source: all_train_pairs[source] for source in active_sources
    }
    validation_pairs = {
        source: all_validation_pairs[source] for source in active_sources
    }
    test_pairs = {
        source: all_test_pairs[source] for source in active_sources
    }
    train_count = sum(len(train_pairs[source]) for source in active_sources)
    validation_count = sum(
        len(validation_pairs[source]) for source in active_sources)
    test_count = sum(len(test_pairs[source]) for source in active_sources)
    # Keep every usable known relation in the exclusion graph, including
    # sources that are present but not optimized in this run. This prevents an
    # excluded-source positive edge from being sampled as an obvious negative
    # for another head.
    train_neighbors = build_positive_neighbors(all_train_pairs)
    all_neighbors = build_positive_neighbors(all_pairs)

    shared_backbone = None
    if args.arch == "shared_residual":
        shared_backbone = SharedBackbone(INPUT_DIM, args.output_dim)

    def make_projection():
        if args.arch == "linear":
            return LinearProjection(INPUT_DIM, args.output_dim)
        if args.arch == "residual":
            return ResidualProjection(
                INPUT_DIM, args.output_dim, args.bottleneck, args.dropout)
        return SharedBackboneResidualHead(
            shared_backbone, args.output_dim, args.bottleneck, args.dropout)

    heads = nn.ModuleDict({
        source: make_projection() for source in active_sources
    })
    heads.to(device)

    parameters = []
    seen_parameter_ids = set()
    for parameter in heads.parameters():
        if id(parameter) not in seen_parameter_ids:
            seen_parameter_ids.add(id(parameter))
            parameters.append(parameter)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    parameter_count = sum(parameter.numel() for parameter in parameters)

    def contrastive_loss_and_sampled_metrics(head, pairs, negative_matrix):
        anchors = torch.tensor([pair[0] for pair in pairs], device=device)
        positives = torch.tensor([pair[1] for pair in pairs], device=device)
        anchor_vectors = head(protein_embeddings[anchors])
        positive_vectors = head(protein_embeddings[positives])
        if negative_matrix is None:
            logits = torch.matmul(anchor_vectors, positive_vectors.T) * MNRL_SCALE
            labels = torch.arange(logits.shape[0], device=device)
        else:
            negative_indices = torch.tensor(
                negative_matrix, device=device, dtype=torch.long)
            batch_size, negative_count = negative_indices.shape
            negative_vectors = head(
                protein_embeddings[negative_indices.reshape(-1)]
            ).reshape(batch_size, negative_count, -1)
            positive_scores = (
                anchor_vectors * positive_vectors
            ).sum(dim=-1) / SAFE_NEGATIVE_TAU
            negative_scores = (
                anchor_vectors.unsqueeze(1) * negative_vectors
            ).sum(dim=-1) / SAFE_NEGATIVE_TAU
            logits = torch.cat(
                [positive_scores[:, None], negative_scores], dim=1)
            labels = torch.zeros(batch_size, device=device, dtype=torch.long)

        loss = F.cross_entropy(logits, labels)
        target_scores = logits.gather(1, labels[:, None])
        candidate_mask = torch.ones_like(logits, dtype=torch.bool)
        candidate_mask.scatter_(1, labels[:, None], False)
        competitors = logits[candidate_mask].reshape(logits.shape[0], -1)
        strictly_better = (competitors > target_scores).sum(dim=1)
        tied = (competitors == target_scores).sum(dim=1)
        ranks = 1 + strictly_better + tied
        query_aurocs = (
            (target_scores > competitors).float()
            + 0.5 * (target_scores == competitors).float()
        ).mean(dim=1)
        return (
            loss,
            ranks.detach().cpu().numpy(),
            query_aurocs.detach().cpu().numpy(),
        )

    def evaluate(evaluation_pairs, seed, description=None):
        heads.eval()
        rng = random.Random(seed)
        accumulators = {
            source: SampledMetricAccumulator() for source in active_sources
        }
        total_batches = sum(
            len(chunk_pairs(list(evaluation_pairs[source]), args.batch_size))
            for source in active_sources)
        progress = tqdm(
            total=total_batches,
            desc=description or "Evaluation",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            mininterval=0.5,
            ascii=True,
        )
        with torch.no_grad():
            for source in active_sources:
                for pairs in chunk_pairs(
                    list(evaluation_pairs[source]), args.batch_size
                ):
                    negatives = None
                    if args.safe_negatives:
                        negatives = sample_safe_negative_matrix(
                            [pair[0] for pair in pairs],
                            evaluation_negatives_per_anchor,
                            len(gene_ids),
                            all_neighbors,
                            rng,
                        )
                    loss, ranks, query_aurocs = (
                        contrastive_loss_and_sampled_metrics(
                        heads[source], pairs, negatives)
                    )
                    accumulators[source].update(
                        loss.item(), ranks, query_aurocs)
                    progress.update(1)
        progress.close()

        heads.train()
        by_source = {
            source: accumulators[source].compute() for source in active_sources
        }
        macro = macro_average(by_source, active_sources)
        return {
            "selection_loss": macro["loss"],
            "source_loss": macro["loss"],
            "source_accuracy": macro["hits1"],
            "loss_by_source": {
                source: by_source[source]["loss"] for source in active_sources
            },
            "accuracy_by_source": {
                source: by_source[source]["hits1"] for source in active_sources
            },
            "by_source": by_source,
            "macro": macro,
        }

    total_head_updates = Counter()

    def model_payload(epoch, validation_metrics, selection, test_metrics=None):
        return {
            "heads": heads.state_dict(),
            "shared_backbone": (
                shared_backbone.state_dict()
                if shared_backbone is not None else None),
            "config": {
                "trainer_script": str(Path(__file__).resolve()),
                "epochs_requested": args.epochs,
                "epoch_saved": epoch,
                "sources": list(active_sources),
                "available_sources": list(AVAILABLE_SOURCES),
                "excluded_sources": [
                    source for source in AVAILABLE_SOURCES
                    if source not in active_sources
                ],
                "input_dim": INPUT_DIM,
                "output_dim": args.output_dim,
                "architecture": args.arch,
                "parameter_count": parameter_count,
                "bottleneck": args.bottleneck,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "seed": args.seed,
                "split_seed": args.split_seed,
                "split_type": "per_source_random_edge_holdout_transductive",
                "validation_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "model_selection": selection,
                "early_stopping_patience": args.patience,
                "safe_negatives": args.safe_negatives,
                "negatives_per_anchor": negatives_per_anchor,
                "evaluation_negatives_per_anchor": (
                    evaluation_negatives_per_anchor),
                "task_sampling": args.task_sampling,
                "epoch_semantics": (
                    "every training edge exactly once"
                    if args.task_sampling == "proportional"
                    else (
                        "fixed baseline optimizer-step and positive-presentation "
                        "budgets with task resampling")),
                "loss": "relation-specific contrastive loss",
                "sampled_metrics": list(SAMPLED_METRICS),
                "sampled_metric_protocol": (
                    "one positive plus independently filtered negatives per query"),
                "sampled_auroc": "query-macro pairwise AUROC",
                "sampled_auprc": (
                    "query-macro average precision; equals MRR with one positive"),
                "pairs": str(pairs_path),
                "genes": str(genes_path),
                "protein_embeddings": str(embeddings_path),
                "train_pair_count": train_count,
                "validation_pair_count": validation_count,
                "test_pair_count": test_count,
                "head_updates": dict(total_head_updates),
                "run_id": output_dir.name,
                "run_directory": str(output_dir.resolve()),
                "training_log": str(log_path.resolve()),
            },
        }

    def print_metrics(title, metrics, head_updates=None):
        emit(f"\n{title}")
        emit(
            f"{'head':<8} {'loss':>7} {'H@1':>6} {'H@5':>6} "
            f"{'H@10':>6} {'MRR':>6} {'MeanR':>7} {'MedR':>6} "
            f"{'AUROC':>7} {'AUPRC':>7}")
        for source in active_sources:
            item = metrics["by_source"][source]
            emit(format_metric_row(source, item))
        emit(format_metric_row("macro", metrics["macro"]))
        if head_updates is not None:
            updates = "  ".join(
                f"{source}={int(head_updates.get(source, 0)):,}"
                for source in active_sources)
            emit(f"updates  {updates}")

    def format_metric_row(label, item, split=None):
        prefix = f"{label:<8}"
        if split is not None:
            prefix += f" {split:<5}"
        return (
            f"{prefix} {item['loss']:>7.4f} {item['hits1']:>6.3f} "
            f"{item['hits5']:>6.3f} {item['hits10']:>6.3f} "
            f"{item['mrr']:>6.3f} {item['mean_rank']:>7.2f} "
            f"{item['median_rank']:>6.1f} {item['auroc']:>7.3f} "
            f"{item['auprc']:>7.3f}")

    def print_epoch_metrics(
        epoch, train_metrics, validation_metrics, status, wait, seconds
    ):
        emit(
            f"\nEpoch {epoch:03d}/{args.epochs}  |  {status.upper()}  |  "
            f"early-stop wait {wait}/{args.patience}  |  "
            f"{format_duration(seconds)}")
        emit(
            f"{'head':<8} {'split':<5} {'loss':>7} {'H@1':>6} {'H@5':>6} "
            f"{'H@10':>6} {'MRR':>6} {'MeanR':>7} {'MedR':>6} "
            f"{'AUROC':>7} {'AUPRC':>7}")
        emit("-" * 98)
        for source in active_sources:
            emit(format_metric_row(
                source, train_metrics["by_source"][source], "train"))
            emit(format_metric_row(
                "", validation_metrics["by_source"][source], "valid"))
        emit("-" * 98)
        emit(format_metric_row("macro", train_metrics["macro"], "train"))
        emit(format_metric_row("", validation_metrics["macro"], "valid"))

    best_path = output_dir / "protein_multihead.pt"
    last_path = output_dir / "protein_multihead_last.pt"
    checkpoint_path = output_dir / "checkpoint.pt"
    epoch_csv_path = output_dir / "epoch_metrics.csv"
    test_csv_path = output_dir / "test_metrics.csv"
    training_plot_path = output_dir / "training_curves.png"
    test_plot_path = output_dir / "test_performance.png"
    training_plot_pdf_path = training_plot_path.with_suffix(".pdf")
    training_plot_svg_path = training_plot_path.with_suffix(".svg")
    training_plot_tiff_path = training_plot_path.with_suffix(".tiff")
    test_plot_pdf_path = test_plot_path.with_suffix(".pdf")
    test_plot_svg_path = test_plot_path.with_suffix(".svg")
    test_plot_tiff_path = test_plot_path.with_suffix(".tiff")
    run_config_path = output_dir / "run_config.json"

    epoch_fields = [
        "epoch", "improved", "early_stop_wait", "epoch_seconds",
    ]
    for split in ("train", "val"):
        for metric in SAMPLED_METRICS:
            epoch_fields.append(f"{split}_macro_{metric}")
        for source in active_sources:
            for metric in SAMPLED_METRICS:
                epoch_fields.append(f"{split}_{source}_{metric}")
    epoch_history = []
    if args.resume_from and epoch_csv_path.exists():
        with epoch_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not set(epoch_fields).issubset(set(reader.fieldnames or [])):
                raise RuntimeError(
                    "resume run uses an incompatible epoch_metrics.csv schema")
            for row in reader:
                epoch_history.append({
                    "epoch": int(row["epoch"]),
                    **{
                        f"train_{metric}": float(
                            row[f"train_macro_{metric}"])
                        for metric in SAMPLED_METRICS
                    },
                    **{
                        f"val_{metric}": float(row[f"val_macro_{metric}"])
                        for metric in SAMPLED_METRICS
                    },
                })
    elif not args.resume_from:
        with epoch_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            csv.DictWriter(handle, fieldnames=epoch_fields).writeheader()

    run_config = {
        "run_id": output_dir.name,
        "run_directory": str(output_dir.resolve()),
        "trainer_script": str(Path(__file__).resolve()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "split_seed": args.split_seed,
        "architecture": args.arch,
        "parameter_count": parameter_count,
        "sources": list(active_sources),
        "available_sources": list(AVAILABLE_SOURCES),
        "excluded_sources": [
            source for source in AVAILABLE_SOURCES
            if source not in active_sources
        ],
        "input_dim": INPUT_DIM,
        "output_dim": args.output_dim,
        "bottleneck": args.bottleneck,
        "dropout": args.dropout,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "safe_negatives": args.safe_negatives,
        "negatives_per_anchor": negatives_per_anchor,
        "evaluation_negatives_per_anchor": evaluation_negatives_per_anchor,
        "task_sampling": args.task_sampling,
        "validation_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "train_pair_count": train_count,
        "validation_pair_count": validation_count,
        "test_pair_count": test_count,
        "sampled_metrics": list(SAMPLED_METRICS),
        "sampled_metric_protocol": (
            "one positive plus independently filtered negatives per query"),
        "sampled_auroc": "query-macro pairwise AUROC",
        "sampled_auprc": (
            "query-macro average precision; equals MRR with one positive"),
        "plot_settings": {
            "font_family": PLOT_FONT_FAMILY,
            "font_size_pt": PLOT_FONT_SIZE_PT,
            "dpi": PLOT_DPI,
            "formats": ["png", "pdf", "svg", "tiff"],
            "canvas_pixels": [
                round(PLOT_FIGURE_SIZE_INCHES[0] * PLOT_DPI),
                round(PLOT_FIGURE_SIZE_INCHES[1] * PLOT_DPI),
            ],
        },
        "pairs": str(pairs_path.resolve()),
        "genes": str(genes_path.resolve()),
        "protein_embeddings": str(embeddings_path.resolve()),
    }
    with run_config_path.open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2)

    start_epoch = 0
    best_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    if args.resume_from:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False)
        heads.load_state_dict(checkpoint["heads"])
        if shared_backbone is not None and checkpoint.get("shared_backbone"):
            shared_backbone.load_state_dict(checkpoint["shared_backbone"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        best_epoch = int(checkpoint["best_epoch"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        epochs_without_improvement = int(
            checkpoint["epochs_without_improvement"])
        total_head_updates.update(checkpoint.get("head_updates", {}))

    runtime_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")
    batches_per_epoch = sum(
        len(chunk_pairs(list(train_pairs[source]), args.batch_size))
        for source in active_sources)
    emit("SWORD protein multi-head training")
    emit(
        f"{args.arch}  {INPUT_DIM}->{args.output_dim}  |  "
        f"{parameter_count:,} parameters  |  {runtime_name}")
    emit(f"heads  {' '.join(active_sources)}")
    excluded_sources = [
        source for source in AVAILABLE_SOURCES if source not in active_sources
    ]
    if excluded_sources:
        emit(f"excluded  {' '.join(excluded_sources)}")
    emit(
        f"train {train_count:,}  validation {validation_count:,}  "
        f"test {test_count:,}  |  {batches_per_epoch:,} batches/epoch")
    emit(
        f"seed {args.seed}  split seed {args.split_seed}  |  "
        f"epochs {args.epochs}  batch {args.batch_size}  "
        f"train negatives/anchor {negatives_per_anchor}  "
        f"eval negatives/anchor {evaluation_negatives_per_anchor}")
    emit(f"task sampling  {args.task_sampling}")
    emit(f"run directory  {output_dir.resolve()}")
    if zero_vector_count or skipped_records:
        emit(
            f"warning: zero vectors {zero_vector_count:,}, "
            f"skipped records {skipped_records:,}")
    if start_epoch:
        emit(f"resume from epoch {start_epoch:,}")
    emit(
        "sampled metrics use one positive per query; query-macro AUPRC "
        "therefore equals MRR")

    training_start = time.time()
    last_epoch = start_epoch
    stopped_early = False
    for epoch_index in range(start_epoch, args.epochs):
        epoch = epoch_index + 1
        epoch_start = time.time()
        epoch_rng = random.Random(args.seed + epoch * 1_000_003)
        schedule = make_epoch_batches(
            train_pairs, args.batch_size, epoch_rng, active_sources,
            args.task_sampling)
        train_accumulators = {
            source: SampledMetricAccumulator() for source in active_sources
        }

        progress = tqdm(
            schedule,
            desc=f"Train {epoch:03d}/{args.epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            mininterval=0.5,
            ascii=True,
        )
        for batch_index, (source, pairs) in enumerate(progress, start=1):
            negatives = None
            if args.safe_negatives:
                negatives = sample_safe_negative_matrix(
                    [pair[0] for pair in pairs],
                    negatives_per_anchor,
                    len(gene_ids),
                    train_neighbors,
                    epoch_rng,
                )
            optimizer.zero_grad(set_to_none=True)
            source_loss, ranks, query_aurocs = (
                contrastive_loss_and_sampled_metrics(
                    heads[source], pairs, negatives)
            )
            source_loss.backward()
            optimizer.step()

            train_accumulators[source].update(
                source_loss.item(), ranks, query_aurocs)
            total_head_updates[source] += 1
            if batch_index % 50 == 0:
                progress.set_postfix(loss=f"{source_loss.item():.4f}")

        train_by_source = {
            source: train_accumulators[source].compute()
            for source in active_sources
        }
        train_metrics = {
            "by_source": train_by_source,
            "macro": macro_average(train_by_source, active_sources),
        }

        validation_metrics = evaluate(
            validation_pairs,
            args.split_seed + 1_000_003,
            description=f"Valid {epoch:03d}/{args.epochs}",
        )
        improved = (
            validation_metrics["selection_loss"]
            < best_validation_loss - args.min_delta)
        if improved:
            best_validation_loss = validation_metrics["selection_loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                model_payload(epoch, validation_metrics, "best_validation_loss"),
                best_path)
            status = "best"
        else:
            epochs_without_improvement += 1
            status = "hold"

        torch.save(
            {
                "heads": heads.state_dict(),
                "shared_backbone": (
                    shared_backbone.state_dict()
                    if shared_backbone is not None else None),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "head_updates": dict(total_head_updates),
            },
            checkpoint_path,
        )
        last_epoch = epoch
        epoch_seconds = time.time() - epoch_start
        print_epoch_metrics(
            epoch,
            train_metrics,
            validation_metrics,
            status,
            epochs_without_improvement,
            epoch_seconds,
        )

        epoch_row = {
            "epoch": epoch,
            "improved": int(improved),
            "early_stop_wait": epochs_without_improvement,
            "epoch_seconds": epoch_seconds,
        }
        for split, metrics in (
            ("train", train_metrics), ("val", validation_metrics)
        ):
            for metric in SAMPLED_METRICS:
                epoch_row[f"{split}_macro_{metric}"] = (
                    metrics["macro"][metric])
            for source in active_sources:
                for metric in SAMPLED_METRICS:
                    epoch_row[f"{split}_{source}_{metric}"] = (
                        metrics["by_source"][source][metric])
        with epoch_csv_path.open(
            "a", encoding="utf-8-sig", newline=""
        ) as handle:
            csv.DictWriter(handle, fieldnames=epoch_fields).writerow(epoch_row)
        epoch_history.append({
            "epoch": epoch,
            **{
                f"train_{metric}": train_metrics["macro"][metric]
                for metric in SAMPLED_METRICS
            },
            **{
                f"val_{metric}": validation_metrics["macro"][metric]
                for metric in SAMPLED_METRICS
            },
        })
        save_training_plot(
            epoch_history,
            training_plot_path,
            active_sources,
            negatives_per_anchor,
            evaluation_negatives_per_anchor,
        )

        if args.patience and epochs_without_improvement >= args.patience:
            stopped_early = True
            break

    if last_epoch > start_epoch:
        final_validation = validation_metrics
    else:
        final_validation = evaluate(
            validation_pairs,
            args.split_seed + 1_000_003,
            description="Validation",
        )
    torch.save(
        model_payload(last_epoch, final_validation, "last_training_epoch"),
        last_path)
    if best_epoch == 0:
        best_epoch = last_epoch
        best_validation_loss = final_validation["selection_loss"]
        torch.save(
            model_payload(last_epoch, final_validation, "best_validation_loss"),
            best_path)

    best_checkpoint = torch.load(
        best_path, map_location=device, weights_only=False)
    heads.load_state_dict(best_checkpoint["heads"])
    if shared_backbone is not None and best_checkpoint.get("shared_backbone"):
        shared_backbone.load_state_dict(best_checkpoint["shared_backbone"])
    best_validation = best_checkpoint["config"]["validation_metrics"]
    test_metrics = evaluate(
        test_pairs,
        args.split_seed + 2_000_003,
        description="Final test",
    )
    best_checkpoint["config"]["validation_metrics"] = best_validation
    best_checkpoint["config"]["test_metrics"] = test_metrics
    best_checkpoint["config"]["artifacts"] = {
        "training_log": str(log_path.resolve()),
        "epoch_metrics_csv": str(epoch_csv_path.resolve()),
        "test_metrics_csv": str(test_csv_path.resolve()),
        "training_curves": str(training_plot_path.resolve()),
        "training_curves_pdf": str(training_plot_pdf_path.resolve()),
        "training_curves_svg": str(training_plot_svg_path.resolve()),
        "training_curves_tiff": str(training_plot_tiff_path.resolve()),
        "test_performance_plot": str(test_plot_path.resolve()),
        "test_performance_pdf": str(test_plot_pdf_path.resolve()),
        "test_performance_svg": str(test_plot_svg_path.resolve()),
        "test_performance_tiff": str(test_plot_tiff_path.resolve()),
    }
    torch.save(best_checkpoint, best_path)

    best_updates = best_checkpoint["config"].get("head_updates", {})
    elapsed = time.time() - training_start

    test_fields = [
        "source", "positive_edges", *SAMPLED_METRICS,
        "seed", "split_seed", "best_epoch",
    ]
    with test_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=test_fields)
        writer.writeheader()
        for source in active_sources:
            row = {
                "source": source,
                "positive_edges": len(test_pairs[source]),
                "seed": args.seed,
                "split_seed": args.split_seed,
                "best_epoch": best_epoch,
            }
            row.update({
                metric: test_metrics["by_source"][source][metric]
                for metric in SAMPLED_METRICS
            })
            writer.writerow(row)
        macro_row = {
            "source": "macro",
            "positive_edges": test_count,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "best_epoch": best_epoch,
        }
        macro_row.update({
            metric: test_metrics["macro"][metric]
            for metric in SAMPLED_METRICS
        })
        writer.writerow(macro_row)

    save_training_plot(
        epoch_history,
        training_plot_path,
        active_sources,
        negatives_per_anchor,
        evaluation_negatives_per_anchor,
        export_tiff=True,
    )
    save_test_plot(
        test_metrics,
        test_plot_path,
        active_sources,
        evaluation_negatives_per_anchor,
        export_tiff=True,
    )

    summary_path = output_root / "runs_summary.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    summary_fields = [
        "completed_at", "run_id", "run_directory", "seed", "split_seed",
        "architecture", "sources", "output_dim", "bottleneck", "dropout",
        "batch_size", "negatives_per_anchor",
        "evaluation_negatives_per_anchor", "task_sampling", "learning_rate",
        "parameter_count",
        "epochs_requested", "epochs_completed", "best_epoch", "stopped_early",
        *[f"updates_{source}" for source in AVAILABLE_SOURCES],
        *[f"validation_macro_{metric}" for metric in SAMPLED_METRICS],
        *[f"test_macro_{metric}" for metric in SAMPLED_METRICS],
        "runtime_seconds",
    ]
    for source in AVAILABLE_SOURCES:
        summary_fields.extend([
            f"validation_{source}_{metric}" for metric in SAMPLED_METRICS
        ])
        summary_fields.extend([
            f"test_{source}_{metric}" for metric in SAMPLED_METRICS
        ])
    summary_fields.extend([
        "best_model", "last_model", "training_log", "epoch_metrics_csv",
        "test_metrics_csv", "training_curves", "training_curves_pdf",
        "training_curves_svg", "training_curves_tiff",
        "test_performance_plot", "test_performance_pdf",
        "test_performance_svg", "test_performance_tiff",
    ])
    summary_row = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": output_dir.name,
        "run_directory": str(output_dir.resolve()),
        "seed": args.seed,
        "split_seed": args.split_seed,
        "architecture": args.arch,
        "sources": " ".join(active_sources),
        "output_dim": args.output_dim,
        "bottleneck": args.bottleneck,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "negatives_per_anchor": negatives_per_anchor,
        "evaluation_negatives_per_anchor": evaluation_negatives_per_anchor,
        "task_sampling": args.task_sampling,
        "learning_rate": args.lr,
        "parameter_count": parameter_count,
        "epochs_requested": args.epochs,
        "epochs_completed": last_epoch,
        "best_epoch": best_epoch,
        "stopped_early": int(stopped_early),
        **{
            f"updates_{source}": int(best_updates.get(source, 0))
            for source in AVAILABLE_SOURCES
        },
        "runtime_seconds": elapsed,
        "best_model": str(best_path.resolve()),
        "last_model": str(last_path.resolve()),
        "training_log": str(log_path.resolve()),
        "epoch_metrics_csv": str(epoch_csv_path.resolve()),
        "test_metrics_csv": str(test_csv_path.resolve()),
        "training_curves": str(training_plot_path.resolve()),
        "training_curves_pdf": str(training_plot_pdf_path.resolve()),
        "training_curves_svg": str(training_plot_svg_path.resolve()),
        "training_curves_tiff": str(training_plot_tiff_path.resolve()),
        "test_performance_plot": str(test_plot_path.resolve()),
        "test_performance_pdf": str(test_plot_pdf_path.resolve()),
        "test_performance_svg": str(test_plot_svg_path.resolve()),
        "test_performance_tiff": str(test_plot_tiff_path.resolve()),
    }
    for metric in SAMPLED_METRICS:
        summary_row[f"validation_macro_{metric}"] = (
            best_validation["macro"][metric])
        summary_row[f"test_macro_{metric}"] = test_metrics["macro"][metric]
    for source in AVAILABLE_SOURCES:
        for metric in SAMPLED_METRICS:
            summary_row[f"validation_{source}_{metric}"] = (
                best_validation["by_source"][source][metric]
                if source in active_sources else "")
            summary_row[f"test_{source}_{metric}"] = (
                test_metrics["by_source"][source][metric]
                if source in active_sources else "")
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    summary_lock_path = metadata_dir / ".runs_summary.lock"
    with exclusive_file_lock(summary_lock_path):
        previous_rows = []
        if summary_path.exists() and summary_path.stat().st_size > 0:
            with summary_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                previous_rows = [
                    row for row in csv.DictReader(handle)
                    if row.get("run_directory") != summary_row["run_directory"]
                ]
        summary_temp_path = metadata_dir / (
            f".runs_summary.{os.getpid()}.tmp")
        with summary_temp_path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=summary_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(previous_rows)
            writer.writerow(summary_row)
        os.replace(summary_temp_path, summary_path)

    run_config.update({
        "status": "early_stopped" if stopped_early else "completed",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "epochs_completed": last_epoch,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "validation_macro_metrics": best_validation["macro"],
        "test_macro_metrics": test_metrics["macro"],
        "artifacts": {
            "best_model": str(best_path.resolve()),
            "last_model": str(last_path.resolve()),
            "training_log": str(log_path.resolve()),
            "epoch_metrics_csv": str(epoch_csv_path.resolve()),
            "test_metrics_csv": str(test_csv_path.resolve()),
            "training_curves": str(training_plot_path.resolve()),
            "training_curves_pdf": str(training_plot_pdf_path.resolve()),
            "training_curves_svg": str(training_plot_svg_path.resolve()),
            "training_curves_tiff": str(training_plot_tiff_path.resolve()),
            "test_performance_plot": str(test_plot_path.resolve()),
            "test_performance_pdf": str(test_plot_pdf_path.resolve()),
            "test_performance_svg": str(test_plot_svg_path.resolve()),
            "test_performance_tiff": str(test_plot_tiff_path.resolve()),
        },
    })
    with run_config_path.open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2)

    print_metrics(
        f"Best validation  |  epoch {best_epoch:,}",
        best_validation,
        best_updates,
    )
    print_metrics("Final test  |  evaluated once", test_metrics)
    ending = "early stopped" if stopped_early else "completed"
    emit(
        f"\n{ending} in {format_duration(elapsed)}  |  "
        f"best epoch {best_epoch:,}  |  "
        f"best validation loss {best_validation_loss:.4f}")
    emit(f"best model  {best_path.resolve()}")
    emit(f"last model  {last_path.resolve()}")
    emit(f"test metrics  {test_csv_path.resolve()}")
    emit(f"plots  {training_plot_path.resolve()}  {test_plot_path.resolve()}")
    emit(
        f"publication PDF  {training_plot_pdf_path.resolve()}  "
        f"{test_plot_pdf_path.resolve()}")
    emit(
        f"publication SVG  {training_plot_svg_path.resolve()}  "
        f"{test_plot_svg_path.resolve()}")
    emit(
        f"publication TIFF  {training_plot_tiff_path.resolve()}  "
        f"{test_plot_tiff_path.resolve()}")
    emit(f"run summary  {summary_path.resolve()}")
    log_handle.close()


if __name__ == "__main__":
    main()
