"""Repeated known-gene recovery and final candidate ranking with trained heads.

The evaluation universe for each repeat is ``(candidates U known) - seeds``.
Known seed genes are query evidence and are therefore never allowed to rank as
answers.  Hidden known genes are the relevant items.  Each active checkpoint
head aggregates the highest configurable fraction of seed similarities, and
the active-head ranks are combined with equal-weight reciprocal-rank fusion.

This module is both a command-line program and the shared implementation used
by ``ranking_grid.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "evaluation_data"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results" / "single_runs"
MODEL_ROOT = SCRIPT_DIR.parents[1] / "1_train" / "models" / "protein_multihead" / "all_output"

# User-editable default model selection. This is one run-directory name under
# MODEL_ROOT. Set it to "auto" only when selecting the newest run is intended.
DEFAULT_MODEL_RUN = (
    "seed42_20261005_101049_759729_shared_residual_dim512_bs32_neg49_"
    "lr0.0002_src-go-coexp-hic-string-tf_v3"
)

INPUT_DIM = 480
DEFAULT_KS = (10, 20, 50, 100)
EVALUATION_PROTOCOL_VERSION = 4
HEAD_LABELS = {
    "go": "GO",
    "coexp": "Co-expression",
    "hic": "Hi-C",
    "string": "PPI (STRING)",
    "tf": "TF regulation",
    "rrf": "Equal-weight RRF (fusion)",
}
COLORS = {
    "rrf": "#000000",
    "go": "#0072B2",
    "coexp": "#CC79A7",
    "hic": "#009E73",
    "string": "#D55E00",
    "tf": "#E69F00",
}


@dataclass(frozen=True)
class EvaluationParameters:
    seed_ratio: float = 0.50
    similarity_top_ratio: float = 0.10
    num_repeats: int = 100
    split_seed: int = 42
    rrf_k: float = 60.0
    ks: tuple[int, ...] = DEFAULT_KS


@dataclass
class RankingContext:
    known_ids: list[str]
    candidate_ids: list[str]
    all_ids: list[str]
    active_heads: list[str]
    similarities: dict[str, np.ndarray]
    model_path: str
    model_sha256: str
    model_metadata: dict
    data_dir: str
    data_manifest: dict


class LinearProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(features), p=2, dim=-1)


class ResidualProjection(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, bottleneck: int, dropout: float
    ):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim))
        self.adapter = nn.Sequential(
            nn.Linear(out_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, out_dim),
            nn.Dropout(dropout),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        base = self.base(features)
        return F.normalize(
            base + self.residual_scale * self.adapter(base), p=2, dim=-1
        )


class SharedBackbone(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


class SharedBackboneResidualHead(nn.Module):
    def __init__(
        self,
        shared_backbone: SharedBackbone,
        out_dim: int,
        bottleneck: int,
        dropout: float,
    ):
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        shared = self.shared_backbone(features)
        return F.normalize(
            shared + self.residual_scale * self.adapter(shared), p=2, dim=-1
        )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def format_ratio(value: float) -> str:
    return f"{value:.4g}".replace(".", "p")


def validate_parameters(parameters: EvaluationParameters, known_count: int) -> None:
    if not 0.0 < parameters.seed_ratio < 1.0:
        raise ValueError("seed_ratio must be strictly between 0 and 1")
    if not 0.0 < parameters.similarity_top_ratio <= 1.0:
        raise ValueError("similarity_top_ratio must be in (0, 1]")
    if parameters.num_repeats < 2:
        raise ValueError("num_repeats must be at least 2")
    if parameters.rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    if not parameters.ks or any(k < 1 for k in parameters.ks):
        raise ValueError("all K values must be positive")
    seed_count = int(math.floor(known_count * parameters.seed_ratio + 0.5))
    if not 1 <= seed_count < known_count:
        raise ValueError("seed_ratio leaves an empty seed or hidden-known set")


def find_default_model() -> Path:
    candidates: list[Path] = []
    if MODEL_ROOT.exists():
        for model_path in MODEL_ROOT.glob("*/protein_multihead.pt"):
            if (model_path.parent / "test_metrics.csv").exists():
                candidates.append(model_path)
    if not candidates:
        raise FileNotFoundError(
            "no completed protein_multihead.pt was found; provide --model explicitly"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def resolve_requested_model(
    model_path: Path | None, model_run: str | None
) -> Path | None:
    """Resolve a full checkpoint path or one run folder below MODEL_ROOT."""
    if model_path is not None:
        resolved = model_path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"model checkpoint does not exist: {resolved}")
        return resolved

    selected_run = model_run if model_run is not None else DEFAULT_MODEL_RUN
    if selected_run is None or selected_run.strip().lower() == "auto":
        return None
    selected_run = selected_run.strip()
    if Path(selected_run).name != selected_run:
        raise ValueError(
            "--model-run accepts one run-directory name only; "
            "use --model for a complete checkpoint path"
        )
    resolved = (MODEL_ROOT / selected_run / "protein_multihead.pt").resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"model run was not found: {selected_run}\n"
            f"expected checkpoint: {resolved}"
        )
    return resolved


def infer_sources(config: dict, head_state: dict) -> list[str]:
    configured = config.get("sources")
    if configured:
        return [str(source) for source in configured]
    prefixes = {str(key).split(".", 1)[0] for key in head_state}
    canonical = ["go", "coexp", "hic", "string", "tf"]
    sources = [source for source in canonical if source in prefixes]
    if not sources:
        raise ValueError("could not infer relation heads from checkpoint")
    return sources


def load_projection_heads(
    model_path: Path, device: torch.device, requested_heads: Sequence[str] | None
) -> tuple[nn.ModuleDict, list[str], dict]:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "heads" not in checkpoint:
        raise KeyError("checkpoint has no relation-specific 'heads' state")
    config = dict(checkpoint.get("config", {}))
    head_state = checkpoint["heads"]
    sources = infer_sources(config, head_state)
    if requested_heads:
        unknown = sorted(set(requested_heads) - set(sources))
        if unknown:
            raise ValueError(
                f"requested heads absent from checkpoint: {', '.join(unknown)}"
            )
        active_heads = [head for head in sources if head in set(requested_heads)]
    else:
        active_heads = list(sources)
    if not active_heads:
        raise ValueError("at least one relation head is required")

    architecture = config.get("architecture", config.get("arch", "linear"))
    input_dim = int(config.get("input_dim", INPUT_DIM))
    output_dim = int(config.get("output_dim", 768))
    bottleneck = int(config.get("bottleneck", 256))
    dropout = float(config.get("dropout", 0.1))
    if input_dim != INPUT_DIM:
        raise ValueError(f"checkpoint input dimension is {input_dim}, expected 480")

    shared_backbone: SharedBackbone | None = None
    if architecture == "shared_residual":
        shared_backbone = SharedBackbone(input_dim, output_dim)

    def make_head() -> nn.Module:
        if architecture == "linear":
            return LinearProjection(input_dim, output_dim)
        if architecture == "residual":
            return ResidualProjection(input_dim, output_dim, bottleneck, dropout)
        if architecture == "shared_residual" and shared_backbone is not None:
            return SharedBackboneResidualHead(
                shared_backbone, output_dim, bottleneck, dropout
            )
        raise ValueError(f"unsupported checkpoint architecture: {architecture}")

    all_heads = nn.ModuleDict({source: make_head() for source in sources}).to(device)
    all_heads.load_state_dict(head_state, strict=True)
    shared_state = checkpoint.get("shared_backbone")
    if shared_backbone is not None and shared_state is not None:
        shared_backbone.load_state_dict(shared_state, strict=True)
    all_heads.eval()
    selected = nn.ModuleDict({head: all_heads[head] for head in active_heads})
    selected.eval()
    metadata = {
        "architecture": architecture,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "bottleneck": bottleneck,
        "dropout": dropout,
        "checkpoint_sources": sources,
        "active_heads": active_heads,
        "training_seed": config.get("seed"),
        "epoch_saved": config.get("epoch_saved"),
        "run_id": config.get("run_id"),
    }
    return selected, active_heads, metadata


def l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-9):
        raise ValueError("embedding matrix contains a zero vector")
    return (array / norms).astype(np.float32, copy=False)


@torch.inference_mode()
def project_array(
    head: nn.Module,
    array: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("projection batch size must be positive")
    output: list[np.ndarray] = []
    for start in range(0, len(array), batch_size):
        batch = torch.from_numpy(array[start : start + batch_size]).to(device)
        output.append(head(batch).float().cpu().numpy())
    projected = np.concatenate(output, axis=0).astype(np.float32, copy=False)
    return l2_normalize(projected)


def load_prepared_data(
    data_dir: Path, allow_nonfull_data: bool
) -> tuple[list[str], np.ndarray, list[str], np.ndarray, dict]:
    required = {
        "known_table": data_dir / "known_genes.parquet",
        "known_embeddings": data_dir / "known_protein_emb_esm2_480.npy",
        "candidate_table": data_dir / "candidate_genes.parquet",
        "candidate_embeddings": data_dir / "candidate_protein_emb_esm2_480.npy",
        "manifest": data_dir / "dataset_manifest.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "prepared ranking data are incomplete; first run "
            "prepare_ranking_esm2_embeddings.py. Missing: " + ", ".join(missing)
        )
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    verification = manifest.get("verification", {})
    if not verification.get("passed", False):
        raise RuntimeError("prepared-data reference verification did not pass")
    if verification.get("mode") != "full" and not allow_nonfull_data:
        raise RuntimeError(
            "prepared data were not produced after full reference verification; "
            "rerun preparation in full mode or explicitly use --allow-nonfull-data"
        )

    known_table = pd.read_parquet(required["known_table"])
    candidate_table = pd.read_parquet(required["candidate_table"])
    known_ids = known_table["id"].astype(str).tolist()
    candidate_ids = candidate_table["id"].astype(str).tolist()
    if len(known_ids) != len(set(known_ids)):
        raise ValueError("known evaluation table contains duplicate IDs")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate evaluation table contains duplicate IDs")
    known_embeddings = np.load(required["known_embeddings"]).astype(np.float32)
    candidate_embeddings = np.load(required["candidate_embeddings"]).astype(
        np.float32
    )
    if known_embeddings.shape != (len(known_ids), INPUT_DIM):
        raise ValueError("known ID table and embedding matrix are misaligned")
    if candidate_embeddings.shape != (len(candidate_ids), INPUT_DIM):
        raise ValueError("candidate ID table and embedding matrix are misaligned")
    if not np.isfinite(known_embeddings).all() or not np.isfinite(
        candidate_embeddings
    ).all():
        raise ValueError("prepared embedding matrices contain NaN or infinity")
    return known_ids, known_embeddings, candidate_ids, candidate_embeddings, manifest


def prepare_ranking_context(
    model_path: Path | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    requested_heads: Sequence[str] | None = None,
    device_name: str | None = None,
    projection_batch_size: int = 1024,
    allow_nonfull_data: bool = False,
) -> RankingContext:
    model_path = (model_path or find_default_model()).expanduser().resolve()
    data_dir = data_dir.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    known_ids, known_raw, candidate_ids, candidate_raw, manifest = load_prepared_data(
        data_dir, allow_nonfull_data
    )
    heads, active_heads, model_metadata = load_projection_heads(
        model_path, device, requested_heads
    )

    vector_by_id: dict[str, np.ndarray] = {
        gene_id: vector for gene_id, vector in zip(candidate_ids, candidate_raw)
    }
    for gene_id, vector in zip(known_ids, known_raw):
        if gene_id in vector_by_id:
            cosine = float(
                np.dot(l2_normalize(vector_by_id[gene_id][None])[0],
                       l2_normalize(vector[None])[0])
            )
            if cosine < 0.9999:
                raise ValueError(
                    f"known/candidate overlap vector mismatch for {gene_id}: {cosine}"
                )
        vector_by_id[gene_id] = vector
    all_ids = list(candidate_ids) + [
        gene_id for gene_id in known_ids if gene_id not in set(candidate_ids)
    ]
    all_raw = l2_normalize(
        np.stack([vector_by_id[gene_id] for gene_id in all_ids]).astype(np.float32)
    )
    all_index = {gene_id: index for index, gene_id in enumerate(all_ids)}
    known_positions = np.asarray([all_index[gene_id] for gene_id in known_ids])

    similarities: dict[str, np.ndarray] = {}
    print(
        f"Model: {model_path}\n"
        f"Architecture: {model_metadata['architecture']} | "
        f"heads: {', '.join(active_heads)}\n"
        f"Known: {len(known_ids):,} | candidates: {len(candidate_ids):,} | "
        f"combined universe: {len(all_ids):,}\n"
        f"Projecting on: {device}",
        flush=True,
    )
    for head_name in active_heads:
        projected = project_array(
            heads[head_name], all_raw, device, projection_batch_size
        )
        similarities[head_name] = (
            projected @ projected[known_positions].T
        ).astype(np.float32)
    del heads
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return RankingContext(
        known_ids=known_ids,
        candidate_ids=candidate_ids,
        all_ids=all_ids,
        active_heads=active_heads,
        similarities=similarities,
        model_path=str(model_path),
        model_sha256=sha256_file(model_path),
        model_metadata=model_metadata,
        data_dir=str(data_dir),
        data_manifest=manifest,
    )


def descending_ranks(scores: np.ndarray, gene_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    gene_array = np.asarray(gene_ids, dtype=str)
    order = np.lexsort((gene_array, -scores))
    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.int32)
    return ranks, order


def aggregate_top_fraction(
    similarity: np.ndarray, seed_columns: np.ndarray, ratio: float
) -> tuple[np.ndarray, int]:
    seed_similarity = similarity[:, seed_columns]
    top_count = max(1, int(math.ceil(len(seed_columns) * ratio)))
    if top_count >= len(seed_columns):
        return seed_similarity.mean(axis=1), len(seed_columns)
    partition = np.partition(
        seed_similarity, kth=seed_similarity.shape[1] - top_count, axis=1
    )
    return partition[:, -top_count:].mean(axis=1), top_count


def ranking_metrics(
    order: np.ndarray,
    ranking_ids: Sequence[str],
    hidden_ids: set[str],
    ks: Sequence[int],
) -> tuple[dict[str, float], dict[str, int]]:
    ordered_ids = [ranking_ids[index] for index in order]
    relevant = np.asarray([gene_id in hidden_ids for gene_id in ordered_ids], dtype=bool)
    relevant_count = len(hidden_ids)
    if int(relevant.sum()) != relevant_count:
        raise RuntimeError("not every hidden known gene is present in the ranking")
    relevant_ranks = np.flatnonzero(relevant) + 1
    metrics: dict[str, float] = {
        "mean_rank": float(np.mean(relevant_ranks)),
        "median_rank": float(np.median(relevant_ranks)),
    }
    cumulative = np.cumsum(relevant)
    for requested_k in ks:
        k = min(int(requested_k), len(ranking_ids))
        hits = int(cumulative[k - 1])
        metrics[f"recall_at_{requested_k}"] = hits / relevant_count
        hit_positions = np.flatnonzero(relevant[:k]) + 1
        if len(hit_positions):
            precision_at_hits = cumulative[hit_positions - 1] / hit_positions
            denominator = min(relevant_count, k)
            ap = float(precision_at_hits.sum() / denominator)
        else:
            ap = 0.0
        metrics[f"average_precision_at_{requested_k}"] = ap
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = float((relevant[:k] * discounts).sum())
        ideal_count = min(relevant_count, k)
        idcg = float(discounts[:ideal_count].sum())
        metrics[f"ndcg_at_{requested_k}"] = dcg / idcg if idcg else 0.0
        random_expected_hits = k * relevant_count / len(ranking_ids)
        metrics[f"enrichment_at_{requested_k}"] = (
            hits / random_expected_hits if random_expected_hits else float("nan")
        )
    rank_map = {
        gene_id: int(rank)
        for gene_id, rank in zip(ordered_ids, np.arange(1, len(ordered_ids) + 1))
        if gene_id in hidden_ids
    }
    return metrics, rank_map


def repeat_permutations(
    known_count: int, repeats: int, split_seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(split_seed)
    return [rng.permutation(known_count) for _ in range(repeats)]


def evaluate_repeated_splits(
    context: RankingContext, parameters: EvaluationParameters
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_parameters(parameters, len(context.known_ids))
    known_count = len(context.known_ids)
    seed_count = int(math.floor(known_count * parameters.seed_ratio + 0.5))
    all_index = {gene_id: index for index, gene_id in enumerate(context.all_ids)}
    known_array = np.asarray(context.known_ids, dtype=object)
    all_array = np.asarray(context.all_ids, dtype=object)
    permutations = repeat_permutations(
        known_count, parameters.num_repeats, parameters.split_seed
    )
    repeat_rows: list[dict] = []
    hidden_rank_rows: list[dict] = []
    split_rows: list[dict] = []

    for repeat_index, permutation in enumerate(permutations, start=1):
        seed_columns = np.asarray(permutation[:seed_count], dtype=np.int64)
        hidden_columns = np.asarray(permutation[seed_count:], dtype=np.int64)
        seed_ids = set(known_array[seed_columns].tolist())
        hidden_ids = set(known_array[hidden_columns].tolist())
        ranking_mask = np.asarray(
            [gene_id not in seed_ids for gene_id in context.all_ids], dtype=bool
        )
        ranking_ids = all_array[ranking_mask].astype(str).tolist()
        head_ranks: dict[str, np.ndarray] = {}
        top_count_used = None
        for head_name in context.active_heads:
            all_scores, top_count = aggregate_top_fraction(
                context.similarities[head_name],
                seed_columns,
                parameters.similarity_top_ratio,
            )
            top_count_used = top_count
            scores = all_scores[ranking_mask]
            ranks, order = descending_ranks(scores, ranking_ids)
            head_ranks[head_name] = ranks
            metrics, rank_map = ranking_metrics(
                order, ranking_ids, hidden_ids, parameters.ks
            )
            repeat_rows.append(
                {
                    "repeat": repeat_index,
                    "method": head_name,
                    "seed_count": seed_count,
                    "hidden_count": len(hidden_ids),
                    "ranking_count": len(ranking_ids),
                    "similarity_top_count": top_count,
                    **metrics,
                }
            )
            hidden_rank_rows.extend(
                {
                    "repeat": repeat_index,
                    "method": head_name,
                    "gene": gene_id,
                    "rank": rank,
                }
                for gene_id, rank in rank_map.items()
            )

        rrf_scores = np.zeros(len(ranking_ids), dtype=np.float64)
        for head_name in context.active_heads:
            rrf_scores += 1.0 / (parameters.rrf_k + head_ranks[head_name])
        _, rrf_order = descending_ranks(rrf_scores, ranking_ids)
        metrics, rank_map = ranking_metrics(
            rrf_order, ranking_ids, hidden_ids, parameters.ks
        )
        repeat_rows.append(
            {
                "repeat": repeat_index,
                "method": "rrf",
                "seed_count": seed_count,
                "hidden_count": len(hidden_ids),
                "ranking_count": len(ranking_ids),
                "similarity_top_count": top_count_used,
                **metrics,
            }
        )
        hidden_rank_rows.extend(
            {
                "repeat": repeat_index,
                "method": "rrf",
                "gene": gene_id,
                "rank": rank,
            }
            for gene_id, rank in rank_map.items()
        )
        split_rows.extend(
            {
                "repeat": repeat_index,
                "gene": gene_id,
                "role": "seed" if gene_id in seed_ids else "hidden",
            }
            for gene_id in context.known_ids
        )
    return (
        pd.DataFrame(repeat_rows),
        pd.DataFrame(hidden_rank_rows),
        pd.DataFrame(split_rows),
    )


def summarize_repeat_metrics(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    identity_columns = {
        "repeat",
        "method",
        "seed_count",
        "hidden_count",
        "ranking_count",
        "similarity_top_count",
    }
    metric_columns = [
        column for column in repeat_metrics.columns if column not in identity_columns
    ]
    rows: list[dict] = []
    for method, method_frame in repeat_metrics.groupby("method", sort=False):
        for metric in metric_columns:
            values = method_frame[metric].astype(float).to_numpy()
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            sem = std / math.sqrt(len(values)) if len(values) else float("nan")
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "sem": sem,
                    "ci95_low": mean - 1.96 * sem,
                    "ci95_high": mean + 1.96 * sem,
                    "median": float(np.median(values)),
                    "q025": float(np.quantile(values, 0.025)),
                    "q975": float(np.quantile(values, 0.975)),
                    "repeats": len(values),
                }
            )
    return pd.DataFrame(rows)


def metric_higher_is_better(metric: str) -> bool:
    return metric not in {"mean_rank", "median_rank"}


def random_ranking_baseline(
    metric: str, ranking_count: int, hidden_count: int
) -> float:
    """Return the exact expectation for a uniformly random ranking."""
    if ranking_count <= 0 or hidden_count <= 0:
        return float("nan")
    if metric in {"mean_rank", "median_rank"}:
        return (ranking_count + 1.0) / 2.0

    try:
        requested_k = int(metric.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"unsupported ranking metric: {metric}") from exc
    k = min(requested_k, ranking_count)
    if metric.startswith("recall_at_"):
        return k / ranking_count
    if metric.startswith("enrichment_at_"):
        return 1.0
    if metric.startswith("ndcg_at_"):
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        expected_dcg = (hidden_count / ranking_count) * float(discounts.sum())
        ideal_count = min(hidden_count, k)
        ideal_dcg = float(discounts[:ideal_count].sum())
        return expected_dcg / ideal_dcg if ideal_dcg else 0.0
    if metric.startswith("average_precision_at_"):
        positions = np.arange(1, k + 1, dtype=np.float64)
        if ranking_count == 1:
            expected_precision_contribution = np.ones_like(positions)
        else:
            expected_precision_contribution = (
                hidden_count
                / ranking_count
                * (
                    1.0
                    + (positions - 1.0)
                    * (hidden_count - 1.0)
                    / (ranking_count - 1.0)
                )
            )
        denominator = min(hidden_count, k)
        return float(
            np.sum(expected_precision_contribution / positions) / denominator
        )
    raise ValueError(f"unsupported ranking metric: {metric}")


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return adjusted
    finite_values = values[finite_indices]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    count = len(ranked)
    corrected = ranked * count / np.arange(1, count + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.clip(corrected, 0.0, 1.0)
    restored = np.empty_like(corrected)
    restored[order] = corrected
    adjusted[finite_indices] = restored
    return adjusted


def calculate_random_significance(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare repeat-level performance with the exact random expectation.

    The one-sided Wilcoxon test asks whether a method is consistently better
    than random across the repeated known-gene splits.  These splits overlap,
    so this is a stability test, not an independent external-validation test.
    """
    identity_columns = {
        "repeat",
        "method",
        "seed_count",
        "hidden_count",
        "ranking_count",
        "similarity_top_count",
    }
    metric_columns = [
        column for column in repeat_metrics.columns if column not in identity_columns
    ]
    rows: list[dict] = []
    for method, method_frame in repeat_metrics.groupby("method", sort=False):
        ranking_counts = method_frame["ranking_count"].astype(int).unique()
        hidden_counts = method_frame["hidden_count"].astype(int).unique()
        if len(ranking_counts) != 1 or len(hidden_counts) != 1:
            raise RuntimeError("ranking and hidden counts must be constant within a run")
        ranking_count = int(ranking_counts[0])
        hidden_count = int(hidden_counts[0])
        for metric in metric_columns:
            values = method_frame[metric].astype(float).to_numpy()
            observed = float(np.mean(values))
            baseline = random_ranking_baseline(metric, ranking_count, hidden_count)
            higher = metric_higher_is_better(metric)
            favorable_differences = values - baseline if higher else baseline - values
            if np.allclose(favorable_differences, 0.0):
                p_value = 1.0
            else:
                try:
                    p_value = float(
                        wilcoxon(
                            favorable_differences,
                            alternative="greater",
                            zero_method="wilcox",
                            method="auto",
                        ).pvalue
                    )
                except ValueError:
                    p_value = 1.0
            favorable = observed > baseline if higher else observed < baseline
            if higher:
                fold_over_random = observed / baseline if baseline > 0 else float("nan")
                absolute_improvement = observed - baseline
            else:
                fold_over_random = baseline / observed if observed > 0 else float("nan")
                absolute_improvement = baseline - observed
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "direction": "higher" if higher else "lower",
                    "observed_mean": observed,
                    "random_expectation": baseline,
                    "absolute_improvement": absolute_improvement,
                    "fold_over_random": fold_over_random,
                    "favorable": bool(favorable),
                    "stability_p_value": p_value,
                    "repeats": len(values),
                    "test": "one-sided Wilcoxon signed-rank vs exact random expectation",
                }
            )
    result = pd.DataFrame(rows)
    result["fdr_q_value"] = benjamini_hochberg(result["stability_p_value"])
    result["significant_fdr_0p05"] = (
        result["favorable"] & (result["fdr_q_value"] < 0.05)
    )
    return result


def choose_primary_k(ks: Sequence[int]) -> int:
    """Choose the central reporting depth unless the legacy K=50 is present.

    The percentage-based grid uses three ordered depths (Top 1%, 5%, 10%);
    choosing the middle K makes Top 5% the per-run headline depth.  Existing
    fixed-K runs retain K=50 as their headline whenever it is requested.
    """
    ordered = sorted(set(int(k) for k in ks))
    return 50 if 50 in ordered else ordered[len(ordered) // 2]


def build_primary_overview(
    summary: pd.DataFrame,
    significance: pd.DataFrame,
    active_heads: Sequence[str],
    primary_k: int,
) -> pd.DataFrame:
    metrics = [
        f"recall_at_{primary_k}",
        f"average_precision_at_{primary_k}",
        f"ndcg_at_{primary_k}",
        "mean_rank",
    ]
    rows: list[dict] = []
    for method in ordered_methods(active_heads):
        row: dict[str, object] = {
            "method": method,
            "method_label": HEAD_LABELS.get(method, method),
        }
        method_significance: list[bool] = []
        for metric in metrics:
            summary_row = summary[
                (summary.method == method) & (summary.metric == metric)
            ]
            test_row = significance[
                (significance.method == method) & (significance.metric == metric)
            ]
            if summary_row.empty or test_row.empty:
                continue
            summary_record = summary_row.iloc[0]
            test_record = test_row.iloc[0]
            row[f"{metric}_mean"] = float(summary_record["mean"])
            row[f"{metric}_ci95_low"] = float(summary_record["ci95_low"])
            row[f"{metric}_ci95_high"] = float(summary_record["ci95_high"])
            row[f"{metric}_random"] = float(test_record["random_expectation"])
            row[f"{metric}_fold"] = float(test_record["fold_over_random"])
            row[f"{metric}_fdr_q"] = float(test_record["fdr_q_value"])
            is_significant = bool(test_record["significant_fdr_0p05"])
            row[f"{metric}_significant"] = is_significant
            method_significance.append(is_significant)
        row["all_primary_criteria_significant"] = bool(
            method_significance and all(method_significance)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def format_probability(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def build_result_interpretation(
    overview: pd.DataFrame,
    active_heads: Sequence[str],
    primary_k: int,
    repeats: int,
) -> tuple[str, str]:
    rrf = overview[overview.method == "rrf"]
    if rrf.empty:
        return "无法判定：缺少RRF评测结果。", "unavailable"
    row = rrf.iloc[0]
    metrics = [
        (f"Recall@{primary_k}", f"recall_at_{primary_k}"),
        (f"MAP@{primary_k}", f"average_precision_at_{primary_k}"),
        (f"NDCG@{primary_k}", f"ndcg_at_{primary_k}"),
        ("Mean Rank", "mean_rank"),
    ]
    verdict_ok = bool(row.get("all_primary_criteria_significant", False))
    verdict_code = "significant_better_than_random" if verdict_ok else "not_consistently_significant"
    verdict = (
        "结论：RRF融合排序在主要指标上均显著优于随机排序（FDR q<0.05）。"
        if verdict_ok
        else "结论：RRF融合排序未在全部主要指标上稳定且显著地优于随机排序。"
    )
    lines = [
        verdict,
        f"实际关系头：{len(active_heads)}个（{', '.join(active_heads)}）；RRF是融合方法，不是额外任务头。",
        f"统计口径：{repeats}次重复划分；以下显著性为相对随机排序的单侧Wilcoxon稳定性检验，并进行BH-FDR校正。",
        "",
        f"{'指标':<14}{'模型均值':>12}{'95% CI':>23}{'随机期望':>12}{'提升倍数':>12}{'FDR q':>12}",
        "-" * 85,
    ]
    for label, metric in metrics:
        mean = float(row[f"{metric}_mean"])
        low = float(row[f"{metric}_ci95_low"])
        high = float(row[f"{metric}_ci95_high"])
        baseline = float(row[f"{metric}_random"])
        fold = float(row[f"{metric}_fold"])
        q_value = float(row[f"{metric}_fdr_q"])
        lines.append(
            f"{label:<14}{mean:>12.4f}{f'[{low:.4f}, {high:.4f}]':>23}"
            f"{baseline:>12.4f}{fold:>12.2f}{format_probability(q_value):>12}"
        )
    lines.extend(
        [
            "",
            "解释：Recall/MAP/NDCG越高越好；Mean Rank越低越好；提升倍数均按‘优于随机’方向计算。",
            "限制：重复划分共享同一批已知基因，该检验说明结果稳定优于随机，但不能替代独立外部数据验证。",
        ]
    )
    return "\n".join(lines), verdict_code


def build_terminal_report(
    overview: pd.DataFrame,
    active_heads: Sequence[str],
    primary_k: int,
    verdict_code: str,
) -> str:
    """Build an ASCII-only report that renders correctly in Windows consoles."""
    rrf = overview[overview.method == "rrf"].iloc[0]
    verdict = (
        "SIGNIFICANTLY BETTER THAN RANDOM"
        if verdict_code == "significant_better_than_random"
        else "NOT CONSISTENTLY SIGNIFICANT"
    )
    metrics = [
        (f"Recall@{primary_k}", f"recall_at_{primary_k}"),
        (f"MAP@{primary_k}", f"average_precision_at_{primary_k}"),
        (f"NDCG@{primary_k}", f"ndcg_at_{primary_k}"),
        ("Mean Rank", "mean_rank"),
    ]
    lines = [
        "=" * 86,
        f"RESULT | RRF ranking is {verdict} (BH-FDR q < 0.05)",
        f"Relation heads used ({len(active_heads)}): {', '.join(active_heads)}",
        "RRF is the fusion result; it is not an additional trained head.",
        "-" * 86,
        f"{'Metric':<14}{'Model mean':>12}{'95% CI':>23}{'Random':>12}{'Fold':>10}{'FDR q':>13}",
        "-" * 86,
    ]
    for label, metric in metrics:
        mean = float(rrf[f"{metric}_mean"])
        low = float(rrf[f"{metric}_ci95_low"])
        high = float(rrf[f"{metric}_ci95_high"])
        baseline = float(rrf[f"{metric}_random"])
        fold = float(rrf[f"{metric}_fold"])
        q_value = float(rrf[f"{metric}_fdr_q"])
        lines.append(
            f"{label:<14}{mean:>12.4f}{f'[{low:.4f}, {high:.4f}]':>23}"
            f"{baseline:>12.4f}{fold:>10.2f}{format_probability(q_value):>13}"
        )
    lines.extend(
        [
            "-" * 86,
            "Higher is better except Mean Rank; fold is always expressed in the favorable direction.",
            "Inference scope: repeated-split stability vs random, not independent external validation.",
            "=" * 86,
        ]
    )
    return "\n".join(lines)


def final_candidate_ranking(
    context: RankingContext, parameters: EvaluationParameters
) -> pd.DataFrame:
    known_set = set(context.known_ids)
    final_ids = [gene_id for gene_id in context.candidate_ids if gene_id not in known_set]
    all_index = {gene_id: index for index, gene_id in enumerate(context.all_ids)}
    rows = np.asarray([all_index[gene_id] for gene_id in final_ids], dtype=np.int64)
    all_seed_columns = np.arange(len(context.known_ids), dtype=np.int64)
    result = pd.DataFrame({"gene": final_ids})
    rrf_score = np.zeros(len(final_ids), dtype=np.float64)
    top_count_used = None
    for head_name in context.active_heads:
        all_scores, top_count = aggregate_top_fraction(
            context.similarities[head_name],
            all_seed_columns,
            parameters.similarity_top_ratio,
        )
        top_count_used = top_count
        scores = all_scores[rows]
        ranks, _ = descending_ranks(scores, final_ids)
        result[f"score_{head_name}"] = scores
        result[f"rank_{head_name}"] = ranks
        rrf_score += 1.0 / (parameters.rrf_k + ranks)
    rrf_rank, rrf_order = descending_ranks(rrf_score, final_ids)
    result["rrf_score"] = rrf_score
    result["rank_rrf"] = rrf_rank
    result["known_seed_count"] = len(context.known_ids)
    result["similarity_top_count"] = top_count_used
    result["active_head_count"] = len(context.active_heads)
    result = result.iloc[rrf_order].reset_index(drop=True)
    return result


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13.0,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 9.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def ordered_methods(active_heads: Sequence[str]) -> list[str]:
    return ["rrf", *active_heads]


def plot_metric_family(
    summary: pd.DataFrame,
    significance: pd.DataFrame,
    active_heads: Sequence[str],
    ks: Sequence[int],
    metric_prefix: str,
    ylabel: str,
    title: str,
    output_base: Path,
) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    for method in ordered_methods(active_heads):
        means, low, high = [], [], []
        for k in ks:
            metric = f"{metric_prefix}_{k}"
            row = summary[(summary.method == method) & (summary.metric == metric)]
            if row.empty:
                means.append(np.nan); low.append(np.nan); high.append(np.nan)
            else:
                record = row.iloc[0]
                means.append(float(record["mean"]))
                low.append(float(record["ci95_low"]))
                high.append(float(record["ci95_high"]))
        means_array = np.asarray(means)
        errors = np.vstack(
            [means_array - np.asarray(low), np.asarray(high) - means_array]
        )
        ax.errorbar(
            ks,
            means_array,
            yerr=errors,
            marker="o",
            capsize=3,
            color=COLORS.get(method, "#666666"),
            label=HEAD_LABELS.get(method, method),
            zorder=3 if method == "rrf" else 2,
        )
        if method == "rrf":
            for k, mean in zip(ks, means_array):
                metric = f"{metric_prefix}_{k}"
                test_row = significance[
                    (significance.method == method)
                    & (significance.metric == metric)
                ]
                if not test_row.empty and bool(
                    test_row.iloc[0]["significant_fdr_0p05"]
                ):
                    q_value = float(test_row.iloc[0]["fdr_q_value"])
                    stars = "***" if q_value < 0.001 else "**" if q_value < 0.01 else "*"
                    ax.annotate(
                        stars,
                        (k, mean),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color=COLORS["rrf"],
                    )
    baseline_values: list[float] = []
    for k in ks:
        metric = f"{metric_prefix}_{k}"
        row = significance[significance.metric == metric]
        baseline_values.append(
            float(row.iloc[0]["random_expectation"]) if not row.empty else np.nan
        )
    ax.plot(
        ks,
        baseline_values,
        color="#7F7F7F",
        linestyle="--",
        linewidth=1.3,
        marker="s",
        markersize=3.5,
        label="Random ranking expectation",
        zorder=1,
    )
    ax.set_title(title, pad=10, weight="semibold")
    ax.set_xlabel("Ranking cutoff K")
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(ks))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    ax.text(
        0.0,
        -0.20,
        "Mean and 95% CI across repeated splits; RRF: * FDR q<0.05, ** q<0.01, *** q<0.001.",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#555555",
    )
    save_figure(fig, output_base)


def plot_hidden_rank_distribution(
    hidden_ranks: pd.DataFrame,
    active_heads: Sequence[str],
    random_expected_rank: float,
    output_base: Path,
) -> None:
    configure_plot_style()
    methods = ordered_methods(active_heads)
    values = [
        hidden_ranks.loc[hidden_ranks.method == method, "rank"].to_numpy()
        for method in methods
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    box = ax.boxplot(
        values,
        tick_labels=[HEAD_LABELS.get(method, method) for method in methods],
        showfliers=False,
        patch_artist=True,
        orientation="horizontal",
        medianprops={"color": "#222222", "linewidth": 1.3},
    )
    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(COLORS.get(method, "#999999"))
        patch.set_alpha(0.72)
    ax.set_title("Hidden known-gene rank distribution", pad=10, weight="semibold")
    ax.set_xlabel("Rank (lower is better)")
    ax.axvline(
        random_expected_rank,
        color="#7F7F7F",
        linestyle="--",
        linewidth=1.3,
        label=f"Random expectation ({random_expected_rank:,.1f})",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, output_base)


def plot_rrf_rank_ecdf(
    hidden_ranks: pd.DataFrame, ranking_count: int, output_base: Path
) -> None:
    configure_plot_style()
    ranks = np.sort(
        hidden_ranks.loc[hidden_ranks.method == "rrf", "rank"].to_numpy()
    )
    y = np.arange(1, len(ranks) + 1) / len(ranks)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.step(
        ranks,
        y,
        where="post",
        color="#000000",
        linewidth=2.0,
        label="Observed RRF",
    )
    random_x = np.asarray([1.0, float(ranking_count)])
    ax.plot(
        random_x,
        random_x / ranking_count,
        color="#7F7F7F",
        linestyle="--",
        linewidth=1.3,
        label="Random ranking expectation",
    )
    ax.set_title("RRF hidden-gene cumulative rank distribution", pad=10, weight="semibold")
    ax.set_xlabel("Rank threshold")
    ax.set_ylabel("Fraction of hidden known genes recovered")
    ax.set_ylim(0, 1.01)
    ax.grid(color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, output_base)


def plot_final_top_candidates(final_ranking: pd.DataFrame, output_base: Path) -> None:
    configure_plot_style()
    frame = final_ranking.head(min(20, len(final_ranking))).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    ax.barh(frame["gene"], frame["rrf_score"], color="#0072B2", alpha=0.88)
    ax.set_title(
        "Final candidate prioritization (inference output)",
        pad=10,
        weight="semibold",
    )
    ax.set_xlabel("RRF score")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.0,
        -0.12,
        "This figure reports candidate priority; it is not evidence of validation significance.",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#555555",
    )
    save_figure(fig, output_base)


def significance_stars(q_value: float) -> str:
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return "ns"


def plot_rrf_hidden_genes_recovered(
    summary: pd.DataFrame,
    significance: pd.DataFrame,
    ks: Sequence[int],
    hidden_count: int,
    output_base: Path,
) -> None:
    """Show an intuitive count of hidden known genes recovered by RRF."""
    configure_plot_style()
    observed, low, high, random_values, q_values = [], [], [], [], []
    for k in ks:
        metric = f"recall_at_{k}"
        summary_row = summary[
            (summary.method == "rrf") & (summary.metric == metric)
        ].iloc[0]
        test_row = significance[
            (significance.method == "rrf") & (significance.metric == metric)
        ].iloc[0]
        observed.append(float(summary_row["mean"]) * hidden_count)
        low.append(float(summary_row["ci95_low"]) * hidden_count)
        high.append(float(summary_row["ci95_high"]) * hidden_count)
        random_values.append(float(test_row["random_expectation"]) * hidden_count)
        q_values.append(float(test_row["fdr_q_value"]))

    observed_array = np.asarray(observed)
    low_array = np.asarray(low)
    high_array = np.asarray(high)
    fig, ax = plt.subplots(figsize=(7.2, 4.9), constrained_layout=True)
    ax.plot(
        ks,
        observed_array,
        color="#000000",
        marker="o",
        linewidth=2.2,
        label="Observed RRF",
        zorder=3,
    )
    ax.fill_between(
        ks,
        low_array,
        high_array,
        color="#7F7F7F",
        alpha=0.20,
        linewidth=0,
        label="95% CI across splits",
    )
    ax.plot(
        ks,
        random_values,
        color="#A0A0A0",
        linestyle="--",
        marker="s",
        linewidth=1.4,
        label="Random ranking expectation",
        zorder=2,
    )
    for k, value, random_value, q_value in zip(
        ks, observed, random_values, q_values
    ):
        ax.annotate(
            f"{value:.1f} vs {random_value:.1f}  {significance_stars(q_value)}",
            (k, value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
        )
    ax.set_title("Hidden known genes recovered by RRF", pad=10, weight="semibold")
    ax.set_xlabel("Top-K candidates inspected")
    ax.set_ylabel(f"Hidden known genes recovered (out of {hidden_count})")
    ax.set_xticks(list(ks))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.0,
        -0.18,
        "Labels show observed vs random counts; * FDR q<0.05, ** q<0.01, *** q<0.001.",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#555555",
    )
    save_figure(fig, output_base)


def plot_method_comparison_at_k(
    summary: pd.DataFrame,
    significance: pd.DataFrame,
    active_heads: Sequence[str],
    metric: str,
    metric_label: str,
    title: str,
    output_base: Path,
) -> None:
    """Compare only active heads and RRF at one interpretable cutoff."""
    configure_plot_style()
    methods = ordered_methods(active_heads)
    labels, means, lows, highs, q_values = [], [], [], [], []
    for method in methods:
        summary_row = summary[
            (summary.method == method) & (summary.metric == metric)
        ].iloc[0]
        test_row = significance[
            (significance.method == method) & (significance.metric == metric)
        ].iloc[0]
        labels.append(HEAD_LABELS.get(method, method))
        means.append(float(summary_row["mean"]))
        lows.append(float(summary_row["ci95_low"]))
        highs.append(float(summary_row["ci95_high"]))
        q_values.append(float(test_row["fdr_q_value"]))
    random_value = float(
        significance[significance.metric == metric].iloc[0]["random_expectation"]
    )
    y = np.arange(len(methods))
    means_array = np.asarray(means)
    errors = np.vstack(
        [means_array - np.asarray(lows), np.asarray(highs) - means_array]
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.9), constrained_layout=True)
    bars = ax.barh(
        y,
        means_array,
        color=[COLORS.get(method, "#777777") for method in methods],
        alpha=0.84,
        height=0.62,
        xerr=errors,
        error_kw={"ecolor": "#222222", "capsize": 3, "elinewidth": 1.0},
    )
    ax.axvline(
        random_value,
        color="#7F7F7F",
        linestyle="--",
        linewidth=1.4,
        label=f"Random expectation = {random_value:.4f}",
    )
    maximum = max(highs)
    for bar, mean, q_value in zip(bars, means, q_values):
        ax.text(
            min(mean + maximum * 0.025, maximum * 1.17),
            bar.get_y() + bar.get_height() / 2,
            f"{mean:.3f} {significance_stars(q_value)}",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, maximum * 1.25)
    ax.set_xlabel(metric_label)
    ax.set_title(title, pad=10, weight="semibold")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.0,
        -0.15,
        "Bars are means with 95% CI; significance is versus random after BH-FDR correction.",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#555555",
    )
    save_figure(fig, output_base)


def plot_rrf_repeat_stability(
    repeat_metrics: pd.DataFrame,
    significance: pd.DataFrame,
    primary_k: int,
    output_base: Path,
) -> None:
    """Expose the full repeated-split distribution instead of only its mean."""
    configure_plot_style()
    metric = f"recall_at_{primary_k}"
    values = repeat_metrics.loc[
        repeat_metrics.method == "rrf", metric
    ].astype(float).to_numpy()
    test_row = significance[
        (significance.method == "rrf") & (significance.metric == metric)
    ].iloc[0]
    random_value = float(test_row["random_expectation"])
    q_value = float(test_row["fdr_q_value"])
    rng = np.random.default_rng(20260901)
    x = 1.0 + rng.uniform(-0.075, 0.075, size=len(values))
    fig, ax = plt.subplots(figsize=(5.8, 4.9), constrained_layout=True)
    violin = ax.violinplot(
        [values], positions=[1], widths=0.55, showmeans=False, showmedians=False
    )
    for body in violin["bodies"]:
        body.set_facecolor("#0072B2")
        body.set_edgecolor("#00507A")
        body.set_alpha(0.25)
    ax.scatter(x, values, s=14, color="#0072B2", alpha=0.45, linewidths=0)
    box = ax.boxplot(
        [values],
        positions=[1],
        widths=0.20,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "white", "edgecolor": "#222222"},
        medianprops={"color": "#D55E00", "linewidth": 1.8},
    )
    _ = box
    ax.axhline(
        random_value,
        color="#7F7F7F",
        linestyle="--",
        linewidth=1.4,
        label=f"Random expectation = {random_value:.4f}",
    )
    ax.set_xticks([1], ["Equal-weight RRF (fusion)"])
    ax.set_xlim(0.55, 1.45)
    ax.set_ylabel(f"Recall@{primary_k}")
    ax.set_title(
        f"RRF Recall@{primary_k} across {len(values)} repeated splits",
        pad=10,
        weight="semibold",
    )
    ax.text(
        0.98,
        0.97,
        f"mean={np.mean(values):.3f}; FDR q={format_probability(q_value)} "
        f"({significance_stars(q_value)})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.5},
    )
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    save_figure(fig, output_base)


def plot_rrf_improvement_forest(
    summary: pd.DataFrame,
    significance: pd.DataFrame,
    primary_k: int,
    output_base: Path,
) -> None:
    """Summarize magnitude and significance in one direct-effect figure."""
    configure_plot_style()
    metric_specs = [
        (f"recall_at_{primary_k}", f"Recall@{primary_k}"),
        (f"average_precision_at_{primary_k}", f"MAP@{primary_k}"),
        (f"ndcg_at_{primary_k}", f"NDCG@{primary_k}"),
        ("mean_rank", "Mean Rank"),
    ]
    folds, low_folds, high_folds, q_values, labels = [], [], [], [], []
    for metric, label in metric_specs:
        summary_row = summary[
            (summary.method == "rrf") & (summary.metric == metric)
        ].iloc[0]
        test_row = significance[
            (significance.method == "rrf") & (significance.metric == metric)
        ].iloc[0]
        baseline = float(test_row["random_expectation"])
        mean = float(summary_row["mean"])
        low = max(float(summary_row["ci95_low"]), np.finfo(float).eps)
        high = max(float(summary_row["ci95_high"]), np.finfo(float).eps)
        if metric_higher_is_better(metric):
            fold, low_fold, high_fold = mean / baseline, low / baseline, high / baseline
        else:
            fold, low_fold, high_fold = baseline / mean, baseline / high, baseline / low
        folds.append(fold)
        low_folds.append(low_fold)
        high_folds.append(high_fold)
        q_values.append(float(test_row["fdr_q_value"]))
        labels.append(label)
    y = np.arange(len(labels))
    folds_array = np.asarray(folds)
    errors = np.vstack(
        [folds_array - np.asarray(low_folds), np.asarray(high_folds) - folds_array]
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    ax.errorbar(
        folds_array,
        y,
        xerr=errors,
        fmt="o",
        color="#000000",
        markerfacecolor="#0072B2",
        markeredgecolor="#000000",
        markersize=7,
        capsize=4,
        linewidth=1.2,
    )
    ax.axvline(1.0, color="#D55E00", linestyle="--", linewidth=1.4)
    for index, (fold, q_value) in enumerate(zip(folds, q_values)):
        ax.text(
            fold * 1.08,
            index,
            f"{fold:.2f}x; q={format_probability(q_value)}",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.8, max(high_folds) * 2.0)
    ax.set_xlabel("Improvement factor over random (log scale; >1 is better)")
    ax.set_title("RRF effect size and statistical significance", pad=10, weight="semibold")
    ax.grid(axis="x", which="both", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.0,
        -0.16,
        "Points are favorable-direction fold changes; bars are transformed 95% CIs.",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#555555",
    )
    save_figure(fig, output_base)


def create_plots(
    repeat_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    significance: pd.DataFrame,
    hidden_ranks: pd.DataFrame,
    final_ranking: pd.DataFrame,
    context: RankingContext,
    parameters: EvaluationParameters,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    primary_k = choose_primary_k(parameters.ks)
    ranking_count = len(context.all_ids) - int(
        math.floor(len(context.known_ids) * parameters.seed_ratio + 0.5)
    )
    hidden_count = len(context.known_ids) - int(
        math.floor(len(context.known_ids) * parameters.seed_ratio + 0.5)
    )
    plot_rrf_hidden_genes_recovered(
        summary,
        significance,
        parameters.ks,
        hidden_count,
        figures_dir / "01_rrf_hidden_genes_recovered",
    )
    plot_rrf_repeat_stability(
        repeat_metrics,
        significance,
        primary_k,
        figures_dir / "02_rrf_repeat_stability",
    )
    plot_rrf_improvement_forest(
        summary,
        significance,
        primary_k,
        figures_dir / "03_rrf_effect_size_significance",
    )
    plot_method_comparison_at_k(
        summary,
        significance,
        context.active_heads,
        f"recall_at_{primary_k}",
        f"Recall@{primary_k}",
        f"Known-gene recovery by active head and fusion at K={primary_k}",
        figures_dir / "04_method_comparison_recall",
    )
    plot_method_comparison_at_k(
        summary,
        significance,
        context.active_heads,
        f"ndcg_at_{primary_k}",
        f"NDCG@{primary_k}",
        f"Ranking quality by active head and fusion at K={primary_k}",
        figures_dir / "05_method_comparison_ndcg",
    )
    plot_rrf_rank_ecdf(
        hidden_ranks,
        ranking_count,
        figures_dir / "06_rrf_rank_ecdf_vs_random",
    )
    if not final_ranking.empty:
        plot_final_top_candidates(
            final_ranking, figures_dir / "07_final_candidate_top20_inference_only"
        )


def build_figure_guide(primary_k: int, active_heads: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "figure": "01_rrf_hidden_genes_recovered",
                "type": "validation",
                "question": "RRF在不同Top-K中实际找回多少隐藏已知基因？",
                "how_to_read": "黑线越高越好；灰色虚线是随机排序；标签直接给出模型与随机找回数量；星号为FDR显著性。",
            },
            {
                "figure": "02_rrf_repeat_stability",
                "type": "validation",
                "question": f"RRF的Recall@{primary_k}在重复划分中是否稳定？",
                "how_to_read": "每个点代表一次划分；小提琴和箱线表示整体分布；若绝大多数点高于随机虚线，说明结果不是由少数划分驱动。",
            },
            {
                "figure": "03_rrf_effect_size_significance",
                "type": "validation",
                "question": "RRF相对随机排序改善了多少，并且是否显著？",
                "how_to_read": "橙色虚线1表示无改善；点越靠右改善越大；横线为95%区间；每行同时标注提升倍数和FDR q值。",
            },
            {
                "figure": "04_method_comparison_recall",
                "type": "diagnostic",
                "question": f"在Recall@{primary_k}上，RRF和各实际关系头谁更有效？",
                "how_to_read": "只包含实际启用的关系头与RRF；柱越长越好；灰色虚线为随机期望；误差线为95%区间。",
            },
            {
                "figure": "05_method_comparison_ndcg",
                "type": "diagnostic",
                "question": f"在NDCG@{primary_k}上，哪个头能把隐藏基因放得更靠前？",
                "how_to_read": "柱越长表示正确基因越集中在榜单前部；灰色虚线为随机期望；星号表示相对随机显著。",
            },
            {
                "figure": "06_rrf_rank_ecdf_vs_random",
                "type": "validation",
                "question": "全部隐藏基因的整体排名分布是否优于随机？",
                "how_to_read": "黑色RRF曲线在灰色随机线左上方越远越好，表示在较小Rank阈值内已覆盖更多隐藏基因。",
            },
            {
                "figure": "07_final_candidate_top20_inference_only",
                "type": "inference",
                "question": "使用全部已知基因后，最终优先候选基因是谁？",
                "how_to_read": "柱越长表示RRF优先级越高；该图仅用于候选输出，不用于证明模型有效或显著。",
            },
        ]
    ).assign(active_heads=", ".join(active_heads))


def run_identifier(parameters: EvaluationParameters) -> str:
    return (
        f"seedratio-{format_ratio(parameters.seed_ratio)}__"
        f"topratio-{format_ratio(parameters.similarity_top_ratio)}__"
        f"repeats-{parameters.num_repeats}__splitseed-{parameters.split_seed}"
    )


def execute_evaluation(
    context: RankingContext,
    parameters: EvaluationParameters,
    output_dir: Path,
    overwrite: bool = False,
    generate_plots: bool = True,
) -> dict:
    started = time.time()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; choose another directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if overwrite and figures_dir.exists():
        for generated_file in figures_dir.iterdir():
            if generated_file.is_file():
                generated_file.unlink()

    repeat_metrics, hidden_ranks, splits = evaluate_repeated_splits(
        context, parameters
    )
    summary = summarize_repeat_metrics(repeat_metrics)
    significance = calculate_random_significance(repeat_metrics)
    primary_k = choose_primary_k(parameters.ks)
    overview = build_primary_overview(
        summary, significance, context.active_heads, primary_k
    )
    figure_guide = build_figure_guide(primary_k, context.active_heads)
    interpretation_text, verdict_code = build_result_interpretation(
        overview, context.active_heads, primary_k, parameters.num_repeats
    )
    terminal_report = build_terminal_report(
        overview, context.active_heads, primary_k, verdict_code
    )
    final_ranking = final_candidate_ranking(context, parameters)
    atomic_write_csv(tables_dir / "metrics_by_repeat.csv", repeat_metrics)
    atomic_write_csv(tables_dir / "metrics_summary_long.csv", summary)
    atomic_write_csv(tables_dir / "random_baseline_significance.csv", significance)
    atomic_write_csv(tables_dir / "primary_results_overview.csv", overview)
    atomic_write_csv(tables_dir / "figure_guide.csv", figure_guide)
    atomic_write_csv(tables_dir / "hidden_gene_ranks.csv", hidden_ranks)
    atomic_write_csv(tables_dir / "split_assignments.csv", splits)
    atomic_write_csv(tables_dir / "final_candidate_ranking.csv", final_ranking)
    atomic_write_csv(
        tables_dir / "final_top100_candidates.csv", final_ranking.head(100)
    )
    (output_dir / "result_interpretation.txt").write_text(
        interpretation_text + "\n", encoding="utf-8"
    )
    if generate_plots:
        create_plots(
            repeat_metrics,
            summary,
            significance,
            hidden_ranks,
            final_ranking,
            context,
            parameters,
            figures_dir,
        )
        guide_lines = ["排序评测图片阅读指南", "=" * 72, ""]
        for record in figure_guide.to_dict(orient="records"):
            guide_lines.extend(
                [
                    f"{record['figure']}  [{record['type']}]",
                    f"作用：{record['question']}",
                    f"读法：{record['how_to_read']}",
                    "",
                ]
            )
        (figures_dir / "README_如何阅读图片.txt").write_text(
            "\n".join(guide_lines), encoding="utf-8"
        )

    seed_count = int(
        math.floor(len(context.known_ids) * parameters.seed_ratio + 0.5)
    )
    hidden_count = len(context.known_ids) - seed_count
    combined_count = len(context.all_ids)
    ranking_count = combined_count - seed_count
    top_count_evaluation = max(
        1, int(math.ceil(seed_count * parameters.similarity_top_ratio))
    )
    top_count_final = max(
        1, int(math.ceil(len(context.known_ids) * parameters.similarity_top_ratio))
    )
    rrf_summary = summary[summary.method == "rrf"]
    rrf_metrics = {
        row.metric: {
            "mean": float(row["mean"]),
            "std": float(row["std"]),
            "ci95_low": float(row["ci95_low"]),
            "ci95_high": float(row["ci95_high"]),
        }
        for _, row in rrf_summary.iterrows()
    }
    run_summary = {
        "status": "complete",
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "completed_at_local": datetime.now().isoformat(timespec="seconds"),
        "runtime_seconds": round(time.time() - started, 3),
        "run_id": run_identifier(parameters),
        "parameters": asdict(parameters),
        "model": {
            "path": context.model_path,
            "sha256": context.model_sha256,
            **context.model_metadata,
        },
        "data": {
            "directory": context.data_dir,
            "known_count": len(context.known_ids),
            "candidate_count": len(context.candidate_ids),
            "known_candidate_overlap": len(
                set(context.known_ids) & set(context.candidate_ids)
            ),
            "combined_count": combined_count,
            "seed_count_per_repeat": seed_count,
            "hidden_count_per_repeat": hidden_count,
            "ranking_count_per_repeat": ranking_count,
            "final_unknown_candidate_count": len(final_ranking),
        },
        "aggregation": {
            "evaluation_top_seed_count": top_count_evaluation,
            "final_top_seed_count": top_count_final,
            "head_weights": "equal",
            "active_heads": context.active_heads,
            "rrf_formula": "sum_h 1 / (rrf_k + rank_h)",
            "seeds_removed_before_ranking": True,
        },
        "rrf_metrics": rrf_metrics,
        "statistical_assessment": {
            "primary_k": primary_k,
            "verdict": verdict_code,
            "random_baseline": "exact expectation under a uniform random ranking",
            "test": "one-sided Wilcoxon signed-rank across repeated splits",
            "multiple_testing": "Benjamini-Hochberg FDR across all method-metric tests",
            "significance_threshold": 0.05,
            "independence_warning": (
                "Repeated splits reuse the same known-gene set; significance is a "
                "stability comparison with random ranking, not independent external validation."
            ),
            "rrf_primary_results": json.loads(
                overview[overview.method == "rrf"].iloc[0].to_json()
            ),
        },
        "artifacts": {
            "tables": str(tables_dir),
            "figures": str(figures_dir),
            "final_ranking": str(tables_dir / "final_candidate_ranking.csv"),
            "significance": str(tables_dir / "random_baseline_significance.csv"),
            "primary_overview": str(tables_dir / "primary_results_overview.csv"),
            "figure_guide": str(tables_dir / "figure_guide.csv"),
            "interpretation": str(output_dir / "result_interpretation.txt"),
        },
        "terminal_report": terminal_report,
    }
    atomic_write_json(output_dir / "run_summary.json", run_summary)
    atomic_write_json(
        output_dir / "run_config.json",
        {
            "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
            "parameters": asdict(parameters),
            "model_sha256": context.model_sha256,
            "model_path": context.model_path,
            "active_heads": context.active_heads,
            "data_dir": context.data_dir,
        },
    )
    return run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeated known-gene recovery and final RRF candidate ranking"
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model",
        type=Path,
        help="complete path to protein_multihead.pt; overrides DEFAULT_MODEL_RUN",
    )
    model_group.add_argument(
        "--model-run",
        help=(
            "run-directory name under models/protein_multihead/all_output; "
            f"source default: {DEFAULT_MODEL_RUN!r}; use 'auto' for newest"
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--heads",
        nargs="+",
        help="optional subset; default uses every head stored in the checkpoint",
    )
    parser.add_argument("--seed-ratio", type=float, default=0.50)
    parser.add_argument("--similarity-top-ratio", type=float, default=0.10)
    parser.add_argument("--num-repeats", type=int, default=100)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--projection-batch-size", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--allow-nonfull-data",
        action="store_true",
        help="allow prepared data whose ESM2 reference check used sample mode",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_model = resolve_requested_model(args.model, args.model_run)
    parameters = EvaluationParameters(
        seed_ratio=args.seed_ratio,
        similarity_top_ratio=args.similarity_top_ratio,
        num_repeats=args.num_repeats,
        split_seed=args.split_seed,
        rrf_k=args.rrf_k,
        ks=tuple(sorted(set(args.ks))),
    )
    context = prepare_ranking_context(
        model_path=selected_model,
        data_dir=args.data_dir,
        requested_heads=args.heads,
        device_name=args.device,
        projection_batch_size=args.projection_batch_size,
        allow_nonfull_data=args.allow_nonfull_data,
    )
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = DEFAULT_RESULTS_DIR / f"{timestamp}__{run_identifier(parameters)}"
    print(
        f"Evaluation parameters: seed_ratio={parameters.seed_ratio:g}, "
        f"similarity_top_ratio={parameters.similarity_top_ratio:g}, "
        f"repeats={parameters.num_repeats}, rrf_k={parameters.rrf_k:g}",
        flush=True,
    )
    summary = execute_evaluation(
        context,
        parameters,
        output_dir,
        overwrite=False,
        generate_plots=not args.no_plots,
    )
    print(
        "\n" + summary["terminal_report"] + "\n\n"
        f"Completed in {summary['runtime_seconds']:.1f}s | "
        f"final candidates: {summary['data']['final_unknown_candidate_count']:,}\n"
        f"Output: {Path(summary['artifacts']['tables']).parent}",
        flush=True,
    )


if __name__ == "__main__":
    main()
