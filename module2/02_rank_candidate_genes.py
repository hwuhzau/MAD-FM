"""Re-rank Module 1 genes with ESM-2 and five relation-specific heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from workflow_utils import command_for, run


def read_ids(path: Path) -> list[str]:
    if path.suffix.lower() in {".csv", ".tsv"}:
        separator = "\t" if path.suffix.lower() == ".tsv" else ","
        values = pd.read_csv(path, sep=separator).iloc[:, 0].dropna().astype(str).tolist()
    else:
        values = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    return list(dict.fromkeys(values))


def prepare_inputs(args: argparse.Namespace) -> tuple[Path, pd.DataFrame]:
    module1 = pd.read_csv(args.module1_results)
    if args.gene_column not in module1.columns:
        raise ValueError(f"Module 1 table lacks {args.gene_column!r}")
    module1[args.gene_column] = module1[args.gene_column].astype(str)
    module1 = module1.drop_duplicates(args.gene_column, keep="first")
    catalog_ids = read_ids(args.embedding_ids)
    matrix = np.load(args.embeddings, mmap_mode="r")
    if matrix.shape != (len(catalog_ids), 480):
        raise ValueError(
            f"Embedding catalog mismatch: {matrix.shape} for {len(catalog_ids)} IDs"
        )
    if len(catalog_ids) != len(set(catalog_ids)):
        raise ValueError("Embedding ID catalog contains duplicates")
    index = {gene: row for row, gene in enumerate(catalog_ids)}
    requested_known = read_ids(args.known_genes)
    known_ids = [gene for gene in requested_known if gene in index]
    known_set = set(known_ids)
    candidate_ids = [
        gene for gene in module1[args.gene_column]
        if gene in index and gene not in known_set
    ]
    missing_candidates = [gene for gene in module1[args.gene_column] if gene not in index]
    missing_known = [gene for gene in requested_known if gene not in index]
    if len(known_ids) < 2:
        raise ValueError("At least two reference genes must have ESM-2 embeddings")
    if not candidate_ids:
        raise ValueError("No Module 1 candidate has a matching ESM-2 embedding")
    prepared = args.output_dir / "prepared_inputs"
    prepared.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": known_ids}).to_parquet(prepared / "known_genes.parquet", index=False)
    pd.DataFrame({"id": candidate_ids}).to_parquet(prepared / "candidate_genes.parquet", index=False)
    np.save(
        prepared / "known_protein_emb_esm2_480.npy",
        np.asarray(matrix[[index[gene] for gene in known_ids]], dtype=np.float32),
    )
    np.save(
        prepared / "candidate_protein_emb_esm2_480.npy",
        np.asarray(matrix[[index[gene] for gene in candidate_ids]], dtype=np.float32),
    )
    (prepared / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "source": "MAD-FM Module 1 to Module 2 interface",
                "verification": {"passed": True, "mode": "module1_bridge"},
                "counts": {
                    "known_retained": len(known_ids),
                    "candidate_retained": len(candidate_ids),
                    "known_missing_embedding": len(missing_known),
                    "candidate_missing_embedding": len(missing_candidates),
                },
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (prepared / "excluded_candidates_without_embedding.txt").write_text(
        "\n".join(missing_candidates) + ("\n" if missing_candidates else ""),
        encoding="utf-8",
    )
    (prepared / "excluded_known_without_embedding.txt").write_text(
        "\n".join(missing_known) + ("\n" if missing_known else ""),
        encoding="utf-8",
    )
    return prepared, module1


def selected_ranking(experiment_dir: Path) -> pd.DataFrame:
    report_path = experiment_dir / "analysis" / "analysis_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    run_id = report["selected_aggregation"]["run_id"]
    path = experiment_dir / "runs" / run_id / "tables" / "final_candidate_ranking.csv"
    return pd.read_csv(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module1-results", type=Path, required=True)
    parser.add_argument("--known-genes", type=Path, required=True)
    parser.add_argument("--embedding-ids", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gene-column", default="gene_id")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-repeats", type=int, default=100)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    prepared, module1 = prepare_inputs(args)
    experiment_name = "module2_candidate_ranking"
    command = command_for(
        "ranking_grid.py",
        "--model", args.model.resolve(),
        "--data-dir", prepared,
        "--output-root", args.output_dir,
        "--experiment-name", experiment_name,
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
    if args.dry_run:
        return
    ranking = selected_ranking(args.output_dir / experiment_name)
    ranking = ranking.rename(
        columns={
            "gene": args.gene_column,
            "rrf_score": "module2_rrf_score",
            "rank_rrf": "final_rank",
            "score_go": "module2_score_go",
            "score_coexp": "module2_score_coexpression",
            "score_hic": "module2_score_hic",
            "score_string": "module2_score_ppi",
            "score_tf": "module2_score_tf",
            "rank_go": "module2_rank_go",
            "rank_coexp": "module2_rank_coexpression",
            "rank_hic": "module2_rank_hic",
            "rank_string": "module2_rank_ppi",
            "rank_tf": "module2_rank_tf",
        }
    )
    merged = ranking.merge(module1, on=args.gene_column, how="inner")
    merged = merged.sort_values("final_rank").reset_index(drop=True)
    output = args.output_dir / "madfm_ranked_candidates.csv"
    merged.to_csv(output, index=False)
    print(f"Integrated MAD-FM ranking: {output.resolve()}")


if __name__ == "__main__":
    main()

