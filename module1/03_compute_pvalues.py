"""Compute uncorrected gene-level p-values from observed and permuted results."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp


def parse_subnet(value: str) -> list[float]:
    return eval(value, {"__builtins__": {}}, {"nan": np.nan})


def replace_nan(values: list[float]) -> list[int]:
    return [-1 if pd.isna(value) else int(value) for value in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--permuted", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/module1/uncorrected_pvalues.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observed = pd.read_csv(args.observed)
    permuted = pd.read_csv(args.permuted)
    observed["subnet"] = observed["subnet"].apply(parse_subnet).apply(replace_nan)
    permuted["subnet"] = permuted["subnet"].apply(parse_subnet).apply(replace_nan)
    results = []
    for gene_id in observed["gene_id"].unique():
        positive = observed[observed.gene_id == gene_id].sort_values("r2")
        if len(positive) < 6:
            raise ValueError(f"Gene {gene_id} has fewer than six observed folds")
        trimmed_positive = positive["r2"].to_numpy()[4:-2]
        positive_mean = float(np.mean(trimmed_positive))
        positive_best = float(trimmed_positive[-1])
        gene_null = permuted[permuted.gene_id == gene_id]
        null_means = []
        for permutation_id in gene_null["shuffle_id"].unique():
            values = (
                gene_null[gene_null.shuffle_id == permutation_id]
                .sort_values("r2")["r2"].to_numpy()
            )
            if len(values) < 6:
                raise ValueError(
                    f"Gene {gene_id}, permutation {permutation_id} has fewer than six folds"
                )
            null_means.append(float(np.mean(values[4:-2])))
        _, p_value = ttest_1samp(null_means, popmean=positive_mean, alternative="less")
        subnet_rows = positive.iloc[4:-2]
        subnets = np.asarray(subnet_rows["subnet"].tolist())
        voted_subnet = [
            int(np.sum(column) >= len(column) - np.sum(column))
            for column in subnets.T
        ]
        if positive_best < 0.01:
            p_value = 0.999
        results.append(
            {
                "gene_id": gene_id,
                "pos_mean": positive_mean,
                "pos_no1": positive_best,
                "p_value": float(p_value),
                "-log(p_value)": -math.log10(float(p_value)),
                "voted_subnet": voted_subnet,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"Uncorrected p-values: {args.output.resolve()}")


if __name__ == "__main__":
    main()

