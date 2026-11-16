"""Apply Bonferroni correction and export Module 1 candidate genes."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, default=Path("results/module1"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    if "p_value" not in frame.columns:
        raise ValueError("p_value column not found in input file")
    frame["p_value"] = (frame["p_value"] * len(frame)).clip(upper=1.0)
    frame["-log(p_value)"] = frame["p_value"].apply(
        lambda value: 300.0 if value <= 0 else -math.log10(value)
    )
    frame.drop(columns=["pos_mean", "pos_no1"], errors="ignore", inplace=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corrected = args.output_dir / "corrected_pvalues.csv"
    significant = args.output_dir / "significant_genes.csv"
    frame.to_csv(corrected, index=False)
    frame[frame.p_value < args.threshold].to_csv(significant, index=False)
    print(f"Corrected p-values: {corrected.resolve()}")
    print(f"Significant genes: {significant.resolve()}")


if __name__ == "__main__":
    main()

