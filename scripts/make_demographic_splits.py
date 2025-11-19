#!/usr/bin/env python3

"""
Utility to split a clinical/genomic CSV into demographic-specific subsets.

Example:
    python scripts/make_demographic_splits.py \
        data/processed_tabular_data/mut_cna_177_patients.csv \
        --outdir data/demographic_splits \
        --age-threshold 54
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create demographic CSV slices.")
    p.add_argument(
        "csvs",
        nargs="+",
        help="One or more CSV files to split.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("data") / "demographic_splits",
        help="Directory to write the subset CSVs into.",
    )
    p.add_argument(
        "--gender-col",
        default="gender",
        help="Column containing binary gender labels (0=male, 1=female by preprocess.py convention).",
    )
    p.add_argument(
        "--age-col",
        default="age",
        help="Column containing numeric ages.",
    )
    p.add_argument(
        "--age-threshold",
        type=float,
        default=None,
        help=(
            "Split ages at this value. "
            "If omitted, the script will compute the percentile specified by --age-percentile."
        ),
    )
    p.add_argument(
        "--age-percentile",
        type=float,
        default=50.0,
        help="When --age-threshold is not provided, use this percentile of the non-null ages.",
    )
    return p


def _infer_age_threshold(series: pd.Series, percentile: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        raise SystemExit("No numeric ages found to determine the threshold.")
    return float(np.percentile(values, percentile))


def _write_subset(df: pd.DataFrame, mask: pd.Series, path: Path) -> None:
    subset = df.loc[mask].copy()
    subset.to_csv(path, index=False)
    print(f"[write] {path}  (rows={len(subset)})")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)

    for csv_path_str in args.csvs:
        csv_path = Path(csv_path_str)
        if not csv_path.exists():
            print(f"[skip] {csv_path} does not exist", file=sys.stderr)
            continue

        df = pd.read_csv(csv_path)
        base = csv_path.stem
        print(f"\nProcessing {csv_path} (rows={len(df)})")

        # Gender splits (preprocess.py encodes 0=Male, 1=Female)
        if args.gender_col in df.columns:
            gender_series = pd.to_numeric(df[args.gender_col], errors="coerce")
            male_mask = gender_series.eq(0)
            female_mask = gender_series.eq(1)
            _write_subset(df, male_mask, args.outdir / f"{base}_male.csv")
            _write_subset(df, female_mask, args.outdir / f"{base}_female.csv")
        else:
            print(f"[warn] column '{args.gender_col}' not in {csv_path.name}; skipping gender split.")

        # Age splits
        if args.age_col in df.columns:
            ages = pd.to_numeric(df[args.age_col], errors="coerce")
            if args.age_threshold is None:
                threshold = _infer_age_threshold(ages, args.age_percentile)
                print(f"[info] Derived age threshold={threshold:.2f} from p{args.age_percentile}")
            else:
                threshold = args.age_threshold
                print(f"[info] Using provided age threshold={threshold}")

            young_mask = ages <= threshold
            old_mask = ages > threshold
            _write_subset(df, young_mask, args.outdir / f"{base}_age_le_{threshold:.1f}.csv")
            _write_subset(df, old_mask, args.outdir / f"{base}_age_gt_{threshold:.1f}.csv")
        else:
            print(f"[warn] column '{args.age_col}' not in {csv_path.name}; skipping age split.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
