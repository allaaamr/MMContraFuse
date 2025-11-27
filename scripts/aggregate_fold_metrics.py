#!/usr/bin/env python3
"""
Aggregate per-fold demographic metrics and optionally emit summary plots.

Expected folder layout:
  root/
    fold_0/
      tag_metrics.json
    fold_1/
      tag_metrics.json
    ...

The script groups files by the cohort tag suffix (e.g.,
"gtf_fold0_female" -> "female") so we can average metrics across folds.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate demographic metrics across folds.")
    parser.add_argument(
        "--fold-root",
        type=Path,
        required=True,
        help="Directory containing fold_* subdirectories with *_metrics.json files.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Directory to write summary CSVs/plots into.",
    )
    parser.add_argument(
        "--cohort-suffix",
        default=None,
        help=(
            "Optional suffix extractor. If provided, the script will split each tag "
            "at the last occurrence of this suffix and use the trailing portion as the cohort key. "
            "By default it uses everything after the last '_' (underscore)."
        ),
    )
    return parser.parse_args()


def extract_cohort(tag: str, suffix: str | None) -> str:
    if suffix and suffix in tag:
        idx = tag.rfind(suffix)
        return tag[idx + len(suffix):]
    match = re.search(r"_fold\d+_", tag)
    if match:
        return tag[match.end():]
    if "_" in tag:
        return tag.rsplit("_", maxsplit=1)[-1]
    return tag


def prettify_cohort(name: str) -> str:
    if name.startswith("age_le"):
        return "young"
    if name.startswith("age_gt"):
        return "old"
    return name


def main() -> int:
    args = parse_args()
    if not args.fold_root.exists():
        raise SystemExit(f"{args.fold_root} does not exist.")
    args.outdir.mkdir(parents=True, exist_ok=True)

    metrics_files = sorted(args.fold_root.glob("fold_*/*_metrics.json"))
    if not metrics_files:
        raise SystemExit(f"No *_metrics.json files found under {args.fold_root}")

    grouped: Dict[str, List[Dict]] = {}
    for path in metrics_files:
        with open(path) as f:
            rec = json.load(f)
        tag = rec.get("tag")
        if not tag:
            tag = Path(path).name.replace("_metrics.json", "")
        cohort = extract_cohort(tag, args.cohort_suffix)
        grouped.setdefault(cohort, []).append(rec)

    rows = []
    for cohort, recs in grouped.items():
        def avg(key: str) -> float:
            vals = [r[key] for r in recs if key in r]
            return float(np.mean(vals)) if vals else float("nan")

        row = {
            "cohort": prettify_cohort(cohort),
            "count": len(recs),
            "c_index_mean": avg("c_index"),
            "c_index_std": float(np.std([r["c_index"] for r in recs if "c_index" in r], ddof=0))
            if any("c_index" in r for r in recs) else float("nan"),
            "brier_mean": avg("brier_score"),
            "risk_mean": avg("risk_mean"),
            "risk_std": avg("risk_std"),
        }
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("cohort")
    summary_path = args.outdir / "folds_demographic_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[write] {summary_path}")

    # Plot averaged c-index bar
    plt.figure(figsize=(6, 4))
    plt.bar(summary["cohort"], summary["c_index_mean"], yerr=summary["c_index_std"], color="steelblue", capsize=4)
    plt.ylabel("c-index (mean ± std)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plot_path = args.outdir / "folds_cindex_bar.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[write] {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
