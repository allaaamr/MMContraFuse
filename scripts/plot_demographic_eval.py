#!/usr/bin/env python3
"""
Generate shareable plots from results/demographic_eval outputs:
  1) Bar chart of c-index per cohort
  2) Calibration curves (risk_mean vs event_rate) for all cohorts
  3) Risk-score histograms for each cohort
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_metrics(folder: Path) -> pd.DataFrame:
    rows = []
    for path in folder.glob("*_metrics.json"):
        with open(path) as f:
            rec = json.load(f)
        rec["cohort"] = path.name.replace("_metrics.json", "")
        rows.append(rec)
    if not rows:
        raise SystemExit(f"No *_metrics.json files found in {folder}")
    return pd.DataFrame(rows).sort_values("cohort")


def load_calibration(folder: Path) -> dict[str, pd.DataFrame]:
    cal = {}
    for path in folder.glob("*_calibration.csv"):
        cal[path.name.replace("_calibration.csv", "")] = pd.read_csv(path)
    if not cal:
        raise SystemExit(f"No *_calibration.csv files found in {folder}")
    return cal


def load_predictions(folder: Path) -> dict[str, pd.DataFrame]:
    preds = {}
    for path in folder.glob("*_predictions.csv"):
        df = pd.read_csv(path)
        preds[path.name.replace("_predictions.csv", "")] = df
    if not preds:
        raise SystemExit(f"No *_predictions.csv files found in {folder}")
    return preds


def plot_cindex_bar(metrics: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.bar(metrics["cohort"], metrics["c_index"], color="steelblue")
    plt.ylabel("c-index")
    plt.ylim(0.8, 1.0)
    plt.title("Genomic+2.5D risk model by demographic cohort")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = out_dir / "cindex_bar.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[write] {path}")


def plot_calibration(calibration: dict[str, pd.DataFrame], out_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    for cohort, df in calibration.items():
        plt.plot(df["risk_mean"], df["event_rate"], marker="o", label=cohort)
    plt.xlabel("Mean predicted risk (per bin)")
    plt.ylabel("Observed event rate")
    plt.title("Calibration curves by cohort")
    plt.legend()
    plt.tight_layout()
    path = out_dir / "calibration_curves.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[write] {path}")


def plot_risk_hist(predictions: dict[str, pd.DataFrame], out_dir: Path) -> None:
    cohorts = sorted(predictions.keys())
    cols = 2
    rows = (len(cohorts) + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(8, 3 * rows), squeeze=False)
    for ax, cohort in zip(axes.ravel(), cohorts):
        ax.hist(predictions[cohort]["risk_score"], bins=20, color="tab:blue", alpha=0.8)
        ax.set_title(cohort)
        ax.set_xlabel("Risk score")
        ax.set_ylabel("Count")
    # hide unused axes
    for ax in axes.ravel()[len(cohorts):]:
        ax.set_visible(False)
    fig.suptitle("Risk score distribution per cohort")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / "risk_histograms.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[write] {path}")


def plot_risk_group_bars(predictions: dict[str, pd.DataFrame], out_dir: Path) -> None:
    cohorts = sorted(predictions.keys())
    cols = 2
    rows = (len(cohorts) + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(8, 3 * rows), squeeze=False)
    groups = [0, 1, 2, 3]
    for ax, cohort in zip(axes.ravel(), cohorts):
        df = predictions[cohort]
        counts = df["disc_label"].value_counts().reindex(groups, fill_value=0)
        ax.bar(groups, counts.values, color="tab:orange")
        ax.set_title(cohort)
        ax.set_xlabel("Risk group")
        ax.set_ylabel("Count")
        ax.set_xticks(groups)
    for ax in axes.ravel()[len(cohorts):]:
        ax.set_visible(False)
    fig.suptitle("Risk group distribution (labels 0-3)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / "risk_group_bars.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[write] {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot demographic evaluation summaries.")
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path("results") / "demographic_eval",
        help="Directory containing *_metrics.json, *_calibration.csv, *_predictions.csv files.",
    )
    args = parser.parse_args()

    folder = args.folder
    if not folder.exists():
        raise SystemExit(f"{folder} does not exist")

    metrics = load_metrics(folder)
    calibration = load_calibration(folder)
    predictions = load_predictions(folder)

    plot_cindex_bar(metrics, folder)
    plot_calibration(calibration, folder)
    plot_risk_hist(predictions, folder)
    plot_risk_group_bars(predictions, folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
