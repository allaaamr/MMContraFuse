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
        df = pd.read_csv(path)
        if "calibration_gap" not in df.columns:
            df["calibration_gap"] = df["event_rate"] - df["risk_mean"]
        cal[path.name.replace("_calibration.csv", "")] = df
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


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def plot_cindex_bar(metrics: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.bar(metrics["cohort"], metrics["c_index"], color="steelblue")
    plt.ylabel("c-index")
    min_c = metrics["c_index"].min()
    plt.ylim(max(0.0, min_c - 0.05), 1.0)
    plt.title("Genomic+2.5D risk model by demographic cohort")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = out_dir / "cindex_bar.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[write] {path}")


def plot_brier_bar(metrics: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.bar(metrics["cohort"], metrics["brier_score"], color="slategray")
    plt.ylabel("Brier score (lower is better)")
    ymin = max(0.0, metrics["brier_score"].min() - 0.01)
    ymax = metrics["brier_score"].max() + 0.01
    plt.ylim(ymin, ymax)
    plt.title("Calibration (Brier score) by cohort")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = out_dir / "brier_bar.png"
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


def plot_calibration_residuals(calibration: dict[str, pd.DataFrame], out_dir: Path) -> None:
    cohorts = sorted(calibration.keys())
    cols = 2
    rows = (len(cohorts) + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(8, 3 * rows), squeeze=False)
    for ax, cohort in zip(axes.ravel(), cohorts):
        df = calibration[cohort]
        ax.bar(df["bin"], df["calibration_gap"], color="tab:green")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(cohort)
        ax.set_xlabel("Bin")
        ax.set_ylabel("obs - pred")
    for ax in axes.ravel()[len(cohorts):]:
        ax.set_visible(False)
    fig.suptitle("Calibration gap per cohort (event_rate - risk_mean)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / "calibration_gaps.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
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


def plot_event_censor(summary: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    width = 0.35
    idx = range(len(summary))
    plt.bar(idx, summary["event_rate"], width=width, label="event rate", color="tab:green")
    plt.bar([i + width for i in idx], summary["censor_rate"], width=width, label="censor rate", color="tab:red")
    plt.xticks([i + width / 2 for i in idx], summary["cohort"], rotation=20, ha="right")
    plt.ylabel("Rate")
    plt.title("Event vs. censor rate per cohort")
    plt.legend()
    plt.tight_layout()
    path = out_dir / "event_censor_rates.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[write] {path}")


def plot_parity_gaps(fairness: pd.DataFrame, out_dir: Path) -> None:
    parity = fairness[fairness["metric"] == "demographic_parity"]
    if parity.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.bar(parity["cohort"], parity["gap_vs_overall"], color="mediumpurple")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.ylabel("Gap vs. overall mean risk")
    plt.title("Demographic parity gap")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = out_dir / "demographic_parity_gap.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[write] {path}")


def plot_risk_coverage_curves(coverage: pd.DataFrame, out_dir: Path) -> None:
    if coverage is None or coverage.empty:
        return
    plt.figure(figsize=(6, 4))
    for cohort, df in coverage.groupby("cohort"):
        ordered = df.sort_values("coverage")
        plt.plot(ordered["coverage"], ordered["event_rate"], marker="o", label=cohort)
    plt.xlabel("Coverage (fraction of patients kept)")
    plt.ylabel("Observed event rate in kept set")
    plt.title("Risk-coverage curves")
    plt.legend()
    plt.tight_layout()
    path = out_dir / "risk_coverage.png"
    plt.savefig(path, dpi=200)
    plt.close()
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
    summary = load_optional_csv(folder / "demographic_summary.csv")
    fairness = load_optional_csv(folder / "fairness_metrics.csv")
    coverage = load_optional_csv(folder / "risk_coverage.csv")

    plot_cindex_bar(metrics, folder)
    plot_brier_bar(metrics, folder)
    plot_calibration(calibration, folder)
    plot_calibration_residuals(calibration, folder)
    plot_risk_hist(predictions, folder)
    plot_risk_group_bars(predictions, folder)
    if summary is not None:
        plot_event_censor(summary, folder)
    if fairness is not None:
        plot_parity_gaps(fairness, folder)
    if coverage is not None:
        plot_risk_coverage_curves(coverage, folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
