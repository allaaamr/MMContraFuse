#!/usr/bin/env python3
"""
Aggregate demographic fairness metrics from subset evaluation outputs.

This script expects a folder that contains the *_predictions.csv,
*_metrics.json, and *_calibration.csv artifacts emitted by
scripts/eval_radio_2p5d.py. It computes:

  * Cohort summary table (counts, event/censor rates, mean risk, Brier, ECE)
  * Fairness metrics relative to the pooled cohort (parity gap, TPR/FPR gaps)
  * Risk-coverage curves (event rate and mean risk as coverage decreases)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def load_predictions(folder: Path) -> Dict[str, pd.DataFrame]:
    preds = {}
    for path in folder.glob("*_predictions.csv"):
        preds[path.name.replace("_predictions.csv", "")] = pd.read_csv(path)
    if not preds:
        raise SystemExit(f"No *_predictions.csv under {folder}")
    return preds


def load_metrics(folder: Path) -> Dict[str, Dict]:
    metrics = {}
    for path in folder.glob("*_metrics.json"):
        with open(path) as f:
            metrics[path.name.replace("_metrics.json", "")] = json.load(f)
    return metrics


def load_calibration(folder: Path) -> Dict[str, pd.DataFrame]:
    cal = {}
    for path in folder.glob("*_calibration.csv"):
        df = pd.read_csv(path)
        if "calibration_gap" not in df.columns:
            df["calibration_gap"] = df["event_rate"] - df["risk_mean"]
        cal[path.name.replace("_calibration.csv", "")] = df
    return cal


def compute_summary(
    predictions: Dict[str, pd.DataFrame],
    metrics: Dict[str, Dict],
    calibration: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for cohort, df in predictions.items():
        events = df["event_indicator"].values
        cens = df["censorship"].values
        risk = df["risk_score"].values
        n = len(df)
        event_rate = float(events.mean())
        censor_rate = float(cens.mean())
        mean_risk = float(risk.mean())
        std_risk = float(risk.std())
        brier = float(np.mean((risk - events) ** 2))
        cal_df = calibration.get(cohort)
        if cal_df is not None and cal_df["count"].sum() > 0:
            ece = float(np.sum(np.abs(cal_df["calibration_gap"]) * cal_df["count"]) / cal_df["count"].sum())
        else:
            ece = float("nan")
        rows.append({
            "cohort": cohort,
            "count": int(n),
            "event_rate": event_rate,
            "censor_rate": censor_rate,
            "mean_risk": mean_risk,
            "std_risk": std_risk,
            "brier_score": brier,
            "ece": ece,
            "c_index": metrics.get(cohort, {}).get("c_index"),
        })
    return pd.DataFrame(rows).sort_values("cohort")


def compute_fairness(summary: pd.DataFrame, predictions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    total = summary["count"].sum()
    overall_mean_risk = (summary["mean_risk"] * summary["count"]).sum() / total
    overall_event_rate = (summary["event_rate"] * summary["count"]).sum() / total

    # Threshold at overall median risk for TPR/FPR comparison
    risks_all = np.concatenate([predictions[c]["risk_score"].values for c in predictions])
    events_all = np.concatenate([predictions[c]["event_indicator"].values for c in predictions]).astype(bool)
    non_events_all = ~events_all
    global_threshold = float(np.median(risks_all))
    if events_all.any():
        overall_tpr = float(((risks_all >= global_threshold) & events_all).sum() / events_all.sum())
    else:
        overall_tpr = float("nan")
    if non_events_all.any():
        overall_fpr = float(((risks_all >= global_threshold) & non_events_all).sum() / non_events_all.sum())
    else:
        overall_fpr = float("nan")

    rows = []
    for _, row in summary.iterrows():
        cohort = row["cohort"]
        df = predictions[cohort]
        risk = df["risk_score"].values
        events = df["event_indicator"].values.astype(bool)
        non_events = ~events
        if events.any():
            tpr = float(((risk >= global_threshold) & events).sum() / events.sum())
        else:
            tpr = float("nan")
        if non_events.any():
            fpr = float(((risk >= global_threshold) & non_events).sum() / non_events.sum())
        else:
            fpr = float("nan")
        rows.extend([
            {
                "metric": "demographic_parity",
                "cohort": cohort,
                "value": row["mean_risk"],
                "gap_vs_overall": row["mean_risk"] - overall_mean_risk,
            },
            {
                "metric": "event_rate",
                "cohort": cohort,
                "value": row["event_rate"],
                "gap_vs_overall": row["event_rate"] - overall_event_rate,
            },
            {
                "metric": "tpr_at_median_risk",
                "cohort": cohort,
                "value": tpr,
                "gap_vs_overall": tpr - overall_tpr if not np.isnan(tpr) else float("nan"),
            },
            {
                "metric": "fpr_at_median_risk",
                "cohort": cohort,
                "value": fpr,
                "gap_vs_overall": fpr - overall_fpr if not np.isnan(fpr) else float("nan"),
            },
        ])
    fairness_df = pd.DataFrame(rows)
    return fairness_df


def compute_risk_coverage(predictions: Dict[str, pd.DataFrame], points: int) -> pd.DataFrame:
    coverages = np.linspace(0.2, 1.0, points)
    rows: List[Dict] = []
    for cohort, df in predictions.items():
        risk = df["risk_score"].values
        events = df["event_indicator"].values
        order = np.argsort(risk)
        for cov in coverages:
            k = max(1, int(np.ceil(len(df) * cov)))
            idx = order[-k:]
            rows.append({
                "cohort": cohort,
                "coverage": float(k / len(df)),
                "risk_threshold": float(risk[idx].min()),
                "mean_risk": float(risk[idx].mean()),
                "event_rate": float(events[idx].mean()),
                "count": int(k),
            })
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize demographic fairness metrics.")
    p.add_argument(
        "--folder",
        type=Path,
        default=Path("results") / "demographic_eval",
        help="Directory containing subset evaluation outputs.",
    )
    p.add_argument(
        "--coverage-points",
        type=int,
        default=9,
        help="Number of coverage samples between 20% and 100%.",
    )
    return p


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.folder.exists():
        raise SystemExit(f"{args.folder} does not exist")

    predictions = load_predictions(args.folder)
    metrics = load_metrics(args.folder)
    calibration = load_calibration(args.folder)

    summary = compute_summary(predictions, metrics, calibration)
    fairness = compute_fairness(summary, predictions)
    coverage = compute_risk_coverage(predictions, args.coverage_points)

    summary_path = args.folder / "demographic_summary.csv"
    fairness_path = args.folder / "fairness_metrics.csv"
    coverage_path = args.folder / "risk_coverage.csv"

    summary.to_csv(summary_path, index=False)
    fairness.to_csv(fairness_path, index=False)
    coverage.to_csv(coverage_path, index=False)

    print(f"[write] {summary_path}")
    print(f"[write] {fairness_path}")
    print(f"[write] {coverage_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
