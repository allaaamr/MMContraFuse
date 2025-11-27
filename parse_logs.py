#!/usr/bin/env python3
import argparse, re, os, glob, csv
from typing import Optional, Tuple

HP_RE = re.compile(
    r"Params:\s+adv_end=([0-9.]+)\s+warmup=([0-9]+)\s+prox=([0-9.eE\-]+)\s+cons=([0-9.eE\-]+)\s+tag=([^\s]+)"
)

FOLD_RE = re.compile(r"^\[Fold\s+(\d+)\]")
BASELINE_FLAG_RE = re.compile(r"\[Pretrain validation: baseline bias\]")
VAL_BIAS_RE = re.compile(
    r"\[VAL bias\]\s+young_ci=([0-9.]+)\s+old_ci=([0-9.]+)\s+gap\(old-young\)=([\-0-9.]+)"
)

def parse_log(path: str):
    """
    Yields records:
      {
        'tag','adv_end','warmup','prox','cons',
        'fold','phase','young_ci','old_ci','gap','src'
      }
    with phase in {'before','after'}.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Parse hyperparameters once per file
    hp = None
    for ln in lines:
        m = HP_RE.search(ln)
        if m:
            hp = {
                "adv_end": float(m.group(1)),
                "warmup": int(m.group(2)),
                "prox": m.group(3),
                "cons": m.group(4),
                "tag": m.group(5),
            }
            break
    if hp is None:
        # Fallback: try to infer from results_dir line if present, else mark unknown
        hp = {"adv_end": None, "warmup": None, "prox": None, "cons": None, "tag": "unknown"}

    # State over folds
    cur_fold: Optional[int] = None
    expect_baseline_next = False
    baseline_done_for_fold = False

    # Track best (smallest |gap|) during training per fold
    best_train: dict[int, Tuple[float, float, float]] = {}  # fold -> (young, old, gap)
    baseline: dict[int, Tuple[float, float, float]] = {}    # fold -> (young, old, gap)

    def maybe_emit_fold_end(fold_id: int):
        # nothing to do here—records are written at the end after parsing all lines
        pass

    for ln in lines:
        # Fold change?
        m_fold = FOLD_RE.match(ln.strip())
        if m_fold:
            # close previous fold if any
            if cur_fold is not None:
                maybe_emit_fold_end(cur_fold)
            cur_fold = int(m_fold.group(1))
            expect_baseline_next = False
            baseline_done_for_fold = False
            continue

        # Baseline flag?
        if BASELINE_FLAG_RE.search(ln):
            expect_baseline_next = True
            continue

        # Any [VAL bias] line?
        m_bias = VAL_BIAS_RE.search(ln)
        if m_bias and cur_fold is not None:
            y = float(m_bias.group(1))
            o = float(m_bias.group(2))
            g = float(m_bias.group(3))

            if expect_baseline_next and not baseline_done_for_fold:
                baseline[cur_fold] = (y, o, g)
                baseline_done_for_fold = True
                expect_baseline_next = False
            else:
                # training-phase bias snapshot—track best abs gap
                prev = best_train.get(cur_fold)
                if (prev is None) or (abs(g) < abs(prev[2])):
                    best_train[cur_fold] = (y, o, g)

    # Build rows
    rows = []
    for fold in range(5):
        b = baseline.get(fold)
        a = best_train.get(fold)
        if b is not None:
            rows.append({
                "tag": hp["tag"],
                "adv_end": hp["adv_end"],
                "warmup": hp["warmup"],
                "prox": hp["prox"],
                "cons": hp["cons"],
                "fold": fold,
                "phase": "before",
                "young_ci": b[0],
                "old_ci": b[1],
                "gap": b[2],
                "src": os.path.basename(path),
            })
        if a is not None:
            rows.append({
                "tag": hp["tag"],
                "adv_end": hp["adv_end"],
                "warmup": hp["warmup"],
                "prox": hp["prox"],
                "cons": hp["cons"],
                "fold": fold,
                "phase": "after",
                "young_ci": a[0],
                "old_ci": a[1],
                "gap": a[2],
                "src": os.path.basename(path),
            })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", required=True, help="Directory containing Slurm logs (*.out)")
    ap.add_argument("--pattern", default="*.out", help="Glob pattern inside logs_dir (default: *.out)")
    ap.add_argument("--out_csv", required=True, help="Path to write the compiled CSV")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.logs_dir, args.pattern)))
    if not paths:
        raise SystemExit(f"No log files matched {args.logs_dir}/{args.pattern}")

    all_rows = []
    for p in paths:
        try:
            rows = parse_log(p)
            all_rows.extend(rows)
        except Exception as e:
            print(f"[WARN] Failed parsing {p}: {e}")

    # Optional: sanity filter to the expected 40 rows if you want strictness
    # but we'll keep everything we find; you can post-filter by tag/fold later.

    # Write CSV
    fieldnames = ["tag","adv_end","warmup","prox","cons","fold","phase","young_ci","old_ci","gap","src"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"Wrote {len(all_rows)} rows to {args.out_csv}")
    import pandas as pd

    df = pd.read_csv(args.out_csv)

    # Absolute gap for fairness comparison
    df["abs_gap"] = df["gap"].abs()

    # Aggregate across folds per tag and phase
    summary = (
        df.groupby(["tag", "phase"])
        .agg(mean_young_ci=("young_ci", "mean"),
            mean_gap=("abs_gap", "mean"))
        .unstack("phase")
    )

    # Flatten MultiIndex columns
    summary.columns = [f"{col}_{phase}" for col, phase in summary.columns]
    summary = summary.reset_index()

    # Compute gap reduction and young retention
    summary["gap_reduction"] = summary["mean_gap_before"] - summary["mean_gap_after"]
    summary["young_retention"] = summary["mean_young_ci_after"] / summary["mean_young_ci_before"]

    # Sort: prefer higher gap reduction and retention
    summary = summary.sort_values(["gap_reduction", "young_retention"], ascending=[False, False])

    print(summary[[
        "tag",
        "mean_gap_before", "mean_gap_after", "gap_reduction",
        "mean_young_ci_before", "mean_young_ci_after", "young_retention"
    ]])
if __name__ == "__main__":
    main()
    
