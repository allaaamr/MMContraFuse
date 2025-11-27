#!/usr/bin/env python3
"""
Evaluate a trained radio 2.5D checkpoint on a demographic CSV subset.

Outputs:
  * metrics JSON summarising c-index and counts
  * per-patient predictions CSV (risk score, censor flag, survival time)
  * calibration table CSV (mean predicted risk vs observed event rate per bin)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataset import Generic_MIL_Dataset, Generic_Split
from models.Encoder.deeprisk import Res34_2p5D_Regularized
from models.Encoder.genomic import SNN
from models.Encoder.mri_genomics import FusionFactory
from utils.loss import NLLSurvLoss
from utils.utils import get_split_loader
from sksurv.metrics import concordance_index_censored


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a 2.5D checkpoint on a CSV subset.")
    p.add_argument("--csv", required=True, help="Clinical/genomic CSV to evaluate.")
    p.add_argument("--ckpt", required=True, help="Path to the trained .pt checkpoint.")
    p.add_argument("--mode", default="radio_2.5D",
                   choices=["radio_2.5D", "radiomic", "genomic", "genomic_radio_2.5D"],
                   help="Model branch to instantiate.")
    p.add_argument("--task", choices=["risk", "subtype"], default="risk")
    p.add_argument("--n-classes", type=int, default=4, help="Number of discrete survival bins / classes.")
    p.add_argument("--path-dir", default="path/to/data_root_dir")
    p.add_argument("--mri-dir", default="data/2.5D_MRIs")
    p.add_argument("--results-dir", type=Path, default=Path("results") / "subset_eval")
    p.add_argument("--tag", default=None, help="Optional name for output artifacts.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-calib-bins", type=int, default=5)
    p.add_argument("--cpu-only", action="store_true", help="Force CPU inference.")
    p.add_argument("--amp", action="store_true", default=True, help="Use AMP when CUDA is available.")
    p.add_argument("--match-genomics-cohort", action="store_true", default=True,
                   help="Restrict radio_2.5D to cases with genomics rows (keeps parity with fusion runs).")
    p.add_argument("--match-mri2p5d-cohort", action="store_true", default=True,
                   help="Restrict genomics runs to patients with 2.5D MRIs.")

    # Survival hyper-parameters
    p.add_argument("--alpha-surv", type=float, default=0.0, help="Alpha for NLLSurvLoss.")

    # MRI hyper-parameters (must match training run)
    p.add_argument("--layer-num", type=int, default=32)
    p.add_argument("--p-slice-drop", type=float, default=0.15)
    p.add_argument("--sd-prob", type=float, default=0.1)
    p.add_argument("--attn-dropout", type=float, default=0.1)
    p.add_argument("--p-spatial-drop", type=float, default=0.05)
    p.add_argument("--head-hidden", type=int, default=256)
    p.add_argument("--head-dropout", type=float, default=0.3)
    p.add_argument("--norm", choices=["gn", "in"], default="gn")

    # Fusion hyper-parameters
    p.add_argument("--fusion", choices=['concat', 'bi_attn', 'tri_attn', 'bi_contrast', 'tri_contrast', 'bilinear'],
                   default='bilinear')
    p.add_argument("--drop-out", action="store_true", default=True)
    p.add_argument("--dim-fuse", type=int, default=256)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--head-hidden-fusion", type=int, default=256)
    p.add_argument("--head-dropout-fusion", type=float, default=0.3)
    p.add_argument("--fuse-point-mri", choices=["stem", "block1", "gap"], default="gap")
    p.add_argument("--fuse-k-omic", type=int, default=None)
    p.add_argument("--scale-dim1", type=int, default=8,
                   help="Scale dim for MRI branch inside RadiomicMMF.")
    p.add_argument("--scale-dim2", type=int, default=8,
                   help="Scale dim for omics branch inside RadiomicMMF.")
    p.add_argument("--gate-path", type=int, choices=[0, 1], default=0,
                   help="Enable gating on MRI branch (RadiomicMMF).")
    p.add_argument("--gate-omic", type=int, choices=[0, 1], default=0,
                   help="Enable gating on omics branch (RadiomicMMF).")
    p.add_argument("--skip-fusion", action="store_true", default=False,
                   help="Enable skip concatenation for RadiomicMMF.")
    p.add_argument("--model-size-omic", choices=["small", "big"], default="small")

    return p


def build_model(args, omic_dim: int) -> torch.nn.Module:
    from models.Fusion.GatedTensorFusion import RadiomicMMF

    if args.mode == "radio_2.5D":
        model = Res34_2p5D_Regularized(
            layer_num=args.layer_num,
            num_classes=args.n_classes,
            p_slice_drop=args.p_slice_drop,
            sd_prob=args.sd_prob,
            attn_dropout=args.attn_dropout,
            p_spatial_drop=args.p_spatial_drop,
            head_hidden=args.head_hidden,
            head_dropout=args.head_dropout,
            norm=args.norm
        )
    elif args.mode == "radiomic":
        model = RadiomicMMF(
            omic_input_dim=omic_dim,
            n_classes=args.n_classes,
            fusion=args.fusion,
            scale_dim1=args.scale_dim1,
            scale_dim2=args.scale_dim2,
            gate_path=bool(args.gate_path),
            gate_omic=bool(args.gate_omic),
            skip=args.skip_fusion,
            model_size_omic=args.model_size_omic,
            layer_num=args.layer_num
        )
    elif args.mode == "genomic":
        model = SNN(
            omic_input_dim=omic_dim,
            model_size_omic="small",
            n_classes=args.n_classes
        )
    elif args.mode == "genomic_radio_2.5D":
        model = FusionFactory.build_from_args(args, omic_dim)
    else:
        raise ValueError(f"Unsupported mode {args.mode}")
    return model


def restrict_to_genomics_intersection(dataset: Generic_MIL_Dataset) -> None:
    before = len(dataset.slide_data)
    gfeats = dataset.genomic_features
    ok = gfeats.notnull().all(axis=1)
    trimmed = dataset.slide_data.loc[ok].copy().reset_index(drop=True)
    if trimmed.empty:
        print("[warn] No rows remain after genomics intersection; keeping original dataset.")
        return
    patient_dict = {}
    by_case = trimmed.set_index("case_id")
    for case in trimmed["case_id"].drop_duplicates():
        slide_ids = by_case.loc[case, "slide_id"]
        if isinstance(slide_ids, str):
            slide_ids = np.array([slide_ids])
        else:
            slide_ids = slide_ids.values
        patient_dict[case] = slide_ids
    dataset.slide_data = trimmed
    dataset.patient_dict = patient_dict
    dataset.genomic_features = dataset.slide_data.drop(dataset.metadata, axis=1)
    dataset.patient_data_prep()
    dataset.cls_ids_prep()
    after = len(dataset.slide_data)
    print(f"[INFO] Genomics intersection reduced rows {before} -> {after}")


def restrict_to_mri_intersection(dataset: Generic_MIL_Dataset, mri_dir: str) -> None:
    try:
        files = [f for f in os.listdir(mri_dir) if f.endswith(".npy")]
    except FileNotFoundError:
        files = []
    case_ids = set(os.path.splitext(f)[0] for f in files)
    if not case_ids:
        print(f"[warn] No MRI files found under {mri_dir}; skipping restriction.")
        return
    before = len(dataset.slide_data)
    mask = dataset.slide_data["case_id"].isin(case_ids)
    trimmed = dataset.slide_data[mask].copy().reset_index(drop=True)
    if trimmed.empty:
        print("[warn] MRI intersection removed all rows; keeping original dataset.")
        return
    patient_dict = {}
    by_case = trimmed.set_index("case_id")
    for case in trimmed["case_id"].drop_duplicates():
        slide_ids = by_case.loc[case, "slide_id"]
        if isinstance(slide_ids, str):
            slide_ids = np.array([slide_ids])
        else:
            slide_ids = slide_ids.values
        patient_dict[case] = slide_ids
    dataset.slide_data = trimmed
    dataset.patient_dict = patient_dict
    dataset.genomic_features = dataset.slide_data.drop(dataset.metadata, axis=1)
    dataset.patient_data_prep()
    dataset.cls_ids_prep()
    after = len(dataset.slide_data)
    print(f"[INFO] MRI intersection reduced rows {before} -> {after}")


def restrict_dataset(dataset: Generic_MIL_Dataset, args) -> None:
    if args.mode in ("radio_2.5D", "radiomic") and args.match_genomics_cohort:
        restrict_to_genomics_intersection(dataset)
    elif args.mode == "genomic" and args.match_mri2p5d_cohort:
        restrict_to_mri_intersection(dataset, args.mri_dir)


def make_subset_split(dataset: Generic_MIL_Dataset, args) -> Generic_Split:
    split = Generic_Split(
        slide_data=dataset.slide_data.copy(),
        metadata=dataset.metadata,
        mode=args.mode,
        mri_data_dir=args.mri_dir,
        data_dir=args.path_dir,
        label_col=dataset.label_col,
        patient_dict=dataset.patient_dict,
        num_classes=dataset.num_classes
    )
    if not split.genomic_features.empty:
        scalers = split.get_scaler()
        split.apply_scaler(scalers)
    return split


def evaluate(model, loader, loss_fn, device, amp=True) -> Dict[str, float]:
    model.eval()
    all_scores: List[float] = []
    all_times: List[float] = []
    all_censors: List[float] = []
    all_disc: List[int] = []
    all_slide_ids: List[str] = []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            data_mri, data_wsi, data_omic, y_disc, event_time, censor, slide_ids = batch
            data_mri = data_mri.to(device, non_blocking=True)
            data_wsi = data_wsi.to(device, non_blocking=True) if torch.is_tensor(data_wsi) else data_wsi
            data_omic = data_omic.to(device, non_blocking=True)
            y_disc = y_disc.to(device, non_blocking=True)
            event_time = event_time.to(device, non_blocking=True)
            censor = censor.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                logits = model(x_path=data_wsi, x_omic=data_omic, x_mri=data_mri)

            if isinstance(loss_fn, NLLSurvLoss):
                hazards = torch.sigmoid(logits)
                survival = torch.cumprod(1 - hazards, dim=1)
                risk = -torch.sum(survival, dim=1)
                all_scores.extend(risk.detach().cpu().numpy().tolist())
            else:
                probs = torch.softmax(logits, dim=1)
                risk = 1.0 - probs[:, 0]  # crude "risk" proxy
                all_scores.extend(risk.detach().cpu().numpy().tolist())

            all_times.extend(event_time.detach().cpu().numpy().flatten().tolist())
            all_censors.extend(censor.detach().cpu().numpy().flatten().tolist())
            all_disc.extend(y_disc.detach().cpu().numpy().flatten().astype(int).tolist())
            # Each batch returns slide_ids for the patient; use the first entry
            if isinstance(slide_ids, (list, tuple, np.ndarray)):
                batch_ids = slide_ids
            else:
                batch_ids = [slide_ids]
            for sid in batch_ids:
                if isinstance(sid, (np.ndarray, list)) and len(sid):
                    all_slide_ids.append(str(sid[0]))
                else:
                    all_slide_ids.append(str(sid))

    if not all_scores:
        raise RuntimeError("No samples were evaluated (all batches skipped).")

    return {
        "scores": np.asarray(all_scores),
        "times": np.asarray(all_times),
        "censors": np.asarray(all_censors),
        "disc": np.asarray(all_disc, dtype=int),
        "slide_ids": np.asarray(all_slide_ids, dtype=str),
    }


def build_calibration_table(scores: np.ndarray, censors: np.ndarray, num_bins: int) -> pd.DataFrame:
    order = np.argsort(scores)
    bins = np.array_split(order, num_bins)
    rows = []
    for idx, bin_idx in enumerate(bins):
        if bin_idx.size == 0:
            continue
        s_bin = scores[bin_idx]
        c_bin = censors[bin_idx]
        rows.append({
            "bin": idx,
            "count": int(bin_idx.size),
            "risk_mean": float(s_bin.mean()),
            "risk_min": float(s_bin.min()),
            "risk_max": float(s_bin.max()),
            "event_rate": float((1.0 - c_bin).mean()),  # train loop uses 1-censor for events
        })
    return pd.DataFrame(rows)


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    device = torch.device("cpu" if args.cpu_only else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[device] Using {device}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (Path(args.csv).stem + "__" + Path(args.ckpt).stem)

    label_col = "survival" if args.task == "risk" else "type"
    dataset = Generic_MIL_Dataset(
        csv_path=args.csv,
        mode=args.mode,
        path_dir=args.path_dir,
        mri_dir=args.mri_dir,
        task=args.task,
        shuffle=False,
        seed=1,
        print_info=True,
        create_split=False,
        n_splits=5,
        patient_strat=False,
        n_bins=args.n_classes,
        label_col=label_col
    )
    restrict_dataset(dataset, args)
    subset_split = make_subset_split(dataset, args)

    loader = get_split_loader(subset_split, training=False, mode=args.mode, batch_size=args.batch_size)

    omic_dim = subset_split.genomic_features.shape[1] if not subset_split.genomic_features.empty else 0
    model = build_model(args, omic_dim=omic_dim).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt)

    if args.task == "risk":
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    else:
        loss_fn = nn.CrossEntropyLoss()

    outputs = evaluate(model, loader, loss_fn, device, amp=args.amp)

    # Build case_id lookup for readability
    slide_to_case = subset_split.slide_data.set_index("slide_id")["case_id"].to_dict()
    case_ids = [slide_to_case.get(sid, sid) for sid in outputs["slide_ids"]]

    events = 1.0 - outputs["censors"]
    event_indicator = events.astype(bool)
    c_index = concordance_index_censored(event_indicator, outputs["times"], outputs["scores"], tied_tol=1e-8)[0]
    brier_score = float(np.mean((outputs["scores"] - events) ** 2))

    metrics = {
        "csv": os.path.abspath(args.csv),
        "ckpt": os.path.abspath(args.ckpt),
        "mode": args.mode,
        "task": args.task,
        "count": int(len(outputs["scores"])),
        "c_index": float(c_index),
        "risk_mean": float(outputs["scores"].mean()),
        "risk_std": float(outputs["scores"].std()),
        "brier_score": brier_score,
    }

    metrics_path = args.results_dir / f"{tag}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[write] {metrics_path}")

    preds_df = pd.DataFrame({
        "case_id": case_ids,
        "slide_id": outputs["slide_ids"],
        "risk_score": outputs["scores"],
        "disc_label": outputs["disc"],
        "survival_time": outputs["times"],
        "censorship": outputs["censors"],
        "event_indicator": 1.0 - outputs["censors"],
    })
    preds_path = args.results_dir / f"{tag}_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"[write] {preds_path}")

    calib_df = build_calibration_table(outputs["scores"], outputs["censors"], args.num_calib_bins)
    calib_df["calibration_gap"] = calib_df["event_rate"] - calib_df["risk_mean"]
    calib_path = args.results_dir / f"{tag}_calibration.csv"
    calib_df.to_csv(calib_path, index=False)
    print(f"[write] {calib_path}")

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
