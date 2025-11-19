#!/usr/bin/env python3
"""
Evaluate each saved cross-validation fold checkpoint on its validation split.

This guards against data leakage concerns by reloading the original CSV,
applying the recorded splits, and computing c-index per fold.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from dataset import Generic_MIL_Dataset
from models.Encoder.deeprisk import Res34_2p5D_Regularized
from models.Encoder.genomic import SNN
from models.Encoder.mri_genomics import FusionFactory
from utils.loss import NLLSurvLoss
from utils.utils import get_split_loader
from sksurv.metrics import concordance_index_censored


def build_model(saved_args, omic_dim: int) -> nn.Module:
    mode = saved_args.mode
    if mode == "radio_2.5D":
        model = Res34_2p5D_Regularized(
            layer_num=saved_args.layer_num,
            num_classes=saved_args.n_classes,
            p_slice_drop=saved_args.p_slice_drop,
            sd_prob=saved_args.sd_prob,
            attn_dropout=saved_args.attn_dropout,
            p_spatial_drop=saved_args.p_spatial_drop,
            head_hidden=saved_args.head_hidden,
            head_dropout=saved_args.head_dropout,
            norm=saved_args.norm,
        )
    elif mode == "genomic":
        model = SNN(
            omic_input_dim=omic_dim,
            model_size_omic=saved_args.model_size_omic,
            n_classes=saved_args.n_classes,
        )
    elif mode == "genomic_radio_2.5D":
        model = FusionFactory.build_from_args(saved_args, omic_dim)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return model


def evaluate_loader(model: nn.Module, loader, device) -> Dict[str, float]:
    model.eval()
    loss_fn = NLLSurvLoss()
    scores, times, censors = [], [], []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            data_MRI, data_WSI, data_omic, y_disc, event_time, censor, _ = batch
            data_MRI = data_MRI.to(device, non_blocking=True)
            data_WSI = (
                data_WSI.to(device, non_blocking=True)
                if torch.is_tensor(data_WSI)
                else data_WSI
            )
            data_omic = data_omic.to(device, non_blocking=True)
            event_time = event_time.to(device, non_blocking=True)
            censor = censor.to(device, non_blocking=True)
            logits = model(x_path=data_WSI, x_omic=data_omic, x_mri=data_MRI)
            hazards = torch.sigmoid(logits)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1)
            scores.append(risk.cpu().numpy())
            times.append(event_time.cpu().numpy())
            censors.append(censor.cpu().numpy())

    if not scores:
        raise RuntimeError("Loader produced no batches to evaluate.")

    scores = np.concatenate(scores)
    times = np.concatenate(times).flatten()
    censors = np.concatenate(censors).flatten()
    events = (1.0 - censors).astype(bool)
    c_index = concordance_index_censored(events, times, scores, tied_tol=1e-8)[0]
    return {"c_index": float(c_index), "count": int(len(scores))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved CV folds.")
    parser.add_argument(
        "--args-pkl",
        type=Path,
        required=True,
        help="Path to pickle containing the training argparse Namespace.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory that holds model_fold_*.pt checkpoints.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of fold split_{i}.csv files to evaluate.",
    )
    args = parser.parse_args()

    with open(args.args_pkl, "rb") as f:
        saved_args = pickle.load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not getattr(saved_args, "cpu_only", False) else "cpu"
    )
    print(f"[device] Using {device}")

    dataset = Generic_MIL_Dataset(
        csv_path=saved_args.csv,
        mode=saved_args.mode,
        path_dir=saved_args.path_dir,
        mri_dir=saved_args.mri_dir,
        task=saved_args.task,
        shuffle=False,
        seed=saved_args.seed,
        print_info=True,
        create_split=False,
        n_splits=saved_args.k,
        patient_strat=False,
        n_bins=saved_args.n_classes,
        label_col="survival" if saved_args.task in ["risk", "survival"] else "type",
    )

    fold_metrics: List[Dict[str, float]] = []
    for fold in range(args.folds):
        split_csv = Path(saved_args.split_dir) / f"split_{fold}.csv"
        if not split_csv.exists():
            raise SystemExit(f"Missing split file: {split_csv}")

        _, val_split = dataset.return_splits(str(split_csv))
        val_loader = get_split_loader(val_split, training=False, mode=saved_args.mode)

        omic_dim = val_split.genomic_features.shape[1] if not val_split.genomic_features.empty else 0

        ckpt_path = args.results_dir / f"model_fold_{fold}.pt"
        if not ckpt_path.exists():
            raise SystemExit(f"Missing checkpoint: {ckpt_path}")

        model = build_model(saved_args, omic_dim=omic_dim).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)

        metrics = evaluate_loader(model, val_loader, device)
        metrics["fold"] = fold
        metrics["checkpoint"] = str(ckpt_path)
        fold_metrics.append(metrics)
        print(f"[fold {fold}] c-index={metrics['c_index']:.4f} (n={metrics['count']})")

    avg_c = sum(m["c_index"] for m in fold_metrics) / len(fold_metrics)
    print(f"[avg] c-index={avg_c:.4f}")

    out_path = args.results_dir / "cv_eval_metrics.json"
    with open(out_path, "w") as f:
        json.dump({"folds": fold_metrics, "avg_c_index": avg_c}, f, indent=2)
    print(f"[write] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
