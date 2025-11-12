from __future__ import print_function
import argparse
import gc
import os
import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from torch.utils.data import DataLoader  # noqa

from dataset import Generic_MIL_Dataset
from utils.utils import *
from utils.core_utils import train

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def _restrict_to_genomics_intersection(ds: Generic_MIL_Dataset) -> None:
    """
    In-place: keep only cases with *usable* genomics rows so MRI-only results
    are directly comparable to fusion/genomics runs.

    We treat a case as genomics-available if all genomics feature columns are present
    (non-null) on that row. Adjust the null rule if your dataset encodes missingness
    differently.
    """
    before = len(ds.slide_data)

    # Genomics columns = everything that's not metadata (same logic as dataset build)
    # ds.genomic_features aligns 1:1 with ds.slide_data rows
    gfeats = ds.genomic_features

    # Keep rows where ALL genomics features are non-null
    ok = gfeats.notnull().all(axis=1)

    trimmed = ds.slide_data.loc[ok].copy().reset_index(drop=True)
    if trimmed.empty:
        print("[WARN] Cohort intersection is empty after trimming to genomics-available cases. "
              "Keeping original dataset.", flush=True)
        return

    # Rebuild patient_dict
    patient_dict = {}
    by_case = trimmed.set_index("case_id")
    for case in trimmed["case_id"].drop_duplicates():
        slide_ids = by_case.loc[case, "slide_id"]
        if isinstance(slide_ids, str):
            slide_ids = np.array([slide_ids])
        else:
            slide_ids = slide_ids.values
        patient_dict[case] = slide_ids

    # Commit updates
    ds.slide_data = trimmed
    ds.patient_dict = patient_dict

    # Refresh derived fields to stay consistent
    ds.metadata = getattr(ds, "metadata", list(trimmed.columns[:0]))
    ds.genomic_features = ds.slide_data.drop(ds.metadata, axis=1)
    ds.patient_data_prep()
    ds.cls_ids_prep()

    after = len(ds.slide_data)
    n_pat = len(np.unique(ds.slide_data["case_id"]))
    print(f"[INFO] Cohort matched to genomics availability: rows {before} → {after} | "
          f"unique patients = {n_pat}", flush=True)


def _restrict_to_mri_2p5d_intersection(ds: Generic_MIL_Dataset, mri_dir: str) -> None:
    """
    Mutates the dataset 'ds' in-place so that it only contains patients (case_id)
    for whom a 2.5D MRI file '<case_id>.npy' exists in mri_dir.

    After filtering:
      - slide_data is reduced to the intersection cohort
      - patient_dict is rebuilt
      - patient_data & cls id indices are recomputed
      - genomic_features is refreshed

    This keeps split CSVs usable: any rows missing in the trimmed slide_data
    will just be dropped when masks are applied inside return_splits().
    """
    # Collect available MRI case_ids (2.5D expects <case_id>.npy)
    try:
        files = [f for f in os.listdir(mri_dir) if f.endswith(".npy")]
    except FileNotFoundError:
        files = []
    mri_case_ids = set(os.path.splitext(f)[0] for f in files)

    if len(mri_case_ids) == 0:
        print(f"[WARN] No 2.5D MRI files (*.npy) found in {mri_dir}. "
              f"Skipping cohort intersection trimming.", flush=True)
        return

    before = len(ds.slide_data)
    # Keep only rows whose case_id is in the MRI set
    mask = ds.slide_data["case_id"].isin(mri_case_ids)
    trimmed = ds.slide_data[mask].copy().reset_index(drop=True)

    if len(trimmed) == 0:
        print(f"[WARN] Cohort intersection is empty after trimming to MRI-available cases. "
              f"Keeping original dataset.", flush=True)
        return

    # Rebuild patient_dict: {case_id -> slide_ids}
    patient_dict = {}
    by_case = trimmed.set_index("case_id")
    unique_cases = trimmed["case_id"].drop_duplicates().tolist()
    for case in unique_cases:
        slide_ids = by_case.loc[case, "slide_id"]
        if isinstance(slide_ids, str):
            slide_ids = np.array([slide_ids])
        else:
            slide_ids = slide_ids.values
        patient_dict[case] = slide_ids

    # Commit updates to dataset
    ds.slide_data = trimmed
    ds.patient_dict = patient_dict

    # Refresh derived fields
    ds.metadata = getattr(ds, "metadata", list(trimmed.columns[:0]))  # keep existing metadata list
    ds.genomic_features = ds.slide_data.drop(ds.metadata, axis=1)
    ds.patient_data_prep()
    ds.cls_ids_prep()

    after = len(ds.slide_data)
    n_pat = len(np.unique(ds.slide_data["case_id"]))
    print(f"[INFO] Cohort matched to 2.5D MRI: rows {before} → {after} | unique patients = {n_pat}", flush=True)


def main(args):
    os.makedirs(args.results_dir, exist_ok=True)

    best_metric = -float('inf')
    best_fold = -1
    best_ckpt_path = None
    per_fold_metrics = []
    
    # check if cuda is being used
    print("Using device:", device)
    sys.stdout.flush()
    # 5-fold CV (uses precomputed CSV splits in args.split_dir)
    for i in range(5):
        # ---- Build train/val datasets for this fold ----
        train_dataset, val_dataset = dataset.return_splits(
            csv_path=f'{args.split_dir}/split_{i}.csv'
        )
        print(f'[Fold {i}] training: {len(train_dataset)}, validation: {len(val_dataset)}')
        sys.stdout.flush()

        # ---- Genomics input dim for this fold ----
        args.omic_input_dim = train_dataset.genomic_features.shape[1]
        print(f"[Fold {i}] Genomic Dimension: {args.omic_input_dim}")
        sys.stdout.flush()

        # ---- Train this fold ----
        model, val_metric, val_loader, train_loader = train(
            (train_dataset, val_dataset), i, args
        )

        # ---- Save only the model weights for this fold ----
        fold_ckpt = os.path.join(args.results_dir, f'model_fold_{i}.pt')
        torch.save(model.state_dict(), fold_ckpt)
        per_fold_metrics.append((i, float(val_metric), fold_ckpt))

        # ---- Track best ----
        if val_metric > best_metric:
            best_metric = float(val_metric)
            best_fold = i
            best_ckpt_path = fold_ckpt

        # ---- FREE MEMORY: drop big references & collect ----
        del model, train_loader, val_loader, train_dataset, val_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Persist best info + metrics table ----
    print(f'Best fold: {best_fold} | metric={best_metric:.4f} | ckpt={best_ckpt_path}')
    metrics_csv = os.path.join(args.results_dir, 'fold_metrics.csv')
    with open(metrics_csv, 'w') as f:
        f.write('fold,metric,ckpt\n')
        for i, m, p in per_fold_metrics:
            f.write(f'{i},{m},{p}\n')

    # Copy best checkpoint to canonical name
    if best_ckpt_path is not None:
        dst = os.path.join(args.results_dir, f'best_model_{args.mode}.pt')
        if dst != best_ckpt_path:
            import shutil
            shutil.copyfile(best_ckpt_path, dst)

    # Print average metric
    if per_fold_metrics:
        avg = sum(m for _, m, _ in per_fold_metrics) / len(per_fold_metrics)
        print('Average metric across folds:', avg)


# ------------------ Argparse ------------------
parser = argparse.ArgumentParser(description='Configurations for Analysis on TCGA Data.')

# Checkpoint + Misc. Pathing Parameters
parser.add_argument('--env', type=str, default='server')
parser.add_argument('--xai', action='store_true', help="Enable XAI (e.g., SHAP, IG) analysis")

parser.add_argument('--path_dir', type=str, default='path/to/data_root_dir',
                    help='Data directory to WSI features (extracted via CLAM)')
parser.add_argument('--csv', type=str, default='data/processed_tabular_data/rna_clinical.csv',
                    help='directory to clinical and genomics csv file')
parser.add_argument('--mri_dir', type=str, default='data/2.5D_MRIs',
                    help='directory to MRI data')

parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--k', type=int, default=5, help='Number of folds')
parser.add_argument('--k_start', type=int, default=-1, help='Start fold (unused)')
parser.add_argument('--k_end', type=int, default=-1, help='End fold (unused)')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='Results directory')
parser.add_argument('--split_dir', type=str, default='data/splits',
                    help='Directory containing split_{i}.csv files')

# Model Parameters
parser.add_argument('--task', type=str, choices=['subtype', 'risk'], default='risk',
                    help='Downstream task')
parser.add_argument('--mode', type=str,
                    choices=['genomic', 'path', 'radio_1D', 'radio_2.5D', 'radio_3D',
                             'pathomic', 'radiomic1D', 'radiomic2.5D', 'radiomic3D',
                             'radiopathomics', 'genomic_radio_2.5D'],
                    default='genomic', help='Which modalities to use')
parser.add_argument('--fusion', type=str,
                    choices=['concat', 'bi_attn', 'tri_attn', 'bi_contrast', 'tri_contrast'],
                    default='concat', help='Type of fusion')
parser.add_argument('--drop_out', action='store_true', default=True, help='Enable dropout (p=0.25)')
parser.add_argument('--model_size_wsi', type=str, default='small')
parser.add_argument('--model_size_omic', type=str, default='small')
parser.add_argument('--n_classes', type=int, default=4)

parser.add_argument('--gate_path', action='store_true', default=False)
parser.add_argument('--gate_omic', action='store_true', default=False)
parser.add_argument('--gate_radio', action='store_true', default=False)
parser.add_argument('--scale_dim1', type=int, default=8)
parser.add_argument('--scale_dim2', type=int, default=8)
parser.add_argument('--scale_dim3', type=int, default=8)
parser.add_argument('--skip', action='store_true', default=False)
parser.add_argument('--dropinput', type=float, default=0.0)
parser.add_argument('--path_input_dim', type=int, default=1024)
parser.add_argument('--use_mlp', action='store_true', default=False)

# Optimizer + Survival Loss
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size', type=int, default=1, help='Default 1 due to varying bag sizes')
parser.add_argument('--gc', type=int, default=32, help='Gradient accumulation steps')
parser.add_argument('--max_epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv'],
                    default='nll_surv')
parser.add_argument('--loss', type=str, choices=['cox', 'nll'], default='nll')
parser.add_argument('--label_frac', type=float, default=1.0)
parser.add_argument('--reg', type=float, default=1e-5, help='L2 weight decay')
parser.add_argument('--alpha_surv', type=float, default=0.0)
parser.add_argument('--reg_type', type=str, choices=['None', 'omic', 'pathomic'], default='None')
parser.add_argument('--lambda_reg', type=float, default=1e-5)
parser.add_argument('--weighted_sample', action='store_true', default=True)
parser.add_argument('--early_stopping', action='store_true', default=False)
parser.add_argument('--data', type=str, default='cna_mut_df')
parser.add_argument('--patience', type=int, default=8)
parser.add_argument('--create_split', action='store_true', default=False)

# 2.5D MRI network params
parser.add_argument('--layer_num', type=int, default=32,
                    help="Number of MRI slices (channels) expected by the 2.5D network.")
parser.add_argument('--p_slice_drop', type=float, default=0.15,
                    help="Probability of randomly dropping MRI slices during training (SliceDrop).")
parser.add_argument('--sd_prob', type=float, default=0.1,
                    help="Stochastic depth probability.")
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help="Dropout probability inside MRI attention block.")
parser.add_argument('--p_spatial_drop', type=float, default=0.05,
                    help="Spatial Dropout2d probability.")
parser.add_argument('--norm', type=str, choices=['gn', 'in'], default='gn',
                    help="Normalization type for MRI conv blocks.")

# Fusion params
parser.add_argument('--fuse_point_mri', type=str, choices=['stem', 'block1', 'gap'], default='gap',
                    help="Where to extract MRI features for fusion.")
parser.add_argument('--fuse_k_omic', type=int, default=None,
                    help="How many SNN (genomics) layers to run before fusion (1..N).")
parser.add_argument('--dim_fuse', type=int, default=256,
                    help="Hidden dimension of fused representation for concat/gated fusion.")
parser.add_argument('--d_model', type=int, default=256,
                    help="Dimension for cross-attention fusion.")
parser.add_argument('--nhead', type=int, default=4,
                    help="Attention heads for cross-attention fusion.")
parser.add_argument('--head_hidden', type=int, default=256,
                    help="Hidden size of the final classification head MLP.")
parser.add_argument('--head_dropout', type=float, default=0.3,
                    help="Dropout in the final classification head.")

# NEW: ensure unimodal genomics uses the exact 2.5D MRI cohort
parser.add_argument('--match_mri2p5d_cohort', action='store_true', default=True,
                    help="If True and mode==genomic, restrict dataset to cases with <case_id>.npy in --mri_dir.")
parser.add_argument('--match_genomics_cohort', action='store_true', default=True,
    help="If True and mode==radio_2.5D, restrict dataset to cases with complete genomics rows.")

parser.add_argument('--amp', action='store_true', default=True,
                    help='Enable mixed precision when CUDA is available.')
parser.add_argument('--cpu_only', action='store_true', default=False,
                    help='Force CPU even if CUDA is available (debug).')

args = parser.parse_args()



if args.cpu_only:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] Using: {device}")
if device.type == "cuda":
    print(f"[DEVICE] CUDA device: {torch.cuda.get_device_name(0)}")


args.device = device


# ------------------ Repro ------------------
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(args.seed)

# ------------------ Label selection ------------------
# Depending on the downstream task the label column to predict is:
if args.task in ["risk", "survival"]:
    label_col = "survival"   # continuous survival time / risk target
else:  # subtype classification
    label_col = "type"

# ------------------ Dataset ------------------
dataset = Generic_MIL_Dataset(csv_path=args.csv,
                              mode=args.mode,
                              path_dir=args.path_dir,
                              mri_dir=args.mri_dir,
                              task=args.task,
                              shuffle=False,
                              seed=args.seed,
                              print_info=True,
                              create_split=args.create_split,
                              n_splits=5,
                              patient_strat=False,
                              n_bins=args.n_classes,
                              label_col=label_col)

# If we're doing a genomics-only run but want it directly comparable to fusion with 2.5D,
# shrink cohort to only cases that have 2.5D MRI available.
if args.mode == "genomic" and args.match_mri2p5d_cohort:
    print("[INFO] Matching genomics cohort to 2.5D MRI availability...", flush=True)
    _restrict_to_mri_2p5d_intersection(dataset, args.mri_dir)

# If we're doing an MRI-only run but want it directly comparable to fusion/genomics,
# shrink cohort to only cases that have complete genomics available.
if args.mode == "radio_2.5D" and args.match_genomics_cohort:
    print("[INFO] Matching 2.5D MRI cohort to genomics availability...", flush=True)
    _restrict_to_genomics_intersection(dataset)


# ------------------ Run ------------------
if __name__ == "__main__":
    start = timer()
    results = main(args)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))
