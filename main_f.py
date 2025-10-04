from __future__ import print_function
import numpy as np
import argparse
import os
import sys
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import Generic_MIL_Dataset, create_dataset
from utils.utils import *
from utils.core_utils import train, validate
from utils.train_contrg import train_contig
from utils.downstream import evaluate_downstream_task

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

"""
Unified training framework supporting both standard multi-modal fusion and contrastive pre-training
"""


def main(args):
    """
    Main training function supporting two modes:
    1. Standard: Direct supervised training on labeled data
    2. Contrastive: Contrastive pre-training + downstream fine-tuning
    """
    
    dataset = create_dataset(args)



    if args.training_mode == 'contrastive':
        print("=" * 50)
        print("CONTRASTIVE LEARNING MODE")
        print("=" * 50)
        results = train_contrastive_pipeline(dataset, args)
    else:
        print("=" * 50)
        print("STANDARD TRAINING MODE")
        print("=" * 50)
        results = train_standard_pipeline(dataset, args)
    
    return results


def train_standard_pipeline(dataset, args):
    """Standard supervised training across k-fold cross-validation"""
    
    models = []
    metrics = []
    val_splits = []
    val_loaders = []
    train_splits = []
    train_loaders = []
    
    # K-fold cross-validation
    k_start = 0 if args.k_start == -1 else args.k_start
    k_end = args.k if args.k_end == -1 else args.k_end
    
    for fold in range(k_start, k_end):
        print(f"\n{'=' * 50}")
        print(f"FOLD {fold + 1}/{args.k}")
        print(f"{'=' * 50}")
        
        # Get train/val splits
        train_dataset, val_dataset = dataset.return_splits(
            csv_path=f'{args.split_dir}/split_{fold}.csv'
        )
        
        print(f'Training: {len(train_dataset)}, Validation: {len(val_dataset)}')
        
        # Set input dimensions
        args.omic_input_dim = train_dataset.genomic_features.shape[1]
        print(f"Genomic Dimension: {args.omic_input_dim}")
        args.radio_input_dim = train_dataset.radiomics_features.shape[1]
        print(f"Radiomics Dimension: {args.radio_input_dim}")
        
        # Train model
        model, metric, val_loader, train_loader = train(
            (train_dataset, val_dataset), fold, args)
        
        # Store results
        models.append(model)
        metrics.append(metric)
        val_splits.append(val_dataset)
        val_loaders.append(val_loader)
        train_splits.append(train_dataset)
        train_loaders.append(train_loader)
    
    # Find and save best model
    save_best_model(models, metrics, val_loaders, val_splits, 
                   train_loaders, train_splits, args)
    
    # Print summary
    print_summary(metrics, args)
    
    return {
        'models': models,
        'metrics': metrics,
        'val_splits': val_splits,
        'train_splits': train_splits
    }


def train_contrastive_pipeline(dataset, args):
    """Contrastive pre-training followed by downstream evaluation"""
    
    print("\n" + "=" * 50)
    print("Phase 1: Contrastive Pre-training")
    print("=" * 50)
    
    models = []
    metrics = []
    val_splits = []
    val_loaders = []
    train_splits = []
    train_loaders = []
    
    
    for fold in range(0, 5):
        train_dataset, val_dataset = dataset.return_splits(
            csv_path=f'{args.split_dir}/split_{fold}.csv'
        )
        
        print(f'Training: {len(train_dataset)}, Validation: {len(val_dataset)}')
        
        args.omic_input_dim = val_dataset.genomic_features.shape[1]
        print(f"Genomic Dimension: {args.omic_input_dim}")
        
        args.radio_input_dim = val_dataset.radiomics_features.shape[1] -1
        print(f"Radiomics Dimension: {args.radio_input_dim}")
        
        # Create data loaders
        train_loader = get_split_loader(
            train_dataset, 
            training=True,
            weighted=args.weighted_sample,
            mode=args.mode,
            batch_size=args.batch_size
        )
        
        val_loader = get_split_loader(
            val_dataset,
            mode=args.mode,
            batch_size=args.batch_size
        )
        
    # ---- Either train or load ----
        config = {
            "omic_input_dim": args.omic_input_dim,
            "radio_input_dim": args.radio_input_dim,
            "genomics_output_dim": args.genomics_output_dim,
            "radiomics_output_dim": args.radiomics_output_dim,
            "projection_dim": args.projection_dim,
            "temperature": args.temperature,
            "radio_type": "1D",
        }
        if args.contrg_mode == 'load':
            print(f"[ContRG] Loading checkpoint from {args.contrg_ckpt}")
            contrg_model = load_contrg_from_ckpt(config, args.contrg_ckpt)
        else:
            print("[ContRG] Training from scratch...")
            contrg_model = train_contig(
                train_loader=train_loader,
                val_loader=val_loader,
                **config,
                learning_rate=args.lr,
                weight_decay=args.reg,
                max_epochs=args.contrastive_epochs  # use the CLI value
            )
            # # Save contrastive model
            save_path = os.path.join(args.results_dir, f"contrg_model_{args.mode}.pt")
            torch.save(contrg_model.state_dict(), save_path)
            print(f"Saved contrastive model to {save_path}")
        
        print("\n" + "=" * 50)
        print("Phase 2: Downstream Task Evaluation")
        print("=" * 50)
        
        # Evaluate on downstream task
        results = {}
        for freeze_mode in [True, False]:
            mode_name = "Linear Probing" if freeze_mode else "Fine-tuning"
            print(f"\n{mode_name}...")
            
            model, va_metric, train_loader, val_loader  = evaluate_downstream_task(
                args=args,
                contrg_model=contrg_model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=args.n_classes,
                task=args.task,
                freeze_encoder=freeze_mode,
                num_epochs=args.max_epochs
            )
        
        results[mode_name.lower().replace(' ', '_')] = {
            'classifier': model,
            'val_metric': va_metric
        }
            
    
    return {
        'contrg_model': contrg_model,
        'downstream_results': results
    }


def save_best_model(models, metrics, val_loaders, val_splits, 
                   train_loaders, train_splits, args):
    """Save the best model and associated data"""
    
    best_fold = np.argmax(metrics)
    best_model = models[best_fold]
    
    # Save model and data
    torch.save(best_model.state_dict(), 
              os.path.join(args.results_dir, f"best_model_{args.mode}.pt"))
    
    # Save loaders and splits
    save_pickle(val_loaders[best_fold], 
               os.path.join(args.results_dir, f"val_loader_{args.mode}.pkl"))
    save_pickle(val_splits[best_fold],
               os.path.join(args.results_dir, f"val_split_{args.mode}.pkl"))
    save_pickle(train_loaders[best_fold],
               os.path.join(args.results_dir, f"train_loader_{args.mode}.pkl"))
    save_pickle(train_splits[best_fold],
               os.path.join(args.results_dir, f"train_split_{args.mode}.pkl"))
    save_pickle(args,
               os.path.join(args.results_dir, f"args_{args.mode}.pkl"))
    
    print(f"\nBest model from fold {best_fold} saved.")


def save_pickle(obj, filepath):
    """Helper to save pickle files"""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def print_summary(metrics, args):
    """Print training summary"""
    metric_name = "c-index" if args.task == "risk" else "accuracy"
    
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    
    for i, m in enumerate(metrics):
        print(f"Fold {i}: {metric_name} = {m:.4f}")
    
    print(f"Average {metric_name}: {np.mean(metrics):.4f} ± {np.std(metrics):.4f}")


def seed_torch(seed=7):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_parser():
    """Create argument parser with all options"""
    parser = argparse.ArgumentParser(
        description='Unified Multi-Modal Fusion with Optional Contrastive Learning'
    )
    
    # Training mode
    parser.add_argument('--training_mode', type=str, 
                       choices=['standard', 'contrastive'], default='contrastive',
                       help='Training mode: standard supervised or contrastive pre-training')
    
    # Data paths
    parser.add_argument('--path_dir', type=str, 
                       help='Data directory to WSI features')
    parser.add_argument('--csv', type=str, default="cna_177_patients.csv",
                       help='Path to clinical and genomics csv file')
    parser.add_argument('--mri_dir', type=str, default='data/2.5D_MRIs',
                       help='Directory to MRI data')
    parser.add_argument('--split_dir', type=str, default='data/splits',
                       help='Directory containing train/val splits')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Results directory')
    
    # Task and modality
    parser.add_argument('--task', type=str, choices=['subtype', 'risk'], 
                       default='risk', help='Downstream task')
    parser.add_argument('--mode', type=str, default='genomic',
                       choices=['genomic', 'path', 'radio_1D', 'radio_2.5D', 
                               'radio_3D', 'pathomic', 'radiomic1D', 'radiomic2.5D',
                               'radiomic3D', 'radiopathomics'],
                       help='Which modalities to use')
    parser.add_argument('--fusion', type=str, default='concat',
                       choices=['concat', 'bi_attn', 'tri_attn', 
                               'bi_contrast', 'tri_contrast'],
                       help='Fusion strategy')
    
    # Model architecture
    parser.add_argument('--model_size_wsi', type=str, default='small')
    parser.add_argument('--model_size_omic', type=str, default='small')
    parser.add_argument('--n_classes', type=int, default=4)
    parser.add_argument('--drop_out', action='store_true', default=True)
    parser.add_argument('--path_input_dim', type=int, default=1024)
    
    # Training parameters
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--k', type=int, default=5, help='Number of folds')
    parser.add_argument('--k_start', type=int, default=-1)
    parser.add_argument('--k_end', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gc', type=int, default=32, help='Gradient accumulation')
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--reg', type=float, default=1e-5)
    parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
    
    # Loss functions
    parser.add_argument('--bag_loss', type=str, default='nll_surv',
                       choices=['svm', 'ce', 'ce_surv', 'nll_surv'])
    parser.add_argument('--loss', type=str, choices=['cox', 'nll'], default='nll')
    parser.add_argument('--alpha_surv', type=float, default=0.0)
    
    # Regularization
    parser.add_argument('--reg_type', type=str, default='None',
                       choices=['None', 'omic', 'pathomic'])
    parser.add_argument('--lambda_reg', type=float, default=1e-5)
    
    # Training options
    parser.add_argument('--weighted_sample', action='store_true', default=True)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--create_split', action='store_true', default=False)
    parser.add_argument('--label_frac', type=float, default=1.0)
    
    # Contrastive learning specific
    parser.add_argument('--contrastive_epochs', type=int, default=100,
                       help='Number of epochs for contrastive pre-training')
    parser.add_argument('--downstream_epochs', type=int, default=50,
                       help='Number of epochs for downstream task')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for contrastive loss')
    parser.add_argument('--projection_dim', type=int, default=128,
                       help='Dimension of projection head')
    parser.add_argument('--genomics_output_dim', type=int, default=256)
    parser.add_argument('--radiomics_output_dim', type=int, default=64)
    
    # Misc
    parser.add_argument('--data', type=str, default='cna_mut_df')
    parser.add_argument('--env', type=str, default='server')
    parser.add_argument('--xai', action='store_true', help='Enable XAI analysis')
    
    parser.add_argument(
    '--contrg_mode', type=str, default='train',
    choices=['train', 'load'],
    help="train: run contrastive pre-training; load: skip pre-training and use a saved checkpoint"
    )
    parser.add_argument(
        '--contrg_ckpt', type=str, default='results/contrg_model_radiomic1D.pt',
        help="Path to a saved ContRG state_dict (.pt) to load when contrg_mode=load"
    )
    return parser


if __name__ == "__main__":
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")
    
    # Set random seed
    seed_torch(args.seed)
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Run training
    start = timer()
    results = main(args)
    end = timer()
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print(f"Total time: {(end - start)/60:.2f} minutes")
    print("=" * 50)