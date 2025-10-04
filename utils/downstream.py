from models.Fusion.DownStream import DownstreamModel
import torch
import torch.nn as nn
from utils.core_utils import train_loop, validate
from utils.loss import NLLSurvLoss
from lifelines.utils import concordance_index
import numpy as np
from sksurv.metrics import concordance_index_censored
import sys

from utils.utils import get_optim

# ============================================================================
# Downstream Task Evaluation
# ============================================================================

def evaluate_downstream_task(
    args,
    contrg_model,
    train_loader,
    val_loader,
    num_classes,
    task='risk',
    freeze_encoder=True,
    num_epochs=50
):
    """
    Evaluate on downstream tasks using learned embeddings
    
    Args:
        args: Arguments namespace with all configurations
        contrg_model: Pre-trained ContRG model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_classes: Number of classes for the task
        task: 'risk' for survival or 'subtype' for classification
        freeze_encoder: If True, only train classifier (linear probing)
        num_epochs: Number of training epochs
    
    Returns:
        Trained downstream model and best validation metric
    """
    
    print('\nInit downstream model...', end=' ')
    model = DownstreamModel(contrg_model, num_classes, freeze_encoder)
    model = model.to(args.device)
    print('Done!')
    
    print('\nInit loss function...', end=' ')
    if task == 'risk':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        metric_name = "c-index"
    elif task == 'subtype':
        loss_fn = nn.CrossEntropyLoss()
        metric_name = "accuracy"
    else:
        raise ValueError(f"Unknown task: {task}")
    print('Done!')
    
    print('\nInit optimizer...', end=' ')
    # Only optimize classifier parameters if doing linear probing
    if freeze_encoder:
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
    else:
        optimizer = get_optim(model, args)
    print('Done!')
    

    train_metrics, val_metrics = [], []
    train_losses, val_losses = [], []
    
    mode_name = "Linear Probing" if freeze_encoder else "Fine-tuning"
    print(f'\n{"="*50}')
    print(f'{mode_name} for {num_epochs} epochs')
    print(f'{"="*50}')
    
    # Training loop
    for epoch in range(num_epochs):
        # Train
        tr_loss_main, tr_loss_total, tr_metric = train_loop(
            epoch, model, train_loader, optimizer, loss_fn, args
        )
        
        # Validate
        va_loss_main, va_loss_total, va_metric = validate(
            0, epoch, model, val_loader, loss_fn, args
        )
        
        # Track metrics
        train_metrics.append(tr_metric)
        val_metrics.append(va_metric)
        train_losses.append(tr_loss_total)
        val_losses.append(va_loss_total)
        
      
    # Create plots
    plot_downstream_results(train_metrics, val_metrics, train_losses, val_losses, 
                           metric_name, mode_name, args)
    
    return model

def plot_downstream_results(train_metrics, val_metrics, train_losses, val_losses, 
                           metric_name, mode_name, args):
    """
    Create plots for downstream task results
    """
    import os
    
    # Create plots directory
    plots_dir = os.path.join(args.results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    epochs = range(1, len(train_metrics) + 1)
    
    # Metrics plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics, 'b-', label=f'Train {metric_name}')
    plt.plot(epochs, val_metrics, 'r-', label=f'Val {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{mode_name}: Train vs Validation {metric_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, f'{mode_name.lower().replace(" ", "_")}_{metric_name}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{mode_name}: Train vs Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, f'{mode_name.lower().replace(" ", "_")}_loss.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def compare_downstream_results(results_dict, task, args):
    """
    Compare linear probing vs fine-tuning results
    
    Args:
        results_dict: Dictionary with 'linear_probing' and 'fine_tuning' results
        task: 'risk' or 'subtype'
        args: Arguments namespace
    """
    import pandas as pd
    
    metric_name = "c-index" if task == 'risk' else "accuracy"
    
    # Create comparison table
    comparison = pd.DataFrame({
        'Method': ['Linear Probing', 'Fine-tuning'],
        f'Best {metric_name}': [
            results_dict['linear_probing']['best_metric'],
            results_dict['fine_tuning']['best_metric']
        ]
    })
    
    print("\n" + "="*50)
    print("DOWNSTREAM TASK COMPARISON")
    print("="*50)
    print(comparison.to_string(index=False))
    print("="*50)
    
    # Save comparison
    comparison_path = os.path.join(args.results_dir, 'downstream_comparison.csv')
    comparison.to_csv(comparison_path, index=False)
    print(f"\nComparison saved to {comparison_path}")
    
    return comparison