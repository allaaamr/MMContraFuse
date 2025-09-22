from argparse import Namespace
from collections import OrderedDict
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
import numpy as np
from sksurv.metrics import concordance_index_censored
import torch
from dataset import save_splits
from models.Encoder.genomic import SNN
# from models.healnet import HealNet
# from models.transformer_fusion import TransformerFusion
# from models.DeepRisk import Res34
# from models.radiomic_model import DenseNet2D, EfficientNet2D, ResNet3D
# from models.pathomic_model import MIL_Attention_FC_surv, PorpoiseAMIL
# from models.fusion_model import PorpoiseMMF,RadiomicMMF, PathoRadioMMF, PathoRadioGenomicMMF, GGMMF
from utils.utils import *
from utils.loss import NLLSurvLoss, CoxPHSurvLoss
import sys
import pandas as pd

import torchvision.models.video as vmodels
import torch.nn as nn

import wandb


class MRI3DHead(nn.Module):
    def __init__(self, in_ch: int, task: str, n_bins_or_classes: int):
        super().__init__()
        m = vmodels.r3d_18(weights=None)  # or weights="R3D_18_Weights.KINETICS400_V1" and adapt first conv
        # adapt first conv to match MRI channels (e.g., 2 or 4)
        m.stem[0] = nn.Conv3d(in_ch, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m
        # task head: for risk (discrete-time) output n_bins hazards; for subtype output class logits
        self.head = nn.Linear(feat_dim, n_bins_or_classes)
        self.task = task

    def forward(self, x_mri=None, x_path=None, x_omic=None, **kwargs):
        # expect x_mri shape: (N, C, D/T, H, W)
        z = self.backbone(x_mri)
        out = self.head(z)
        return out  # risk: hazard logits per bin; subtype: class logits

def train(datasets: tuple, cur: int, args):
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split = datasets
    print(f"Training on {len(train_split)} samples")
    print(f"Validating on {len(val_split)} samples")

    print('\nInit loss function...', end=' ')
    if args.task == 'risk':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        metric_name = "c-index"
    elif args.task == 'subtype':
        loss_fn = nn.CrossEntropyLoss()
        metric_name = "accuracy"
    else:
        raise ValueError(f"Unknown task: {args.task}")

    print('\nInit Model...', end=' ')
    if args.mode == 'genomic':
        model_dict = {
            'omic_input_dim': args.omic_input_dim,
            'model_size_omic': args.model_size_omic,
            'n_classes': args.n_classes
        }
        model = SNN(**model_dict)
    elif args.mode == 'radio_3D':
        # infer MRI channel count from one batch (or set explicitly if you prefer)
        tmp_loader = get_split_loader(train_split, training=False, weighted=False, mode=args.mode, batch_size=1)
        with torch.no_grad():
            sample = next(iter(tmp_loader))
            x_mri = sample[0]  # (N, C, D, H, W)
            in_ch = x_mri.size(1)
        n_out = args.n_classes  # risk: #bins ; subtype: #classes
        model = MRI3DHead(in_ch=in_ch, task=args.task, n_bins_or_classes=n_out)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True,
                                    weighted=args.weighted_sample,
                                    mode=args.mode, batch_size=args.batch_size)
    val_loader   = get_split_loader(val_split, mode=args.mode, batch_size=args.batch_size)
    print('Done!')
    sys.stdout.flush()

    best_val_metric = -float('inf') if args.task == 'risk' else 0.0
    patience_counter = 0
    patience = args.patience

    train_metrics, val_metrics = [], []
    train_losses,  val_losses  = [],  []

    wandb.init(
        project="multimodal-survival",   # change to your wandb project name
        name=f"fold_{cur}_{args.mode}_{args.task}",
        config=vars(args)
    )

    for epoch in range(args.max_epochs):
        tr_loss_main, tr_loss_total, tr_metric = train_loop(
            epoch, model, train_loader, optimizer, loss_fn, args
        )
        va_loss_main, va_loss_total, va_metric = validate(
            cur, epoch, model, val_loader, loss_fn, args
        )

        wandb.log({
            "epoch": epoch,
            "train_loss_main": tr_loss_main,
            "train_loss_total": tr_loss_total,
            f"train_{metric_name}": tr_metric,
            "val_loss_main": va_loss_main,
            "val_loss_total": va_loss_total,
            f"val_{metric_name}": va_metric,
            "fold": cur
        })

        train_metrics.append(tr_metric)
        val_metrics.append(va_metric)
        train_losses.append(tr_loss_total)
        val_losses.append(va_loss_total)

        # simple early-stopping by the task metric
        improved = va_metric > best_val_metric
        if improved:
            best_val_metric = va_metric
            patience_counter = 0
        else:
            patience_counter += 1
        if patience and patience_counter >= patience:
            print(f"Early stop at epoch {epoch} (best {metric_name}: {best_val_metric:.4f})")
            break

    print(f'Val {metric_name}: {best_val_metric:.4f}')

    # --- Plots ---
    epochs = range(1, len(train_metrics) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics, label=f'Train {metric_name}')
    plt.plot(epochs, val_metrics,   label=f'Validation {metric_name}')
    plt.xlabel('Epochs'); plt.ylabel(metric_name); plt.title(f'Train vs Validation {metric_name}')
    plt.legend()
    plt.savefig(f"{metric_name}_plot_{cur}.png", dpi=300, bbox_inches='tight')

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses,   label='Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Train vs Validation Loss')
    plt.legend()
    plt.savefig(f"loss_plot_{cur}.png", dpi=300, bbox_inches='tight')

    wandb.log({f"best_val_{metric_name}": best_val_metric, "fold": cur})
    wandb.finish()  

    return model, best_val_metric, val_loader, train_loader

# --------------------
# TRAIN LOOP 
# --------------------
def train_loop(epoch, model, loader, optimizer, loss_fn, args, gc=16):
    model.train()
    loss_main_sum, loss_total_sum = 0.0, 0.0

    # Survival accumulators
    surv_scores, surv_censors, surv_times = [], [], []
    # Classification accumulators
    cls_logits, cls_targets = [], []

    for batch_idx, batch in enumerate(loader):
        data_MRI, data_WSI, data_omic, y_disc, event_time, censor, slide_ids = batch

        # Forward
        h = model(x_path=data_WSI, x_omic=data_omic, x_mri=data_MRI)

        # Loss by task
        if isinstance(loss_fn, NLLSurvLoss):  # risk / survival
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
        else:  # classification
            loss = loss_fn(h, y_disc.long())

        loss_value = float(loss.detach().cpu())
        loss_reg = 0.0  # hook if you add regularization

        # --- Metric accumulators ---
        if isinstance(loss_fn, NLLSurvLoss):
            hazards  = torch.sigmoid(h)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk     = -torch.sum(survival, dim=1).detach().cpu().numpy()  # higher risk = earlier event
            surv_scores.append(risk)
            surv_censors.append(censor.detach().cpu().numpy())
            surv_times.append(event_time.detach().cpu().numpy())
        else:
            cls_logits.append(h.detach().cpu())
            cls_targets.append(y_disc.detach().cpu())

        # Bookkeeping
        loss_main_sum  += loss_value
        loss_total_sum += loss_value + loss_reg

        # Backward / grad accumulation
        (loss / gc + loss_reg).backward()
        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (batch_idx + 1) % 50 == 0:
            sys.stdout.flush()

    # Epoch metrics
    loss_main = loss_main_sum / len(loader)
    loss_total = loss_total_sum / len(loader)

    if isinstance(loss_fn, NLLSurvLoss):
        scores  = np.concatenate(surv_scores, axis=0)
        censors = np.concatenate(surv_censors, axis=0)
        times   = np.concatenate(surv_times, axis=0)
        c_index = concordance_index_censored((1 - censors).astype(bool), times, scores, tied_tol=1e-8)[0]
        metric  = float(c_index)
        print(f'Epoch {epoch}: train_surv_loss={loss_main:.4f}, train_loss={loss_total:.4f}, train_c-index={metric:.4f}')
    else:
        logits = torch.cat(cls_logits, dim=0)
        targets = torch.cat(cls_targets, dim=0)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean().item()
        metric = acc
        print(f'Epoch {epoch}: train_ce_loss={loss_main:.4f}, train_loss={loss_total:.4f}, train_acc={metric:.4f}')

    sys.stdout.flush()
    return loss_main, loss_total, metric

# --------------------
# VALIDATE 
# --------------------
@torch.no_grad()
def validate(cur, epoch, model, loader, loss_fn, args, gc=16):
    model.eval()
    loss_main_sum, loss_total_sum = 0.0, 0.0

    surv_scores, surv_censors, surv_times = [], [], []
    cls_logits, cls_targets = [], []

    for batch_idx, (data_MRI, data_WSI, data_omic, y_disc, event_time, censor, slide_ids) in enumerate(loader):
        h = model(x_path=data_WSI, x_omic=data_omic, x_mri=data_MRI)

        if isinstance(loss_fn, NLLSurvLoss):
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
        else:
            loss = loss_fn(h, y_disc.long())

        loss_value = float(loss.detach().cpu())
        loss_reg = 0.0

        if isinstance(loss_fn, NLLSurvLoss):
            hazards  = torch.sigmoid(h)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk     = -torch.sum(survival, dim=1).detach().cpu().numpy()
            surv_scores.append(risk)
            surv_censors.append(censor.detach().cpu().numpy())
            surv_times.append(event_time.detach().cpu().numpy())
        else:
            cls_logits.append(h.detach().cpu())
            cls_targets.append(y_disc.detach().cpu())

        loss_main_sum  += loss_value
        loss_total_sum += loss_value + loss_reg

    loss_main = loss_main_sum / len(loader)
    loss_total = loss_total_sum / len(loader)

    if isinstance(loss_fn, NLLSurvLoss):
        scores  = np.concatenate(surv_scores, axis=0)
        censors = np.concatenate(surv_censors, axis=0)
        times   = np.concatenate(surv_times, axis=0)
        c_index = concordance_index_censored((1 - censors).astype(bool), times, scores, tied_tol=1e-8)[0]
        metric  = float(c_index)
        print(f'val_surv_loss: {loss_main:.4f}, val_loss: {loss_total:.4f}, val_c-index: {metric:.4f}')
    else:
        logits = torch.cat(cls_logits, dim=0)
        targets = torch.cat(cls_targets, dim=0)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean().item()
        metric = acc
        print(f'val_ce_loss: {loss_main:.4f}, val_loss: {loss_total:.4f}, val_acc: {metric:.4f}')

    sys.stdout.flush()
    return loss_main, loss_total, metric

