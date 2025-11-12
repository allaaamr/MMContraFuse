from argparse import Namespace
from collections import OrderedDict
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
import numpy as np
from sksurv.metrics import concordance_index_censored
import torch
from dataset import save_splits
from models.Encoder.genomic import SNN
from models.Encoder.radiomic_2p5 import Radio2p5DNet
from models.Encoder.deeprisk import Res34_2p5D_Regularized
from models.Encoder.mri_genomics import FusionFactory
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
import tqdm


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
    elif args.mode == 'radio_2.5D':
        # Choose the output size:
        #  - risk: n_classes = number of discrete time bins (you pass via --n_classes)
        #  - subtype: n_classes = number of subtypes (you pass via --n_classes)
        # model = Radio2p5DNet(n_classes=args.n_classes, in_ch=1)

        model = Res34_2p5D_Regularized(
            layer_num=args.layer_num,
            num_classes=args.n_classes,
            p_slice_drop=args.p_slice_drop,
            sd_prob=args.sd_prob,
            attn_dropout=args.attn_dropout,
            p_spatial_drop=args.p_spatial_drop,
            head_hidden=args.head_hidden,
            head_dropout=args.head_dropout,
            norm=args.norm  # try "gn" first when batch_size=1
        )
    elif args.mode == "genomic_radio_2.5D":
        # model_dict = {
        #     'omic_input_dim': args.omic_input_dim,
        #     'model_size_omic': args.model_size_omic,
        #     'n_classes': args.n_classes,
        #     'radio_2.5D_params': {
        #         'layer_num': 32,
        #         'num_classes': args.n_classes,
        #         'p_slice_drop': 0.15,
        #         'sd_prob': 0.1,
        #         'attn_dropout': 0.1,
        #         'p_spatial_drop': 0.05,
        #         'head_hidden': 128,
        #         'head_dropout': 0.3,
        #         'norm': "gn"  # try "gn" first when batch_size=1
        #     },
        #     'fusion_type': args.fusion,
        #     'drop_out': args.drop_out
        # }
        model = FusionFactory.build_from_args(args, args.omic_input_dim)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
    
    model = model.to(args.device)

    use_amp = getattr(args, "amp", True) and args.device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

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

    for epoch in tqdm.tqdm(range(args.max_epochs)):
        tr_loss_main, tr_loss_total, tr_metric = train_loop(
            epoch, model, train_loader, optimizer, loss_fn, args, scaler=scaler, use_amp=use_amp
        )
        va_loss_main, va_loss_total, va_metric = validate(
            cur, epoch, model, val_loader, loss_fn, args, use_amp=use_amp
        )

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
    # Create plots folder inside results_dir
    plots_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Create a unique tag with key hyperparameters
    exp_tag = (
        f"fusion={args.fusion}"
        f"_mri={args.fuse_point_mri}"
        f"_snnk={args.fuse_k_omic if args.fuse_k_omic is not None else 'full'}"
        f"_ln={args.layer_num}"
        f"_psd={args.p_slice_drop}"
        f"_sd={args.sd_prob}"
        f"_attndrop={args.attn_dropout}"
        f"_spdrop={args.p_spatial_drop}"
        f"_norm={args.norm}"
        f"_dfuse={args.dim_fuse}"
        f"_dmodel={args.d_model}"
        f"_nhead={args.nhead}"
        f"_hhead={args.head_hidden}"
        f"_hdrop={args.head_dropout}"
    )

    # Metric plot
    epochs = range(1, len(train_metrics) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics, label=f'Train {metric_name}')
    plt.plot(epochs, val_metrics,   label=f'Validation {metric_name}')
    plt.xlabel('Epochs'); plt.ylabel(metric_name)
    plt.title(f'Train vs Validation {metric_name}')
    plt.legend()
    plt.savefig(os.path.join(
        plots_dir, f"{metric_name}_plot_fold{cur}_{args.task}_{exp_tag}.png"
    ), dpi=300, bbox_inches='tight')
    plt.close('all')

    # Loss plot
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses,   label='Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(
        plots_dir, f"loss_plot_fold{cur}_{args.task}_{exp_tag}.png"
    ), dpi=300, bbox_inches='tight')
    plt.close('all')

    return model, best_val_metric, val_loader, train_loader

# --------------------
# TRAIN LOOP 
# --------------------
def train_loop(epoch, model, loader, optimizer, loss_fn, args, gc=16, scaler=None, use_amp=False):
    model.train()
    loss_main_sum, loss_total_sum = 0.0, 0.0
    n_batches = 0

    # Survival accumulators (subset optional)
    surv_scores, surv_censors, surv_times = [], [], []
    cls_correct, cls_total = 0, 0

    for batch_idx, batch in enumerate(loader):
        if batch is None:
            continue
        data_MRI, data_WSI, data_omic, y_disc, event_time, censor, _ = batch
        data_MRI   = data_MRI.to(args.device, non_blocking=True)
        data_WSI   = data_WSI.to(args.device, non_blocking=True) if torch.is_tensor(data_WSI) else data_WSI
        data_omic  = data_omic.to(args.device, non_blocking=True)
        y_disc     = y_disc.to(args.device, non_blocking=True)
        event_time = event_time.to(args.device, non_blocking=True)
        censor     = censor.to(args.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            h = model(x_path=data_WSI, x_omic=data_omic, x_mri=data_MRI)
        
        # Cast everything the loss touches to float32
        if isinstance(loss_fn, NLLSurvLoss):  # survival
            loss = loss_fn(
                h=h.float(),
                y=y_disc.float(),
                t=event_time.float(),
                c=censor.float()
            )
        else:  # classification
            loss = loss_fn(h.float(), y_disc.long())  # logits to float32, targets long

        # h = model(x_path=data_WSI, x_omic=data_omic, x_mri=data_MRI)

        # if isinstance(loss_fn, NLLSurvLoss):
        #     loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
        # else:
        #     loss = loss_fn(h, y_disc.long())

        loss_value = float(loss.detach().cpu())
        loss_main_sum += loss_value
        loss_total_sum += loss_value
        n_batches += 1

        if isinstance(loss_fn, NLLSurvLoss):
            with torch.no_grad():
                hazards  = torch.sigmoid(h)
                survival = torch.cumprod(1 - hazards, dim=1)
                risk     = -torch.sum(survival, dim=1).cpu().numpy()
                surv_scores.append(risk)
                surv_censors.append(censor.cpu().numpy())
                surv_times.append(event_time.cpu().numpy())
        else:
            with torch.no_grad():
                preds = torch.argmax(h, dim=1)
                cls_correct += (preds == y_disc).sum().item()
                cls_total   += y_disc.numel()

        accum_loss = loss / gc
        if use_amp:
            scaler.scale(accum_loss).backward()
        else:
            accum_loss.backward()

        if (batch_idx + 1) % gc == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    if n_batches == 0:
        print(f"Epoch {epoch}: no usable training batches (all samples skipped).")
        return 0.0, 0.0, float('nan')

    loss_main = loss_main_sum / n_batches
    loss_total = loss_total_sum / n_batches

    if isinstance(loss_fn, NLLSurvLoss):
        try:
            scores  = np.concatenate(surv_scores, axis=0)
            censors = np.concatenate(surv_censors, axis=0)
            times   = np.concatenate(surv_times, axis=0)
            c_index = concordance_index_censored((1 - censors).astype(bool), times, scores, tied_tol=1e-8)[0]
            metric  = float(c_index)
            print(f'Epoch {epoch}: train_surv_loss={loss_main:.4f}, train_loss={loss_total:.4f}, train_c-index={metric:.4f}')
        except ValueError:
            metric = float('nan')
            print(f'Epoch {epoch}: train_surv_loss={loss_main:.4f}, train_loss={loss_total:.4f}, train_c-index=nan (no valid samples)')
    else:
        acc = (cls_correct / max(cls_total, 1)) if cls_total else 0.0
        metric = acc
        print(f'Epoch {epoch}: train_ce_loss={loss_main:.4f}, train_loss={loss_total:.4f}, train_acc={metric:.4f}')

    return loss_main, loss_total, metric

# --------------------
# VALIDATE 
# --------------------
@torch.no_grad()
def validate(cur, epoch, model, loader, loss_fn, args, gc=16, use_amp=False):
    model.eval()
    loss_main_sum, loss_total_sum = 0.0, 0.0
    n_batches = 0

    surv_scores, surv_censors, surv_times = [], [], []
    cls_correct, cls_total = 0, 0

    for batch_idx, batch in enumerate(loader):
        if batch is None:
            continue
        data_MRI, data_WSI, data_omic, y_disc, event_time, censor, _ = batch
        data_MRI   = data_MRI.to(args.device, non_blocking=True)
        data_WSI   = data_WSI.to(args.device, non_blocking=True) if torch.is_tensor(data_WSI) else data_WSI
        data_omic  = data_omic.to(args.device, non_blocking=True)
        y_disc     = y_disc.to(args.device, non_blocking=True)
        event_time = event_time.to(args.device, non_blocking=True)
        censor     = censor.to(args.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            h = model(x_path=data_WSI, x_omic=data_omic, x_mri=data_MRI)

        # Cast everything the loss touches to float32
        if isinstance(loss_fn, NLLSurvLoss):  # survival
            loss = loss_fn(
                h=h.float(),
                y=y_disc.float(),
                t=event_time.float(),
                c=censor.float()
            )
        else:  # classification
            loss = loss_fn(h.float(), y_disc.long())  # logits to float32, targets long
        # h = model(x_path=data_WSI, x_omic=data_omic, x_mri=data_MRI)

        # if isinstance(loss_fn, NLLSurvLoss):
        #     loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
        # else:
        #     loss = loss_fn(h, y_disc.long())

        loss_value = float(loss)
        loss_main_sum += loss_value
        loss_total_sum += loss_value
        n_batches += 1

        if isinstance(loss_fn, NLLSurvLoss):
            hazards  = torch.sigmoid(h)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(torch.cumprod(1 - torch.sigmoid(h), dim=1), dim=1).detach().cpu().numpy()
            surv_scores.append(risk)
            surv_censors.append(censor.cpu().numpy())
            surv_times.append(event_time.cpu().numpy())
        else:
            preds = torch.argmax(h, dim=1)
            cls_correct += (preds == y_disc).sum().item()
            cls_total   += y_disc.numel()

    if n_batches == 0:
        print(f'val: no usable validation batches (all samples skipped).')
        return 0.0, 0.0, float('nan')

    loss_main = loss_main_sum / n_batches
    loss_total = loss_total_sum / n_batches

    if isinstance(loss_fn, NLLSurvLoss):
        try:
            scores  = np.concatenate(surv_scores, axis=0)
            censors = np.concatenate(surv_censors, axis=0)
            times   = np.concatenate(surv_times, axis=0)
            c_index = concordance_index_censored((1 - censors).astype(bool), times, scores, tied_tol=1e-8)[0]
            metric  = float(c_index)
            print(f'val_surv_loss: {loss_main:.4f}, val_loss: {loss_total:.4f}, val_c-index: {metric:.4f}')
        except ValueError:
            metric = float('nan')
            print(f'val_surv_loss: {loss_main:.4f}, val_loss: {loss_total:.4f}, val_c-index: nan (no valid samples)')
    else:
        acc = (cls_correct / max(cls_total, 1)) if cls_total else 0.0
        metric = acc
        print(f'val_ce_loss: {loss_main:.4f}, val_loss: {loss_total:.4f}, val_acc: {metric:.4f}')

    return loss_main, loss_total, metric


