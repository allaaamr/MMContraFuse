from argparse import Namespace
from collections import OrderedDict
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
import numpy as np
from sksurv.metrics import concordance_index_censored
import torch
import torch.nn as nn
from dataset import save_splits
from models.Encoder.genomic import SNN
from models.Encoder.radiomic_2p5 import Radio2p5DNet
from models.Encoder.deeprisk import Res34_2p5D_Regularized
from models.Encoder.mri_genomics import FusionFactory
from models.Fusion.GatedTensorFusion import RadiomicMMF
from utils.utils import *
from utils.loss import NLLSurvLoss, CoxPHSurvLoss
import sys
import pandas as pd
import tqdm
import os

# from models.debiasing import ResidualEditor, AgeAdversary, LambdaScheduler, grl
from models.debiasing_bins import ResidualEditor, AgeAdversary, LambdaScheduler, grl
import torch.nn.functional as F

def _make_bin_onehot(y_bins, n_bins):
    # y_bins: [B] int (0..n_bins-1)
    return F.one_hot(y_bins.long(), num_classes=n_bins).float()

def _binwise_scores_surv(logits):
    hazards  = torch.sigmoid(logits)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk     = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk

def _subgroup_cindex(scores, times, censors, ages):
    import numpy as np
    ys, yo = (ages == 0), (ages == 1)
    out = {}
    for name, mask in (("young", ys), ("old", yo)):
        if mask.sum() >= 2:
            ci = concordance_index_censored((1 - censors[mask]).astype(bool), times[mask], scores[mask], tied_tol=1e-8)[0]
            out[name] = float(ci)
        else:
            out[name] = float('nan')
    out["gap_old_minus_young"] = (out["old"] - out["young"]) if np.isfinite(out.get("old", np.nan)) and np.isfinite(out.get("young", np.nan)) else float('nan')
    return out

def _subgroup_acc(logits, y_true, ages):
    import numpy as np
    preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
    y = y_true.detach().cpu().numpy()
    a = ages.detach().cpu().numpy()
    out = {}
    for name, v in (("young", 0), ("old", 1)):
        m = (a == v)
        out[name] = float((preds[m] == y[m]).mean()) if m.any() else float('nan')
    out["gap_old_minus_young"] = out["old"] - out["young"] if np.isfinite(out.get("old", np.nan)) and np.isfinite(out.get("young", np.nan)) else float('nan')
    return out


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
    elif args.mode == 'radiomic':
        model_dict = {'omic_input_dim': args.omic_input_dim, 'fusion': 'bilinear', 'n_classes': args.n_classes, 
        'gate_path': args.gate_path, 'gate_omic': args.gate_omic, 'scale_dim1': args.scale_dim1, 'scale_dim2': args.scale_dim2, 
        'skip': args.skip}
        model = RadiomicMMF(**model_dict)
    elif args.mode == "genomic_radio_2.5D":
        model = FusionFactory.build_from_args(args, args.omic_input_dim)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
    print('Done!')
    
    model = model.to(args.device)

    # ----- Load + freeze base per-fold without leakage -----
    ckpt_arg = getattr(args, 'frozen_ckpt', None)
    ckpt_path = None
    if ckpt_arg:
        if os.path.isdir(ckpt_arg):
            # Expect files named model_fold_{i}.pt
            cand = os.path.join(ckpt_arg, f"model_fold_{cur}.pt")
            if os.path.isfile(cand):
                ckpt_path = cand
            else:
                print(f"[LOAD][WARN] Expected per-fold checkpoint not found: {cand}. "
                    f"Skipping load for fold {cur} to avoid leakage.")
        elif os.path.isfile(ckpt_arg):
            # If a single file is given, only allow it when it matches this fold
            base = os.path.basename(ckpt_arg)
            expected = f"model_fold_{cur}.pt"
            if base == expected:
                ckpt_path = ckpt_arg
            else:
                print(f"[LOAD][SKIP] Single ckpt '{base}' does not match this fold "
                    f"('{expected}'). Skipping to prevent data leakage.")
        else:
            print(f"[LOAD][WARN] --frozen_ckpt path does not exist: {ckpt_arg}")

    if ckpt_path:
        state = torch.load(ckpt_path, map_location=args.device)
        # strict=False to be robust to minor module name diffs
        missing_unexp = model.load_state_dict(state, strict=False)
        print(f"[LOAD] Loaded fold-specific checkpoint for fold {cur}: {ckpt_path} (strict=False)")
    else:
        if ckpt_arg:
            print(f"[LOAD] No checkpoint loaded for fold {cur} (see warnings above).")

    # Freeze AFTER loading so weights are actually frozen
    if getattr(args, 'freeze_base', False):
        for p in model.parameters():
            p.requires_grad_(False)
        print("[FREEZE] Base model parameters are frozen.")

    use_amp = getattr(args, "amp", True) and args.device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True,
                                    weighted=args.weighted_sample,
                                    mode=args.mode, batch_size=args.batch_size)
    val_loader   = get_split_loader(val_split, mode=args.mode, batch_size=args.batch_size)
    print('Done!')
    sys.stdout.flush()

    # ---- Debias stack: probe rep dim & build if requested ----
    use_debias = bool(getattr(args, 'debias', False)) and hasattr(model, 'repr') and hasattr(model, 'classify_from_repr')
    editor = adv = lam_sched = None

    if use_debias:
        z_sample = None
        for probe in train_loader:
            if probe is None:
                continue
            if len(probe) == 8:
                data_MRI, data_WSI, data_omic, y_disc, event_time, censor, _, _age = probe
            else:
                data_MRI, data_WSI, data_omic, y_disc, event_time, censor, _ = probe
            data_MRI = data_MRI.to(args.device, non_blocking=True)
            data_omic = data_omic.to(args.device, non_blocking=True)
            with torch.no_grad():
                z_sample = model.repr(x_mri=data_MRI, x_omic=data_omic)
            break

        if z_sample is None:
            raise RuntimeError("Could not probe a training batch to build debias stack (no usable batches).")

        rep_dim = int(z_sample.shape[-1])
        editor = ResidualEditor(rep_dim, hidden=args.editor_hidden, resid_scale=args.editor_resid_scale).to(args.device)
        adv    = AgeAdversary(rep_dim, n_bins=args.n_classes, hidden=args.adv_hidden).to(args.device)
        lam_sched = LambdaScheduler(
            lam_start=args.adv_lambda_start, lam_end=args.adv_lambda_end,
            warmup_epochs=args.adv_warmup, total_epochs=args.max_epochs
        )

    # ---- Optimizer AFTER we have modules ----
    print('\nInit optimizer ...', end=' ')
    if use_debias and getattr(args, 'freeze_base', False):
        # Train only debias stack
        params = list(editor.parameters()) + list(adv.parameters())
        if args.opt == 'sgd':
            optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.reg, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.reg)
    else:
        # Train base; if debias on, include its params too
        optimizer = get_optim(model, args)
        if use_debias:
            # optimizer.add_param_group({'params': list(editor.parameters())})
            # optimizer.add_param_group({'params': list(adv.parameters())})
            lr_ed = getattr(args, 'lr_editor', args.lr * 5.0)
            lr_adv = getattr(args, 'lr_adv',   args.lr * 2.0)
            optimizer.add_param_group({'params': list(editor.parameters()), 'lr': lr_ed, 'weight_decay': 0.0})
            optimizer.add_param_group({'params': list(adv.parameters()),    'lr': lr_adv})
            print(f"[OPT] lr(base)={args.lr} lr(editor)={lr_ed} lr(adv)={lr_adv}")
    print('Done!')

    best_val_metric = -float('inf') if args.task == 'risk' else 0.0
    patience_counter = 0
    patience = args.patience

    train_metrics, val_metrics = [], []
    train_losses,  val_losses  = [],  []

    print("\n[Pretrain validation: baseline bias]")
    _ = validate(cur, -1, model, val_loader, loss_fn, args, use_amp=use_amp,
                 use_debias=False, editor=None, adv=None, lam_val=0.0)

    for epoch in tqdm.tqdm(range(args.max_epochs)):
        tr_loss_main, tr_loss_total, tr_metric = train_loop(
            epoch, model, train_loader, optimizer, loss_fn, args, scaler=scaler, use_amp=use_amp,
            use_debias=use_debias, editor=editor, adv=adv, lam_sched=lam_sched
        )
        va_loss_main, va_loss_total, va_metric = validate(
            cur, epoch, model, val_loader, loss_fn, args, use_amp=use_amp,
            use_debias=use_debias, editor=editor, adv=adv, lam_val=(lam_sched.at(epoch) if use_debias else 0.0)
        )

        train_metrics.append(tr_metric)
        val_metrics.append(va_metric)
        train_losses.append(tr_loss_total)
        val_losses.append(va_loss_total)

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
    plots_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

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
def train_loop(epoch, model, loader, optimizer, loss_fn, args, gc=16, scaler=None, use_amp=False,
               use_debias=False, editor=None, adv=None, lam_sched=None):
    model.train()
    loss_main_sum, loss_total_sum = 0.0, 0.0
    n_batches = 0

    surv_scores, surv_censors, surv_times = [], [], []
    cls_correct, cls_total = 0, 0

    for batch_idx, batch in enumerate(loader):
        if batch is None:
            continue
        if len(batch) == 8:
            data_MRI, data_WSI, data_omic, y_disc, event_time, censor, _, age_group = batch
        else:
            data_MRI, data_WSI, data_omic, y_disc, event_time, censor, _ = batch
            age_group = None

        data_MRI   = data_MRI.to(args.device, non_blocking=True)
        data_WSI   = data_WSI.to(args.device, non_blocking=True) if torch.is_tensor(data_WSI) else data_WSI
        data_omic  = data_omic.to(args.device, non_blocking=True)
        y_disc     = y_disc.to(args.device, non_blocking=True)
        event_time = event_time.to(args.device, non_blocking=True)
        censor     = censor.to(args.device, non_blocking=True)
        age_group  = age_group.to(args.device, non_blocking=True) if age_group is not None else None

        with torch.cuda.amp.autocast(enabled=use_amp):
            if use_debias:
                z = model.repr(x_mri=data_MRI, x_omic=data_omic)
                z_prime = editor(z)
                logits  = model.classify_from_repr(z_prime)

                # Task loss
                if isinstance(loss_fn, NLLSurvLoss):
                    loss_task = loss_fn(h=logits.float(), y=y_disc.float(), t=event_time.float(), c=censor.float())
                else:
                    loss_task = loss_fn(logits.float(), y_disc.long())

                # Adversary (conditional on y) via GRL
                # if age_group is not None:
                #     y_onehot = F.one_hot(y_disc.long(), num_classes=args.n_classes).float()
                #     lam = lam_sched.at(epoch) if lam_sched is not None else 1.0
                #     # logits_a = adv(grl(z_prime, lam), y_onehot)
                #     cond = make_combined_onehot(y_disc, censor, n_disc=args.n_classes, n_cens=2)
                #     logits_a = adv(grl(z_prime, lam), cond)

                #     loss_adv = F.cross_entropy(logits_a, age_group.long())
                # else:
                #     loss_adv = 0.0 * logits.sum()
                if age_group is not None:
                    y_onehot = _make_bin_onehot(y_disc, args.n_classes)  # STRICT: only bins
                    lam = lam_sched.at(epoch) if lam_sched is not None else 1.0
                    logits_a = adv(grl(z_prime, lam), y_onehot)
                    loss_adv = F.cross_entropy(logits_a, age_group.long())
                else:
                    loss_adv = 0.0 * logits.sum()
                # Regularizers
                loss_prox = (z_prime - z).pow(2).mean() * getattr(args, 'prox_lambda', 1e-3)
                with torch.no_grad():
                    base_logits = model.classify_from_repr(z)
                loss_cons = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(base_logits, dim=-1),
                                     reduction='batchmean') * getattr(args, 'cons_lambda', 1e-2)

                loss = loss_task + loss_adv + loss_prox + loss_cons
            else:
                logits = model(x_path=data_WSI, x_omic=data_omic, x_mri=data_MRI)
                loss = loss_fn(h=logits.float(), y=y_disc.float(), t=event_time.float(), c=censor.float()) \
                    if isinstance(loss_fn, NLLSurvLoss) else loss_fn(logits.float(), y_disc.long())

        loss_value = float(loss.detach().cpu())
        loss_main_sum += loss_value
        loss_total_sum += loss_value
        n_batches += 1

        if isinstance(loss_fn, NLLSurvLoss):
            with torch.no_grad():
                risk = _binwise_scores_surv(logits)
                surv_scores.append(risk)
                surv_censors.append(censor.cpu().numpy())
                surv_times.append(event_time.cpu().numpy())
        else:
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                cls_correct += (preds == y_disc).sum().item()
                cls_total   += y_disc.numel()

        accum_loss = loss / gc
        if scaler is not None and use_amp:
            scaler.scale(accum_loss).backward()
        else:
            accum_loss.backward()

        if (batch_idx + 1) % gc == 0:
            if scaler is not None and use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            # print("Editor Grad", sum(p.grad.abs().mean().item() for p in editor.parameters() if p.grad is not None))
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
def validate(cur, epoch, model, loader, loss_fn, args, gc=16, use_amp=False,
             use_debias=False, editor=None, adv=None, lam_val=0.0):
    model.eval()
    loss_main_sum, loss_total_sum = 0.0, 0.0
    n_batches = 0

    surv_scores, surv_censors, surv_times = [], [], []
    age_all = []
    logits_all_cls = []
    y_all_cls = []
    cls_correct, cls_total = 0, 0

    for batch_idx, batch in enumerate(loader):
        if batch is None:
            continue
        # print("batch", len(batch))
        if len(batch) == 7:
            data_MRI, data_WSI, data_omic, y_disc, event_time, censor, _ = batch
            age_group = None
        else:
            data_MRI, data_WSI, data_omic, y_disc, event_time, censor, _, age_group = batch
            age_group = age_group.to(args.device, non_blocking=True)
        data_MRI   = data_MRI.to(args.device, non_blocking=True)
        data_WSI   = data_WSI.to(args.device, non_blocking=True) if torch.is_tensor(data_WSI) else data_WSI
        data_omic  = data_omic.to(args.device, non_blocking=True)
        y_disc     = y_disc.to(args.device, non_blocking=True)
        event_time = event_time.to(args.device, non_blocking=True)
        censor     = censor.to(args.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if use_debias and editor is not None:
                z = model.repr(x_mri=data_MRI, x_omic=data_omic)
                z_prime = editor(z)               # apply editor at eval to reflect deployed system
                logits = model.classify_from_repr(z_prime)
                # NOTE: no adversary/GRL during validation; we only evaluate task + bias
                if isinstance(loss_fn, NLLSurvLoss):
                    loss = loss_fn(h=logits.float(), y=y_disc.float(), t=event_time.float(), c=censor.float())
                else:
                    loss = loss_fn(logits.float(), y_disc.long())
            else:
                logits = model(x_path=data_WSI, x_omic=data_omic, x_mri=data_MRI)
                loss = loss_fn(h=logits.float(), y=y_disc.float(), t=event_time.float(), c=censor.float()) \
                    if isinstance(loss_fn, NLLSurvLoss) else loss_fn(logits.float(), y_disc.long())

        loss_value = float(loss)
        loss_main_sum += loss_value
        loss_total_sum += loss_value
        n_batches += 1

        if isinstance(loss_fn, NLLSurvLoss):
            risk = _binwise_scores_surv(logits)
            surv_scores.append(risk)
            surv_censors.append(censor.cpu().numpy())
            surv_times.append(event_time.cpu().numpy())
            if age_group is not None:
                age_all.append(age_group.cpu().numpy())
        else:
            logits_all_cls.append(logits.detach().cpu())
            y_all_cls.append(y_disc.detach().cpu())
            preds = torch.argmax(logits, dim=1)
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
            if len(age_all):
                ages_np = np.concatenate(age_all, axis=0)
                sg = _subgroup_cindex(scores, times, censors, ages_np)
                print(f"[VAL bias] young_ci={sg['young']:.4f} old_ci={sg['old']:.4f} gap(old-young)={sg['gap_old_minus_young']:.4f}")
        except ValueError:
            metric = float('nan')
            print(f'val_surv_loss: {loss_main:.4f}, val_loss: {loss_total:.4f}, val_c-index: nan (no valid samples)')
    else:
        acc = (cls_correct / max(cls_total, 1)) if cls_total else 0.0
        metric = acc
        print(f'val_ce_loss: {loss_main:.4f}, val_loss: {loss_total:.4f}, val_acc: {metric:.4f}')
        if len(age_all) and len(logits_all_cls):
            ages_np = torch.from_numpy(np.concatenate(age_all, axis=0))
            logits_cat = torch.cat(logits_all_cls, dim=0)
            y_cat = torch.cat(y_all_cls, dim=0)
            sg = _subgroup_acc(logits_cat, y_cat, ages_np)
            print(f"[VAL bias] young_acc={sg['young']:.4f} old_acc={sg['old']:.4f} gap(old-young)={sg['gap_old_minus_young']:.4f}")

    return loss_main, loss_total, metric
