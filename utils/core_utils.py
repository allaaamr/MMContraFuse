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


def train(datasets: tuple, cur: int, args: Namespace):

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split = datasets
    # save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    print('\nInit loss function...', end=' ')
    if args.loss == 'nll':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    # if args.loss == 'cox':
    #     loss_fn = CoxPHSurvLoss()




    print('\nInit Model...', end=' ')
    args.fusion = 'trilinear2' if args.fusion == 'None' else args.fusion

    if args.mode =='genomic':
        model_dict = {'omic_input_dim': args.omic_input_dim, 'model_size_omic': args.model_size_omic, 'n_classes': args.n_classes}
        model = SNN(**model_dict)
    
    # if hasattr(model, "relocate"):
    #     model.relocate()
    # else:
    #     model = model.to(torch.device('cuda'))
    # print('Done!')

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, 
        weighted = args.weighted_sample, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split,  mode=args.mode, batch_size=args.batch_size)
    print('Done!')
    sys.stdout.flush()
    # print('\nSetup EarlyStopping...', end=' ')
    # if args.early_stopping:
    #     early_stopping = EarlyStopping(warmup=0, patience=10, stop_epoch=20, verbose = True)
    # else:
    #     early_stopping = None

    # print('\nSetup Validation C-Index Monitor...', end=' ')
    # monitor_cindex = Monitor_CIndex()
    print('Done!')
    best_val_cindex = -float('inf')  # Initialize to a very low value for c-index
    patience_counter = 0
    train_c_indices = []
    val_c_indices = []
    train_losses = []
    val_losses = []


    patience = args.patience 
    for epoch in range(args.max_epochs):
        train_loss_surv, train_loss, train_c_index = train_loop(epoch, model, train_loader, optimizer, loss_fn, 4)
        val_loss_surv, val_loss, val_c_index = validate(cur, epoch, model, val_loader, loss_fn, 4)

        train_c_indices.append(train_c_index)
        val_c_indices.append(val_c_index)
        train_losses.append(train_loss)
        val_losses.append(val_loss)


    print('Val c-Index: {:.4f}'.format(val_c_index))

    # Plot the results
    epochs = range(1, len(train_c_indices) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_c_indices, label='Train c-index')
    plt.plot(epochs, val_c_indices, label='Validation c-index')
    plt.xlabel('Epochs')
    plt.ylabel('c-index')
    plt.title('Train vs Validation c-index Over Epochs')
    plt.legend()

    output_path = f"cindex_plot_{cur}_transformer_d3.png"  # Specify the path and filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss Over Epochs')
    plt.legend()

    output_path = f"loss_plot_{cur}_transformer_d3.png"  # Specify the path and filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    return model, val_c_index,val_loader,train_loader

def train_loop(epoch, model, loader, optimizer, loss_fn, n_classes, writer=None, lambda_reg=0., gc=16):   
    model.train()
    train_loss_surv, train_loss = 0., 0.
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    

    for batch_idx, batch in enumerate(loader):
        data_MRI, data_WSI, data_omic, y_disc, event_time, censor, slide_ids = batch
        #pdb.set_trace() 
        h = model(x_path=data_WSI, x_omic=data_omic, x_mri =data_MRI) # return hazards, S, Y_hat, A_raw, results_dict

        loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
        loss_value = loss.item()

        loss_reg = 0


        if isinstance(loss_fn, NLLSurvLoss):
            hazards = torch.sigmoid(h)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        else:
            risk = h.detach().cpu().numpy().squeeze()

        all_risk_scores.append(risk)
        all_censorships.append(censor.detach().cpu().numpy())
        all_event_times.append(event_time.detach().cpu().numpy())
        #pdb.set_trace()
        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if y_disc.shape[0] == 1 and (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss_value + loss_reg, y_disc.detach().cpu().item(), float(event_time.detach().cpu().item()), float(risk), data_WSI.size(0)))
        elif y_disc.shape[0] != 1 and (batch_idx + 1) % 5 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss_value + loss_reg, y_disc.detach().cpu()[0], float(event_time.detach().cpu()[0]), float(risk[0]), data_WSI.size(0)))
        sys.stdout.flush()
        # backward pass
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    #pdb.set_trace()

    all_risk_scores = np.concatenate(all_risk_scores)
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)

    # c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships) 
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))
    sys.stdout.flush()

    return train_loss_surv, train_loss, c_index

def validate(cur, epoch, model, loader, loss_fn , n_classes, early_stopping=None, monitor_cindex=None, writer=None, lambda_reg=0., results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_MRI, data_WSI, data_omic, y_disc, event_time, censor, slide_ids) in enumerate(loader):
        with torch.no_grad():
        
            h = model(x_path=data_WSI, x_omic=data_omic, x_mri =data_MRI) # return hazards, S, Y_hat, A_raw, results_dict

        loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
        loss_value = loss.item()

 
        loss_reg = 0

        if isinstance(loss_fn, NLLSurvLoss):
            hazards = torch.sigmoid(h)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        else:
            risk = h.detach().cpu().numpy()

        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = censor.detach().cpu().numpy()
        all_event_times[batch_idx] = event_time.detach().cpu().numpy()

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]


    print('val_loss_surv: {:.4f}, val_loss: {:.4f}, val_c_index: {:.4f}'.format(val_loss_surv, val_loss, c_index))
    sys.stdout.flush()
    return val_loss_surv, val_loss, c_index