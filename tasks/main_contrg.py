from __future__ import print_function
import numpy as np
import argparse
import os
import sys
from timeit import default_timer as timer
import numpy as np
import pandas as pd

from data.dataset import Generic_MIL_Dataset
from utils.utils import *
from utils.core_utils import train
from utils.train_contrg import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler

from utils.downstream import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main(args):
    """
    
    1. Phase 1: Contrastive pre-training on unlabeled multi-modal data
    2. Phase 2: Fine-tune or linear probe on downstream tasks with labels
    """
    
    
    # Phase 1: Contrastive Pre-training
    print("Phase 1: Contrastive Pre-training")

    ### Get the Train + Val Dataset Loader.
    train_dataset, val_dataset = dataset.return_splits(csv_path='{}/split_{}.csv'.format(args.split_dir, i=0))

    print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
    datasets = (train_dataset, val_dataset)
    sys.stdout.flush()
    ### Specify the input dimension size if using genomic features.

    args.omic_input_dim = train_dataset.genomic_features.shape[1]
    args.radio_input_dim = train_dataset.radiomics_features.shape[1]
    print("Genomic Dimension", args.omic_input_dim)
    sys.stdout.flush()

    model, cindex_latest, val_loader, train_loader = train(datasets,0, args)


 
    # Train ContRG
    contig_model = train_contig(
        train_loader= train_loader,
        genomics_input_dim = args.omic_input_dim,
        radiomics_input_dim =args.radiomics_input_dim ,
        genomics_output_dim=256,
        radiomics_output_dim=64,
        projection_dim=128,
        temperature=0.1,
        learning_rate=1e-3,
        weight_decay=1e-6,
        max_epochs=20
    )
    
    # Phase 2: Downstream Task Evaluation
    print("\nPhase 2: Downstream Task Evaluation")


    
    # # Evaluate on downstream task
    # classifier, best_acc = evaluate_downstream_task(
    #     contig_model=contig_model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,  # Use separate val set in practice
    #     num_classes=3,
    #     task_type='classification',
    #     freeze_encoder=True,  # Linear probing
    #     num_epochs=args.max_epochs
    # )
    
    # print(f"\nBest downstream task accuracy: {best_acc:.2f}%")


parser = argparse.ArgumentParser(description='Configurations for ContRG')
### Checkpoint + Misc. Pathing Parameters

parser.add_argument('--path_dir',   type=str, default='path/to/data_root_dir', help='Data directory to WSI features (extracted via CLAM')
parser.add_argument('--csv',   type=str, default='cna_177_patients.csv', help='directory to clinical and genomics csv file')
parser.add_argument('--mri_dir',   type=str, default='data/2.5D_MRIs', help='directory to MRI data')

parser.add_argument('--seed', 			 type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', 			     type=int, default=5, help='Number of folds (default: 5)')
parser.add_argument('--k_start',		 type=int, default=-1, help='Start fold (Default: -1, last fold)')
parser.add_argument('--k_end',			 type=int, default=-1, help='End fold (Default: -1, first fold)')
parser.add_argument('--results_dir',     type=str, default='./results', help='Results directory (Default: ./results)')
parser.add_argument('--split_dir',       type=str, default='data/splits', help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca_100)')

### Model Parameters.
parser.add_argument('--task',            type=str, choices=['subtype', 'risk'], default='risk', help='Specifies which downstream task to perform.')
parser.add_argument('--mode',            type=str, choices=['genomic', 'path', 'radio_1D' ,'radio_2.5D', 'radio_3D', 'pathomic', 'radiomic1D', 'radiomic2.5D', 'radiomic3D', 'radiopathomics'], default='genomic', help='Specifies which modalities to use.')
parser.add_argument('--fusion',          type=str, choices=['concat', 'bi_attn', 'tri_attn', 'bi_contrast', 'tri_contrast'], default='concat', help='Type of fusion. (Default: concat).')
parser.add_argument('--drop_out',        action='store_true', default=True, help='Enable dropout (p=0.25)')
parser.add_argument('--model_size_wsi',  type=str, default='small', help='Network size of AMIL model')
parser.add_argument('--model_size_omic', type=str, default='small', help='Network size of SNN model')
parser.add_argument('--n_classes', type=int, default=4)




### Optimizer Parameters + Survival Loss Function
parser.add_argument('--opt',             type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size',      type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--gc',              type=int, default=32, help='Gradient Accumulation Step.')
parser.add_argument('--max_epochs',      type=int, default=20, help='Maximum  number of epochs to train (default: 20)')
parser.add_argument('--lr',				 type=float, default=2e-4, help='Learning rate (default: 0.0001)')
parser.add_argument('--bag_loss',        type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv'], default='nll_surv', help='slide-level classification loss function (default: ce)')
parser.add_argument('--loss',        	 type=str, choices=[ 'cox', 'nll'], default='nll')
parser.add_argument('--label_frac',      type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', 			 type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--alpha_surv',      type=float, default=0.0, help='How much to weigh uncensored patients')
parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'], default='None', help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg',      type=float, default=1e-5, help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--weighted_sample', action='store_true', default=True, help='Enable weighted sampling')
parser.add_argument('--early_stopping',  action='store_true', default=False, help='Enable early stopping')
parser.add_argument('--data',            type=str, default='cna_mut_df')
parser.add_argument('--patience',        type=int,  default=8)
parser.add_argument('--create_split',    action='store_true', default=False)

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Depending on the downstream task the label column to predict is 
if args.task == "risk":
    label_col = "survival"
    n_bins = args.n_classes
else:
	label_col = "type"


	
dataset = Generic_MIL_Dataset(csv_path = args.csv,
                                mode = args.mode,
                                path_dir= args.path_dir,
                                mri_dir = args.mri_dir,
                                task = args.task,
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                create_split = args.create_split,    
                                n_splits = 5,
                                patient_strat= False,
                                n_bins=args.n_classes,
                                label_col = label_col) 

if __name__ == "__main__":
	start = timer()
	results = main(args)
	end = timer()
	print("finished!")
	print("end script")
	print('Script Time: %f seconds' % (end - start))