from __future__ import print_function
import numpy as np
import torch_geometric
import argparse
import pdb
import os
import math
import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd

### Internal Imports
from dataset import Generic_MIL_Survival_Dataset
from utils.core_utils import train
from utils.utils import *
from XAI.explainability import Explainer
### PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main(args):

	if args.xai:
		print(args.mode)
		val_loader = unpickle(os.path.join(args.xai_dir,f"val_loader_{args.model_type}_transf.pkl"))
		val_split = unpickle( os.path.join(args.xai_dir,f"val_split_{args.model_type}_transf.pkl"))		
		train_loader = unpickle(os.path.join(args.xai_dir,f"train_loader_{args.model_type}_transf.pkl"))
		train_split = unpickle( os.path.join(args.xai_dir,f"train_split_{args.model_type}_transf.pkl"))
		explainer = Explainer(args.xai_dir, args.model_type, val_loader, val_split, train_loader,train_split )
		explainer.run()

	else:
		models = []
		c_indices  = []
		val_split = []
		val_loaders = []
		train_split = []
		train_loaders = []
		test_split_indices =[]

		for i in range(0, 5):
			### Get the Train + Val Dataset Loader.
			train_dataset, val_dataset = dataset.return_splits(
					csv_path='{}/split_{}.csv'.format(split_dir, i))

			print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
			datasets = (train_dataset, val_dataset)
			sys.stdout.flush()
			### Specify the input dimension size if using genomic features.

			args.omic_input_dim = train_dataset.genomic_features.shape[1]
			print("Genomic Dimension", args.omic_input_dim)
			sys.stdout.flush()

			### Run Train-Val on Survival Task.
			model, cindex_latest, val_loader, train_loader = train(datasets,i, args)
			models.append(model)
			c_indices.append(cindex_latest)
			test_split_indices.append(val_dataset.case_ids)
			val_split.append(val_dataset)
			val_loaders.append(val_loader)
			train_split.append(train_dataset)
			train_loaders.append(train_loader)

		best_fold = np.argmax(c_indices)
		best_model = models[best_fold]

		torch.save(best_model.state_dict(), os.path.join(args.xai_dir, f"best_model_{args.model_type}_d3.pt"))
		pickle_obj(val_loaders[best_fold], os.path.join(args.xai_dir,f"val_loader_{args.model_type}_d3.pkl"))
		pickle_obj(val_split[best_fold], os.path.join(args.xai_dir,f"val_split_{args.model_type}_d3.pkl"))
		pickle_obj(train_loaders[best_fold], os.path.join(args.xai_dir,f"train_loader_{args.model_type}_d3.pkl"))
		pickle_obj(train_split[best_fold], os.path.join(args.xai_dir,f"train_split_{args.model_type}_d3.pkl"))
		pickle_obj(args, os.path.join(args.xai_dir,f"args_{args.model_type}_d3.pkl"))

		sum =0
		for i in c_indices:
			print(i)
			sum+=i
		print('Average ', sum/len(c_indices))

	end = timer()
	print('Time: %f seconds' % ( end - start))


### Training settings
parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')
### Checkpoint + Misc. Pathing Parameters
parser.add_argument('--env', type=str, default='server')
parser.add_argument('--xai', action='store_true', help="Enable XAI (e.g., SHAP, IG) analysis")

parser.add_argument('--data_root_dir',   type=str, default='path/to/data_root_dir', help='Data directory to WSI features (extracted via CLAM')
parser.add_argument('--seed', 			 type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', 			     type=int, default=5, help='Number of folds (default: 5)')
parser.add_argument('--k_start',		 type=int, default=-1, help='Start fold (Default: -1, last fold)')
parser.add_argument('--k_end',			 type=int, default=-1, help='End fold (Default: -1, first fold)')
parser.add_argument('--results_dir',     type=str, default='./results_new', help='Results directory (Default: ./results)')
# parser.add_argument('--which_splits',    type=str, default='5foldcv', help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
parser.add_argument('--split_dir',       type=str, default='/mnt/lustre-grete/usr/u12402/Multi-Modal-Fusion/data/splits_232', help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca_100)')
parser.add_argument('--log_data',        action='store_true', default=True, help='Log data using tensorboard')
parser.add_argument('--overwrite',     	 action='store_true', default=False, help='Whether or not to overwrite experiments (if already ran)')

### Model Parameters.
parser.add_argument('--model_type',      type=str, default='mcat', help='Type of model (Default: mcat)')
parser.add_argument('--mode',            type=str, choices=['omic', 'path', 'pathomic', 'radio', 'radio_3d_flair','radio_3d_t1c', 'radiomic', 'radiopath', 'radiopathomics' , "deepr", '3mmf'], default='coattn', help='Specifies which modalities to use / collate function in dataloader.')
parser.add_argument('--fusion',          type=str, choices=['None', 'concat', 'bilinear', 'trilinear', 'trilinear2'], default='None', help='Type of fusion. (Default: concat).')
parser.add_argument('--apply_sig',		 action='store_true', default=False, help='Use genomic features as signature embeddings.')
parser.add_argument('--apply_sigfeats',  action='store_true', default=False, help='Use genomic features as tabular features.')
parser.add_argument('--drop_out',        action='store_true', default=True, help='Enable dropout (p=0.25)')
parser.add_argument('--model_size_wsi',  type=str, default='small', help='Network size of AMIL model')
parser.add_argument('--model_size_omic', type=str, default='small', help='Network size of SNN model')

parser.add_argument('--n_classes', type=int, default=4)


### PORPOISE
parser.add_argument('--apply_mutsig', action='store_true', default=False)
parser.add_argument('--gate_path', action='store_true', default=False)
parser.add_argument('--gate_omic', action='store_true', default=False)
parser.add_argument('--gate_radio', action='store_true', default=False)
parser.add_argument('--scale_dim1', type=int, default=8)
parser.add_argument('--scale_dim2', type=int, default=8)
parser.add_argument('--skip', action='store_true', default=False)
parser.add_argument('--dropinput', type=float, default=0.0)
parser.add_argument('--path_input_dim', type=int, default=1024)
parser.add_argument('--use_mlp', action='store_true', default=False)


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

parser.add_argument('--xai_dir', type=str, default='./XAI/models')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
	import random
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if device.type == 'cuda':
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024

print(f"args.env: {args.env}")

print('\nLoad Dataset')
if args.env =="local":
	csv_path = rf'C:\Users\Amr\Desktop\Masters\Multi-Modal-Fusion\data\cna_mut_177df.csv'
	split_dir = fr'C:\Users\Amr\Desktop\Masters\Multi-Modal-Fusion\data\splits_177'
	data_dir = '/mnt/lustre-grete/usr/u12402/Features/path_patches/patches'
	mri_data_dir =rf'C:\Users\Amr\Desktop\Masters\Multi-Modal-Fusion\data\braTs_BBx'
elif args.env =="local-mac":
#	csv_path = f'/Users/alaaabdelazeem/Desktop/Projects/AI/Multi-Modal-Fusion/data/raw/{args.data}.csv'
	csv_path = '/Users/alaaabdelazeem/Desktop/Masters/Multi-Modal-Fusion/data/cna_mut_177df.csv'
	split_dir = '/Users/alaaabdelazeem/Desktop/Masters/Multi-Modal-Fusion/data/splits_177'
	data_dir = '/mnt/lustre-grete/usr/u12402/Features/path_patches/patches'
	mri_data_dir ='/Users/alaaabdelazeem/Desktop/Masters/Multi-Modal-Fusion/data/braTs_BBx'
else :
	csv_path = f'/mnt/lustre-grete/usr/u12402/Multi-Modal-Fusion/data/cna_mut_177df.csv'
	split_dir = f'/mnt/lustre-grete/usr/u12402/Multi-Modal-Fusion/data/splits_177'
	data_dir = '/mnt/lustre-grete/usr/u12402/CLAM/FEATURES_DIRECTORY'
	mri_data_dir ='/mnt/lustre-grete/usr/u12402/Multi-Modal-Fusion/data/braTs_BBx'

dataset = Generic_MIL_Survival_Dataset(csv_path = csv_path,
										mode = args.mode,
										apply_sig = args.apply_sig,
										data_dir= data_dir,
										mri_data_dir = mri_data_dir,
										shuffle = False, 
										seed = args.seed, 
										print_info = True,
										patient_strat= False,
										n_bins=4,
										label_col = 'survival',
										ignore=[]) 

if __name__ == "__main__":
	start = timer()
	results = main(args)
	end = timer()
	print("finished!")
	print("end script")
	print('Script Time: %f seconds' % (end - start))

# python extract_features_fp.py --data_h5_dir /mnt/lustre-grete/usr/u12402/Features/path_patches/patches --data_slide_dir /mnt/lustre-grete/usr/u12402/pathology_images --csv_path /mnt/lustre-grete/usr/u12402/Multi-Modal-Fusion/data/df.csv --feat_dir FEATURES_DIRECTORY  --batch_size 512 --slide_ext .svs
