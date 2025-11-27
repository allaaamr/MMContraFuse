#!/bin/bash

#SBATCH -n 5
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 24:00:00

python main.py \
    --mode radiomic \
    --csv data/MIUGlioma/MIU-Glioma.csv \
    --task risk \
    --mri_dir data/MIUGlioma/2.5D_MRIs \
    --split_dir data/MIUGlioma/splits \
    --results_dir results/genomic_gtf_risk_miu