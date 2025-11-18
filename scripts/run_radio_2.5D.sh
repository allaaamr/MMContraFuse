#!/bin/bash

#SBATCH -n 5
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 24:00:00


# python main.py \
#     --mode radio_2.5D \
#     --task subtype \
#     --create_split \
#     --results_dir results/radio_2.5D_subtype 

python main.py \
    --mode radiomic \
    --csv data/processed_tabular_data/mut_cna_177_patients.csv \
    --task risk \
    --mri_dir data/2.5D_MRIs \
    --split_dir data/splits \
    --results_dir results/genomic_gtf_risk