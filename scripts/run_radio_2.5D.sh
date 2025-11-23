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
    --mode genomic_radio_2.5D \
    --csv data/TCGA/processed_tabular_data/mut_cna_177_patients.csv \
    --task risk \
    --mri_dir data/TCGA/2.5D_MRIs \
    --split_dir data/TCGA/splits
    --results_dir results/radiogenomic_M1_TCGA

    #   --csv data/processed_tabular_data/mut_cna_177_patients.csv \
#   --mri_dir data/2.5D_MRIs \
#   --split_dir data/splits \
#   --results_dir results/genomics_matched_to_2p5d \
#   --match_mri2p5d_cohort
