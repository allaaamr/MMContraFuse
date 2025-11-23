#!/bin/bash

#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --mem=124G
#SBATCH -t 24:00:00

#MIU Dataset
# python main.py \
#   --mode genomic \
#   --csv data/MIUGlioma/MIU-Glioma.csv \
#   --mri_dir data/MIUGlioma/2.5D_MRIs \
#   --split_dir data/MIUGlioma/splits \
#   --results_dir results/genomics_matched_to_2p5d \

#TCGA Dataset
python main.py \
  --mode genomic \
    --csv data/TCGA/processed_tabular_data/mut_cna_177_patients.csv \
    --task risk \
    --mri_dir data/TCGA/2.5D_MRIs \
    --split_dir data/TCGA/splits \
    --results_dir results/radiogenomic_M1_TCGA
