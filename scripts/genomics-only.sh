#!/bin/bash

#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --mem=124G
#SBATCH -t 24:00:00

# python main-genomics.py \
#   --mode genomic \
#   --csv data/processed_tabular_data/mut_cna_177_patients.csv \
#   --mri_dir data/2.5D_MRIs \
#   --split_dir data/splits \
#   --results_dir results/genomics_matched_to_2p5d \
#   --match_mri2p5d_cohort

python main-genomics.py \
    --mode radio_2.5D \
    --csv data/processed_tabular_data/mut_cna_177_patients.csv \
    --mri_dir data/2.5D_MRIs \
    --split_dir data/splits \
    --results_dir results/mri2p5d_matched_to_genomics \
    --match_genomics_cohort
