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
    --mode radio_2.5D \
    --task risk \
    --create_split \
    --results_dir results/radio_2.5D_risk