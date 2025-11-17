#!/bin/bash

#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --mem=124G
#SBATCH -t 24:00:00


RESULTS_ROOT="results/fusion_greedy_${TASK}"
SEED=1

# Common MRI trunk defaults (match your current best settings)
LAYER_NUM=32
P_SLICE_DROP=0.15
SD_PROB=0.10
ATTN_DROPOUT=0.10
P_SPATIAL_DROP=0.05
NORM="gn"

# Common fusion head/defaults
DIM_FUSE=256
DMODEL=256
NHEAD=4
HEAD_HIDDEN=256
HEAD_DROPOUT=0.30

python main.py \
  --mode genomic_radio_2.5D --task risk --n_classes 4 \
  --frozen_ckpt ./results/fusion_greedy_risk/baseline__fusion=concat__mri=gap__snn=full/ \
  --freeze_base \
  --debias \
  --adv_lambda_start 0.0 --adv_lambda_end 1.0 --adv_warmup 3 \
  --prox_lambda 1e-3 --cons_lambda 1e-2 \
  --results_dir ./results_grl_frozen \
  --fusion concat --fuse_point_mri gap \
  --split_dir data/splits \
  --seed ${SEED} \
  \
  --layer_num ${LAYER_NUM} \
  --p_slice_drop ${P_SLICE_DROP} \
  --sd_prob ${SD_PROB} \
  --attn_dropout ${ATTN_DROPOUT} \
  --p_spatial_drop ${P_SPATIAL_DROP} \
  --norm ${NORM} \
  \
  --dim_fuse ${DIM_FUSE} \
  --d_model ${DMODEL} \
  --nhead ${NHEAD} \
  --head_hidden ${HEAD_HIDDEN} \
  --head_dropout ${HEAD_DROPOUT} \
  \
  --csv data/processed_tabular_data/mut_cna_177_patients.csv