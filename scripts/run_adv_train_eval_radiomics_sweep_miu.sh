#!/bin/bash
#
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --mem=124G
#SBATCH -t 24:00:00
#SBATCH -J debias_sweep
#SBATCH --array=0-3

set -euo pipefail
mkdir -p logs

# ---- hyperparam grid (max 4) ----
ADV_END=( 1.0 1.0 0.5 0.5 )
ADV_WARMUP=( 3 1 3 1 )
PROX=( 1e-3 1e-3 5e-4 5e-4 )
CONS=( 1e-2 5e-3 1e-2 5e-3 )
TAG=( e1.0_w3_p1e-3_c1e-2 e1.0_w1_p1e-3_c5e-3 e0.5_w3_p5e-4_c1e-2 e0.5_w1_p5e-4_c5e-3 )

IDX="${SLURM_ARRAY_TASK_ID}"

echo "Running array task ${IDX}"
echo "Params: adv_end=${ADV_END[$IDX]} warmup=${ADV_WARMUP[$IDX]} prox=${PROX[$IDX]} cons=${CONS[$IDX]} tag=${TAG[$IDX]}"

SEED=1

python main.py \
  --mode radiomic --task risk --n_classes 4 \
  --frozen_ckpt ./results/genomic_gtf_risk_miu/ \
  --freeze_base \
  --debias \
  --mri_dir data/MIUGlioma/2.5D_MRIs \
  --adv_lambda_start 0.0 --adv_lambda_end "${ADV_END[$IDX]}" --adv_warmup "${ADV_WARMUP[$IDX]}" \
  --prox_lambda "${PROX[$IDX]}" --cons_lambda "${CONS[$IDX]}" \
  --results_dir "./results_grl_frozen_gtf_miu/sweep_${TAG[$IDX]}" \
  --split_dir data/MIUGlioma/splits \
  --csv data/MIUGlioma/MIU-Glioma.csv
