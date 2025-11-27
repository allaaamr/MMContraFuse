#!/bin/bash
#SBATCH -J genoMRI_greedy
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --mem=124G
#SBATCH -t 24:00:00

set -euo pipefail

# --------------------------
# User-editable defaults
# --------------------------
PY=python
MAIN=main.py

# Pick the mode your code uses for the Geno+MRI fusion model.
# If you wired FusionFactory under a different mode, change this.
FUSION_MODE="genomic_radio_2.5D"   # or "genoradio" if that's what you used

TASK="risk"                  # keep consistent with your survival/risk setup
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

# If your script needs these (based on your repo’s defaults)
CREATE_SPLIT="--create_split"
# You may also want to pass: --csv, --path_dir, --mri_dir, --split_dir, etc.

# --------------------------
# Greedy ablation plan
# Baseline + (one-change-at-a-time)
# --------------------------
# Baseline (id=0): concat fusion, MRI at gap, full SNN depth
# Fusion ablations: xattn, gated
# MRI fuse point ablations: stem, block1
# SNN depth ablation: fuse_k_omic=1  (assuming model_size_omic='small' → 2 blocks total)

mkdir -p "${RESULTS_ROOT}"

EXP_NAME="abl_fusion_xattn__mri=gap__snn=full"
EXTRA_ARGS=( --fusion bi_attn --fuse_point_mri gap )

OUT_DIR="${RESULTS_ROOT}/${EXP_NAME}"
mkdir -p "${OUT_DIR}"

echo "==> Running ${EXP_NAME}"
echo "==> Results -> ${OUT_DIR}"

${PY} "${MAIN}" \
  --mode "${FUSION_MODE}" \
  --task "${TASK}" \
  ${CREATE_SPLIT} \
  --seed ${SEED} \
  --results_dir "${OUT_DIR}" \
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
  \
  "${EXTRA_ARGS[@]}"
