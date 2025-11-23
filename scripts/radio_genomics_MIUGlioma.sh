

# Run Model 2 on MIU Glioma
# python main.py \
#     --mode radiomic \
#     --csv data/MIU-Glioma.csv \
#     --task risk \
#     --mri_dir data/MIU-Glioma \
#     --create_splits \
#     --results_dir results/radiogenomic_M2_MIUGlioma

# Run Model 1 on MIU Glioma
python main.py \
    --mode genomic_radio_2.5D \
    --csv data/MIUGlioma/MIU-Glioma.csv \
    --task risk \
    --mri_dir data/MIUGlioma/2.5D_MRIs \
    --split_dir data/MIUGlioma/splits \
    --results_dir results/radiogenomic_M1_MIUGlioma