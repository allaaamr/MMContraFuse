
1. Download Raw Clinical Data
   LGG data: https://www.cbioportal.org/study?id=lgg_tcga_pan_can_atlas_2018
   GBM data:
   
2. Run Script to Preprocess Genomics & Clinical Data
   python preproces.py \
  --gbm_dir "C:\Users\Amr\Downloads\gbm_tcga_pan_can_atlas_2018\gbm_tcga_pan_can_atlas_2018" \
  --lgg_dir "C:\path\to\lgg_tcga_pan_can_atlas_2018\lgg_tcga_pan_can_atlas_2018" \
  --signatures_csv "C:\Users\Amr\Desktop\Masters\Multi-Modal-Fusion\data\features\signatures.csv" \
  --outdir "C:\Users\Amr\Desktop\Masters\Multi-Modal-Fusion\data\processed" \
  --modalities cna rna \
  --strict

3. Run Model
   python main.py \
   --mode [options: genomics, radio_1D, radio_2.5D, radio_3D, radiogeomics_1D, ..etc.] \
   --task [type , risk ] \
   --create_splits 
