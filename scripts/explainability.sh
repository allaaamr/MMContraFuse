python explainability/run_explainer.py \
 --checkpoint results/radiogenomic_MIUGlioma/best_model_radiomic.pt \
 --args results/radiogenomic_MIUGlioma/args_radiomic.pkl \
 --xai_dir results/xai_results/radiogenomic_gtf_risk_miuglioma \
 --data data/MIUGlioma/2.5D_MRIs \
 --val_loader results/radiogenomic_MIUGlioma/val_loader_radiomic.pkl\
  --train_loader results/radiogenomic_MIUGlioma/train_loader_radiomic.pkl \
  --val_split results/radiogenomic_MIUGlioma/val_split_radiomic.pkl \
  --train_split results/radiogenomic_MIUGlioma/train_split_radiomic.pkl \
  --case_id PatientID_0004

  streamv2v/vid2vid/lora_weights