# MMContraFuse: Multi-Modal Contrastive Learning and Fusion Framework

## 📋 Overview 

A deep learning framework for multi-modal medical data fusion, combining genomic signatures with radiological imaging (1D, 2.5D, and 3D MRI) using contrastive learning and transformer-based architectures for downstream tasks such as tumor subtype classification and survival/risk prediction.
The framework supports flexible configurations:
-🧬 Genomic encoder (tabular omics)
-🧠 Radiomics encoders for 1D, 2D, and 3D MRI data
-🔗 Transformer-based fusion combining multiple modalities
-⚖️ Downstream evaluation for clinical prediction tasks


## 🏗️ Project Structure

```
MMCONTRAFUSE/
├── data/
├── MIUGlioma                          # MIU Glioma Data
│   │   ├── 2.5D_MRIs
│   │   ├── MIU-Glioma.csv
│   │   ├── MU-Glioma-Raw
│   │   └── splits
│   ├── 1D_MRI/                        # TCGA 1D radiomics data
│   ├── 2.5D_MRIs/                     # TCGA 2.5D MRI slices per patient
│   ├── 3D_Segmented_MRIs/             # TCGA 3D volumetric MRI data 
│   ├── processed_tabular_data/        # TCGA Processed clinical/genomic data
│   ├── splits/                        # TCGA Train/validation/test splits
│   ├── dataset.py                     # TCGA Custom dataset for multi-modal data loading
│   ├── preprocess.py                  # TCGA Genomic data preprocessing and filtering
│   └── signatures.csv                 # TCGA Genomic signatures file
├── models/
|   ├── debiasing_bins.py              # Conditional Debiasing Modules
|   ├── debiasing.py                   # Unconditional Debiasing Modules (no risk bin conditioning)
│   ├── Encoder/
│   │   ├── genomic.py                 # Genomic SNN model
│   │   ├── radiomics_1D.py            # 1D radiomics encoder
│   │   └── utils.py                   # Encoder utilities
│   └── Fusion/
│       ├── Concat.py                  # Simple concatenation fusion
│       ├── ContRG.py                  # Contrastive Radiology-Genomics fusion
│       ├── DownStream.py              # Downstream task-specific fusion
│       ├── MMEncoder.py               # Multi-modal encoder architecture
│       └── Transformer.py             # Transformer-based fusion module
├── utils/
│   ├── core_utils.py                  # Core training and evaluation functions
│   ├── downstream.py                  # Downstream task evaluation
│   ├── loss.py                        # Loss functions
│   ├── train_contrg.py                # ContIG-specific training utilities
│   └── utils.py                       # General utilities (data loading, splits, etc.)
├── main.py                            # Main entry point
└── README.md                         
```

## 🚀 Quick Start

### 1. Clone Repository and Dependencies

```bash
# Clone main repository
git clone https://github.com/allaaamr/MMContraFuse.git
cd MMContraFuse
git checkout -m adv


```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```


##  Training and Evaluation

### Training and Evaluation of Stage 1

```bash
   sh scripts/run_radio_2.5D.sh # train TCGA stage 1
   sh scripts/run_radio_2.5D_miu.sh # train MIU Glioma stage 1
```

### Training and Evaluation of Stage 2

```bash
   sbatch scripts/run_adv_train_eval_radiomics_sweep.sh # train TCGA stage 2 with hyperparameter search
   sbatch scripts/run_adv_train_eval_radiomics_sweep_miu.sh # train MIU Glioma stage 2 with hyperparameter search
```

## Compiling results for Stage 2

After running the ablations place the log files for each dataset in seperate directories i.e. hyperparam_search/ and hyperparam_search_miu/

and compile the results with

```bash
   python parse_logs.py --logs_dir hyperparam_search/ --out_csv sweep_summary.csv
   python parse_logs.py --logs_dir hyperparam_search_miu/ --out_csv sweep_summary_miu.csv
```

## Explainability
Our explainability code (for SHAP and IG plots) is found under the following folder:
```bash
   explainability/
```
