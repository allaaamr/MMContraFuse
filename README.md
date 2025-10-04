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
│   ├── 1D_MRI/                        #  1D radiomics data
│   ├── 2.5D_MRIs/                     # 2.5D MRI slices per patient
│   ├── 3D_Segmented_MRIs/             # 3D volumetric MRI data 
│   ├── processed_tabular_data/        # Processed clinical/genomic data
│   ├── splits/                        # Train/validation/test splits
│   ├── dataset.py                     # Custom dataset for multi-modal data loading
│   ├── preprocess.py                  # Genomic data preprocessing and filtering
│   └── signatures.csv                 # Genomic signatures file
├── models/
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


```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```


## 📊 Dataset Preparation

### 🧬 Data Downloading

 Download Raw Clinical & Genomic Data
LGG dataset: TCGA Pan-Cancer Atlas (LGG)
GBM dataset: TCGA Pan-Cancer Atlas (GBM)


### Preprocess Data

Run the preprocessing script to prepare the dataset for training:

```bash
python preprocess.py \
  --gbm_dir "path\to\gbm\dir\installed\earlier \
  --lgg_dir "path\to\lgg\dir\installed\earlier \
  --signatures_csv "data\features\signatures.csv" \
  --outdir "data\processed" \
  --modalities cna rna \
  --strict
```

**This script :**
- ✅ Loads and merges GBM and LGG clinical/genomic data.
- ✅ Filters significant genes using the provided signatures.csv.
- ✅ Harmonizes modalities (CNA, RNA) and outputs processed datasets to the specified directory.


##  Training and Evaluation

### Uni-Modal Training

```bash
   python main.py \
   --mode [options: genomics, radio_1D, radio_2.5D, radio_3D, radiogeomics_1D, ..etc.] \
   --task [type , risk ] \
   --create_splits
```

Train genomic model only

```bash
python main.py --mode genomic --task subtype --csv cna_177_patients.csv
```

 Train radiological model (1D features)

```bash
python main.py --mode radio_1D --task subtype 
```

### Multi-Modal Fusion 
## Contrastive Fusion Architecture Explanation
The contrastive fusion approach implements the ContIG (Contrastive learning for Imaging and Genetics) methodology, which learns joint representations between genomic and radiological data through self-supervised contrastive learning. Here's how each component works:

# 1  #ConrRG: Core Contrastive Model for Radiology & Genomics
path:
```bash
models/Fusion/ContRG.py
```
This is the heart of the contrastive learning system. It implements:

-Dual Encoders: Separate encoders for genomics (SNN) and radiomics (Radiomics1DNet) that extract modality-specific features
-Projection Heads: Two-layer MLPs that map features from each encoder to a shared 128-dimensional embedding space where both modalities can be compared
-Contrastive Loss (InfoNCE): Treats paired genomic-radiomic data from the same patient as positive pairs, and all other combinations in the batch as negative pairs. ---The loss pulls positive pairs together while pushing negative pairs apart
-Bidirectional Learning: Computes loss in both directions (genomics→radiomics and radiomics→genomics) for symmetric learning

# 2 Downstream Task Adapter
path:
```bash
 utils/downstream.py
```
Handles the complete downstream evaluation workflow:

-evaluate_downstream_task(): Main function that takes a pre-trained ContRG model and evaluates it on classification or survival tasks
-Training Loop: Uses standard supervised training with task-specific losses (CrossEntropy for classification, NLLSurvLoss for survival)
-Comparison Modes: Evaluates both linear probing (frozen encoders) and fine-tuning to assess representation quality
-Visualization: Creates plots comparing training/validation metrics and losses

# 3 Contrastive Pre-training
path:
```bash
utils/train_contrg.py
```
Manages the self-supervised pre-training phase

# 4 Unified Training Interface
path:
```bash
main.py
```
Orchestrates the entire pipeline with two distinct modes:
***Standard Mode:*** Traditional supervised learning

Direct end-to-end training on labeled data
No contrastive pre-training

***Contrastive Mode (2-phase approach): ***

Phase 1 - Pre-training:

Trains ContRG model using contrastive loss
No labels needed, just paired genomic-radiomic data
Can either train from scratch or load pre-trained checkpoint


Phase 2 - Downstream:

Takes pre-trained ContRG model
Evaluates on supervised task (classification/survival)
Compares linear probing vs fine-tuning performance


