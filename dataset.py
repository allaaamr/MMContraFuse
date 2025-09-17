from __future__ import print_function, division
import math
import os
import pdb
import pickle
import re
import random
import h5py
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
import nibabel as nib
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class Generic_Dataset(Dataset):
    def __init__(self,
        csv_path = 'genoclinical.csv',
        mode = 'omic',
        shuffle = False, 
        seed = 7, 
        print_info = True,
        label_col: str = "survival",  # <--  "type" or "survival"
        task: str = "subtype",        # <--  "subtype" | "risk_classification" 
        n_bins: int = 4,              # <-- used if task == "survival_binned"
        patient_strat=False,
        eps=1e-6):
        r"""
        Generic_Dataset 

        This dataset is a generic one, 
        It loads the CSV and does dataset-wide preprocessing:

        sets the dictionary and label based on the classification type
        if the classification type is Risk Group Prediction it bins the patients into 4 risk groups based
        on survival duration and assigns each patient a label. 

        If the classification task is type prediction it just defines the type column as the label 
        and basic pre

        Args:
            csv_path (str): Path to the dataset CSV file.
            mode (str): Which modality branch to use ('omic', 'radio', etc.).
            shuffle (bool): Whether to shuffle rows after loading.
            seed (int): Random seed for reproducibility.
            print_info (bool): Print summary info after loading.
            label_col (str): Which column contains labels ("type" or "survival").
            task (str): Which prediction task to run ("type_classification" or "risk_classification").
            n_bins (int): Number of bins if discretizing survival times into risk categories.
            patient_strat (bool): Whether to stratify at patient level.
            eps (float): Small constant for bin edges in survival binning.
        """
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None
        self.mri_data_dir = None
        self.task=task


        # ---- load CSV ----
        slide_data = pd.read_csv(csv_path, low_memory=False)
        label_col = 'survival'

        # optional shuffle
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)
            
        # ----------------------
        # CASE 1: Type classification (categorical label e.g. tumor type) (2 Type Classification)
        # ----------------------
        if self.task == "type_classification":
            
            # build dictionary: patient_id -> all slide_ids (because a single patient can have multiple slides)
            patient_dict = {}
            slide_data = slide_data.set_index('case_id')
            for patient in patients_df['case_id']:
                slide_ids = slide_data.loc[patient, 'slide_id']
                if isinstance(slide_ids, str):
                    slide_ids = np.array(slide_ids).reshape(-1)
                else:
                    slide_ids = slide_ids.values
                patient_dict.update({patient:slide_ids})

            self.patient_dict = patient_dict
        
            slide_data = patients_df
            slide_data.reset_index(drop=True, inplace=True)
            slide_data = slide_data.assign(slide_id=slide_data['case_id'])


            # assign final integer labels to each row 
            slide_data.label = slide_data.type

            self.num_classes=2 # LGG / GBM (types of brain cancer)

            # patient-level view (case_id + label)
            patients_df = slide_data.drop_duplicates(['case_id'])
            self.patient_data = {'case_id':patients_df['case_id'].values, 'label':patients_df['label'].values}
    
            # reorder columns for consistency
            new_cols = list(slide_data.columns[-1:]) + list(slide_data.columns[:-1])  ### PORPOISE
            slide_data = slide_data[new_cols]
            self.slide_data = slide_data

                        # metadata columns (first 12 cols, usually non-feature data)
            metadata = [
                'disc_label', 'Unnamed: 0', 'case_id', 'label', 'slide_id',
                'type', 'age', 'gender', 'survival', 'censorship', 'Unnamed: 0.1'
            ]

        # ----------------------
        # CASE 2: Risk classification (bin survival times into 4 discrete risk groups) (4 Group Classification)
        # ----------------------

        if self.task =="risk_classification":    
            # one patient = one unique row
            patients_df = slide_data.drop_duplicates(['case_id']).copy()
             # only use uncensored patients to define survival quantiles (same methodology as in literature)
             # an uncensored patient is a patient who died thus we know the true survival duration of, we bin the categories based on their info
            uncensored_df = patients_df[patients_df['censorship'] < 1]

            # get quantile-based bins (just transforming survival from continuous to 4 bins : 4 groups )
            disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
             # expand bin edges to cover full survival range
            q_bins[-1] = slide_data[label_col].max() + eps
            q_bins[0] = slide_data[label_col].min() - eps
            
            # cut all patients into those bins (assign each patient to a risk group and add a column called label that contains the risk label of each patient)
            disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
            patients_df.insert(2, 'label', disc_labels.values.astype(int))

            # build dictionary: patient_id -> all slide_ids (because a single patient can have multiple slides)
            patient_dict = {}
            slide_data = slide_data.set_index('case_id')
            for patient in patients_df['case_id']:
                slide_ids = slide_data.loc[patient, 'slide_id']
                if isinstance(slide_ids, str):
                    slide_ids = np.array(slide_ids).reshape(-1)
                else:
                    slide_ids = slide_ids.values
                patient_dict.update({patient:slide_ids})

            self.patient_dict = patient_dict
        
            slide_data = patients_df
            slide_data.reset_index(drop=True, inplace=True)
            slide_data = slide_data.assign(slide_id=slide_data['case_id'])

            # build label_dict = (bin, censorship) → class_id
            label_dict = {}
            key_count = 0
            for i in range(len(q_bins)-1):
                for c in [0, 1]:
                    print('{} : {}'.format((i, c), key_count))
                    label_dict.update({(i, c):key_count})
                    key_count+=1

            self.label_dict = label_dict

            # assign final integer labels to each row 
            for i in slide_data.index:
                key = slide_data.loc[i, 'label']
                slide_data.at[i, 'disc_label'] = key
                censorship = slide_data.loc[i, 'censorship']
                key = (key, int(censorship))
                slide_data.at[i, 'label'] = label_dict[key]

            self.bins = q_bins
            self.num_classes=len(self.label_dict)

            # patient-level view (case_id + label)
            patients_df = slide_data.drop_duplicates(['case_id'])
            self.patient_data = {'case_id':patients_df['case_id'].values, 'label':patients_df['label'].values}
    
            # reorder columns for consistency
            new_cols = list(slide_data.columns[-1:]) + list(slide_data.columns[:-1])  ### PORPOISE
            slide_data = slide_data[new_cols]
            self.slide_data = slide_data

                        # metadata columns (first 12 cols, usually non-feature data)
            metadata = [
                'disc_label', 'Unnamed: 0', 'case_id', 'label', 'slide_id',
                'type', 'age', 'gender', 'survival', 'censorship', 'Unnamed: 0.1'
            ]

         # ---- store final dataframes ----
        self.metadata = slide_data.columns[:12]
        self.genomic_features = self.slide_data.drop(self.metadata, axis=1)
        self.mode = mode
        self.cls_ids_prep()
        self.patient_data_prep()

    def cls_ids_prep(self):
        r"""

        """
        self.patient_cls_ids = [[] for i in range(self.num_classes)]        
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def patient_data_prep(self):
        r"""
        processes patient-level data by creating a dictionary that maps unique patient IDs (case_id) to their respective class labels.
        """
        patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
        patient_labels = []
        
        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]] # get patient label
            patient_labels.append(label)
        
        self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

    def __len__(self):
        return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

    def return_splits(self, csv_path):
        """
        Returns:
            train_df (pd.DataFrame): Training data split.
            val_df (pd.DataFrame): Validation data split.
        """
        all_splits = pd.read_csv(csv_path)
        train_split = all_splits['train']
        mask = self.slide_data['slide_id'].isin(train_split.tolist())
        df_train_slice = self.slide_data[mask].reset_index(drop=True)
        print('df_train_slice ' ,df_train_slice.shape)
        val_split = all_splits['val']
        mask = self.slide_data['slide_id'].isin(val_split.tolist())
        df_val_slice = self.slide_data[mask].reset_index(drop=True)
        print('df_val_slice ' ,df_val_slice.shape)

        train = Generic_Split(df_train_slice, metadata=self.metadata, mode=self.mode, mri_data_dir = self.mri_data_dir, data_dir=self.data_dir, label_col='survival', patient_dict=self.patient_dict, num_classes=self.num_classes)
        val = Generic_Split(df_val_slice, metadata=self.metadata, mode=self.mode, mri_data_dir = self.mri_data_dir,  data_dir=self.data_dir, label_col='survival', patient_dict=self.patient_dict, num_classes=self.num_classes)

        print("****** Normalizing Data ******")
        scalers = train.get_scaler()
        train.apply_scaler(scalers=scalers)
        val.apply_scaler(scalers=scalers)
        print(self.genomic_features.shape)
        return train, val

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    def __getitem__(self, idx):
        return None

class Generic_MIL_Dataset(Generic_Dataset):
    def __init__(self, data_dir, mri_data_dir,mode: str='omic', **kwargs):
        super(Generic_MIL_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.mri_data_dir = mri_data_dir
        self.mode = mode
        self.use_h5 = False
        self.genomic_features = self.slide_data.drop(self.metadata, axis=1)
        print('Mode is ', self.mode)
        print(self.genomic_features.shape)

        r"""
        Inherits from the base.
        Adds I/O & getitem for all modes (radio/path/omic/combos), 
        plus MRI/WSI loaders and normalization hooks.
        Intended to be the “real” dataset that a DataLoader can iterate over.
        """

    def normalize_mri(self, image):
        print(image)
        mean = image.mean()
        std = image.std()

        if std == 0:
            normalized_image = (image - mean)
        else:
            normalized_image = (image - mean) / std
        print('noemalized img ', normalized_image)
        return normalized_image

    def load_mri_3D(self, case_id, modality):
        """
        Load a 3D MRI scan for a given patient and modality.
        Args:
            case_id (string): ID of the patient.
           modality (string): MRI modality (e.g., T1, T1c, T2, Flair, or mask).
        Returns:
            3D MRI scan as a normalized tensor.
        """
        path = os.path.join(self.mri_data_dir, case_id, f"{modality}.nii.gz")
        img = nib.load(path).get_fdata()
        img_tensor = torch.tensor(img, dtype=torch.float32)
        # print('image tensor ' ,img_tensor)
        # normalized_img = self.normalize_mri(img_tensor)

        return img_tensor

    def load_mri_2_5D(self, case_id):
        """
        Args:
            case_id (string): ID of the patient.
        Returns:
            a numpy array containing 32 slices, 8 slice per each MRI scan
        """
        path = os.path.join(self.mri_data_dir, case_id+".npy")
        img = np.load(path , allow_pickle=True)
        img -= np.mean(img, keepdims=True)
        img /= np.std(img, keepdims=True)
        img = torch.as_tensor(img).float()
        # slices_to_keep = [3, 4, 5, 27, 28, 29]
        # img = img[slices_to_keep, :, :]  # Shape: (12, 256, 256)
        return img
    
    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = torch.Tensor([self.slide_data['disc_label'][idx]])
        event_time = torch.Tensor([self.slide_data[self.label_col][idx]])
        c = torch.Tensor([self.slide_data['censorship'][idx]])
        slide_ids = self.patient_dict[case_id]


        data_dir = self.data_dir
        
        if self.data_dir:
                if self.mode == 'path':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0) 
                    return_values =torch.zeros((1,1)), path_features, torch.zeros((1,1)), label, event_time, c,slide_ids
                    return return_values

                elif self.mode == 'genomic':
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (torch.zeros((1,1)), torch.zeros((1,1)), genomic_features.unsqueeze(dim=0), label, event_time, c, slide_ids)
                
                elif self.mode =='radio_3D':
                    T1 = self.load_mri_3D(case_id, 'T1')
                    T2 = self.load_mri_3D(case_id, 'T2')
                    FLAIR = self.load_mri_3D(case_id, 'Flair')
                    # Stack MRI images as channels
                    mri_tensors = torch.stack([ T2, FLAIR], dim=0)
                    # mri_tensors = torch.stack([T1c, T2, FLAIR], dim=0)
                    mri_tensors = mri_tensors.unsqueeze(0)
                    return (mri_tensors, torch.zeros((1, 1)), torch.zeros((1, 1)), label, event_time, c, slide_ids) 

                elif self.mode =='radio_2.5D':
                    mri_tensors = self.load_mri_2_5D(case_id)
                    mri_tensors = mri_tensors.unsqueeze(0)
                    return_values = (mri_tensors, torch.zeros((1, 1)), torch.zeros((1, 1)), label, event_time,  c, slide_ids)
                    return return_values

                elif self.mode == 'pathomic':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (torch.zeros((1,1)) , path_features, genomic_features.unsqueeze(dim=0), label, event_time, c, slide_ids)
                
                elif self.mode =='radiomic':
                    mri_tensors = self.load_mri2(case_id)
                    mri_tensors = mri_tensors
                    # mri_tensors = mri_tensors.permute(1, 2, 0) # for healnet
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    
                    return (mri_tensors.unsqueeze(0), torch.zeros((1, 1)), genomic_features.unsqueeze(dim=0), label, event_time, c, slide_ids) 
                
                elif self.mode =='radiopath':
                    T1c = self.load_mri(case_id, 'T1c')
                    T2 = self.load_mri(case_id, 'T2')
                    FLAIR = self.load_mri(case_id, 'Flair')
                    mask = self.load_mri(case_id, 'mask')
                    mri_tensors = torch.stack([T1c, T2, FLAIR, mask], dim=0)
                    mri_tensors = mri_tensors.unsqueeze(0)
                    
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)

                    return (mri_tensors, path_features, torch.zeros((1, 1)), label, event_time, c, slide_ids) 
                
                elif self.mode =='radiopathomics':
                    mri_tensors = self.load_mri2(case_id)
                    mri_tensors = mri_tensors.unsqueeze(0)
                    
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)

                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])

                    return (mri_tensors, path_features, genomic_features.unsqueeze(dim=0), label, event_time, c, slide_ids) 
                
def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index = True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

	df.to_csv(filename)

class Generic_Split(Generic_MIL_Dataset):
    def __init__(self, slide_data, metadata, mode, mri_data_dir,
        signatures=None, data_dir=None, label_col=None, patient_dict=None, num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.mode = mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.mri_data_dir = mri_data_dir
        self.case_ids = self.slide_data['case_id']
        print('Generic_Split')

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

        self.genomic_features = self.slide_data.drop(self.metadata, axis=1)


    def get_scaler(self):
        scaler_omic = StandardScaler().fit(self.genomic_features)
        return (scaler_omic,)

    def apply_scaler(self, scalers: tuple=None):
        transformed = pd.DataFrame(scalers[0].transform(self.genomic_features))
        transformed.columns = self.genomic_features.columns
        self.genomic_features = transformed