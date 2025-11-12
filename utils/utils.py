import pickle
import numpy as np
import pdb
import math
import pickle
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
from itertools import islice
import matplotlib.pyplot as plt
import os
import warnings
from torch.utils.data import Dataset

class _ExceptionSafeDataset(Dataset):
    """
    Wraps a dataset to catch per-item exceptions and return None instead.
    The collate_fn must be able to drop None entries.
    """
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        try:
            return self.base[idx]
        except FileNotFoundError as e:
            # Try to show a helpful case id if available
            case_id = None
            try:
                if hasattr(self.base, "case_ids"):
                    case_id = self.base.case_ids[idx]
            except Exception:
                pass
            msg = f"[WARN] Missing file for index {idx}"
            if case_id is not None:
                msg += f" (case_id={case_id})"
            msg += f": {e}"
            warnings.warn(msg)
            return None
        except Exception as e:
            warnings.warn(f"[WARN] Skipping index {idx} due to error: {e}")
            return None

def pickle_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def unpickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def get_split_loader(split_dataset, training=False, testing=False, weighted=False, mode='coattn', batch_size=1):
    """
    return either the validation loader or training loader 
    """
    collate = collate_MIL
    kwargs = {}
    pin = torch.cuda.is_available()
    num_workers = 0  # keep 0 unless you have measured benefit
    kwargs.update(dict(pin_memory=pin, num_workers=num_workers, persistent_workers=False))
    safe_dataset = _ExceptionSafeDataset(split_dataset)  # <--- wrap here

    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(
                    safe_dataset,
                    batch_size=batch_size,
                    sampler=WeightedRandomSampler(weights, len(weights)),
                    collate_fn=collate,
                    **kwargs
                )
            else:
                loader = DataLoader(
                    safe_dataset,
                    batch_size=batch_size,
                    sampler=RandomSampler(split_dataset),
                    collate_fn=collate,
                    **kwargs
                )
        else:
            loader = DataLoader(
                safe_dataset,
                batch_size=1,
                sampler=SequentialSampler(split_dataset),
                collate_fn=collate,
                **kwargs
            )
    else:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace=False)
        loader = DataLoader(
            safe_dataset,
            batch_size=1,
            sampler=SubsetSequentialSampler(ids),
            collate_fn=collate,
            **kwargs
        )
    return loader

def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer

def make_weights_for_balanced_classes_split(dataset):
    """
    The variable weight_per_class is a list that stores the weight for each class.
      The idea is to give more weight to underrepresented classes (classes with fewer samples) 
      and less weight to overrepresented classes (classes with more samples).'
    """
    N = float(len(dataset))                                           
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                        
        weight[idx] = weight_per_class[y]                                  

    return torch.DoubleTensor(weight)

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def collate_MIL(batch):
    # Drop items that failed to load
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        # Signal to the training loop to skip this batch
        return None

    mri = torch.cat([item[0] for item in batch], dim=0).type(torch.FloatTensor)
    img = torch.cat([item[1] for item in batch], dim=0)
    omic = torch.cat([item[2] for item in batch], dim=0).type(torch.FloatTensor)
    label = torch.LongTensor([int(item[3].item()) for item in batch])
    event_time = torch.FloatTensor([item[4] for item in batch])
    c = torch.FloatTensor([item[5] for item in batch])
    slide_ids = [item[6] for item in batch]
    return [mri, img, omic, label, event_time, c, slide_ids]




def generate_stratified_kfold(cls_ids, samples, n_splits=5, seed=7):
    """
    Stratified K-fold over indices specified per-class in `cls_ids`.
    Each fold's validation set is disjoint; union of all val sets ~ all samples.
    Implicit val fraction per fold ≈ 1/n_splits (e.g., 20% for 5 folds).

    Args:
        cls_ids (list[np.ndarray|list]): class c -> indices of samples in class c
        samples (int): total number of samples (indices assumed 0..samples-1)
        n_splits (int): number of folds (e.g., 5)
        seed (int): RNG seed

    Yields:
        (train_idx, val_idx): sorted lists of indices for each fold
    """
    all_indices = np.arange(samples, dtype=int)
    folds = [list() for _ in range(n_splits)]
    rng = np.random.default_rng(seed)

    # For each class, shuffle its indices once and split into K chunks
    for c, ids in enumerate(cls_ids):
        ids = np.asarray(ids, dtype=int)
        ids = rng.permutation(ids)                               # shuffle once
        chunks = np.array_split(ids, n_splits)                   # near-equal chunks
        for k in range(n_splits):
            folds[k].extend(chunks[k].tolist())                  # add this class's k-th chunk to fold k

    # Build per-fold train/val
    for k in range(n_splits):
        val_idx = sorted(folds[k])
        train_idx = sorted(np.setdiff1d(all_indices, val_idx))
        yield train_idx, val_idx


def generate_split(cls_ids, samples, n_splits=5, seed=7, val_percent=0.2):
    """
    Generate train/validation splits with class-wise sampling.
    Ensures total validation set is val_percent of dataset,
    divided equally across classes.

    Args:
        cls_ids (list of arrays): cls_ids[c] contains indices of samples in class c
        samples (int): total number of samples
        n_splits (int): how many different splits to generate
        seed (int): random seed for reproducibility
        val_percent (float): percentage (0–1) of samples to use as validation (default 0.2)

    Yields:
        (train_ids, val_ids): lists of indices for train and validation
    """
    indices = np.arange(samples).astype(int)  # all possible sample indices
    
    # total number of validation samples
    total_val = int(round(samples * val_percent))
    n_classes = len(cls_ids)

    # distribute equally across classes
    per_class_val = total_val // n_classes
    val_num = [per_class_val] * n_classes

    np.random.seed(seed)  # ensure reproducibility
    
    for i in range(n_splits):
        all_val_ids = []        # will collect all validation indices
        sampled_train_ids = []  # will collect all training indices

        # iterate over classes
        for c in range(n_classes):
            possible_indices = np.intersect1d(cls_ids[c], indices)  # indices belonging to this class
            remaining_ids = possible_indices.copy()                 # start with all class members

            # --- Validation split ---
            if val_num[c] > 0:
                # randomly select val_num[c] samples for validation
                val_ids = np.random.choice(possible_indices, val_num[c], replace=False)
                # remove chosen validation ids from pool
                remaining_ids = np.setdiff1d(possible_indices, val_ids)
                # add them to the validation set
                all_val_ids.extend(val_ids)

            # --- Training split ---
            sampled_train_ids.extend(remaining_ids)

        # yield the current split (sorted for consistency)
        yield sorted(sampled_train_ids), sorted(all_val_ids)