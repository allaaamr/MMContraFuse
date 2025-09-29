from collections import OrderedDict
from os.path import join
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Encoder.utils import *

class SNN(nn.Module):
    def __init__(self, omic_input_dim: int, model_size_omic: str='small', n_classes: int=4):
        super(SNN, self).__init__()
        self.n_classes = n_classes
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        fc_omic = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)
        self.classifier = nn.Linear(hidden[-1], n_classes)
        # init_max_weights(self)


    def forward(self, x_mri, x_path, x_omic):
        h = self.fc_omic(x_omic)
        h  = self.classifier(h) # logits needs to be a [B x 4] vector  or a [B x 2] vector depending on downstream task
        assert len(h.shape) == 2 and h.shape[1] == self.n_classes


        return h