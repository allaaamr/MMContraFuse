import torch
import numpy as np
import torch.nn as nn


def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + SELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.SELU(),
            nn.AlphaDropout(p=dropout, inplace=False))
