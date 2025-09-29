import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel
import clip

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features_a, features_b):
        """
        Compute contrastive loss between two sets of features
        Args:
            features_a: Features from modality A [batch_size, dim]
            features_b: Features from modality B [batch_size, dim]
        """
        # Normalize features
        features_a = F.normalize(features_a, dim=1)
        features_b = F.normalize(features_b, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features_a, features_b.T) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        batch_size = features_a.shape[0]
        labels = torch.arange(batch_size).to(features_a.device)
        
        # Compute cross-entropy loss in both directions
        loss_a = F.cross_entropy(similarity_matrix, labels)
        loss_b = F.cross_entropy(similarity_matrix.T, labels)
        
        return (loss_a + loss_b) / 2