import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel
import clip
import torchvision.models as models
from models.Encoder.genomic import SNN
from models.Encoder.radiomics_1D import Radiomics1DNet
import ContrastiveLoss

class MultimodalEncoder(nn.Module):
    def __init__(self,
                 omic_input_dim: int, 
                 radio_encoder='1D',
                 radio_output_features_dim = 64,  # The dimensionality of the last layer before the classifier of the radiomics model
                 n_classes: int=4,
                 fusion_dim=512,
                 projection_dim=256):
        super(MultimodalEncoder, self).__init__()
        
        # Genomics encoder
        self.genomics_encoder = SNN(omic_input_dim)
        
        self.radiomics_encoder = None
        # Radiomics encoder
        if radio_encoder=="1D":
            self.radiomics_encoder = Radiomics1DNet()
        elif radio_encoder=="2.5D": # Ahmed insert your model here
            self.radiomics_encoder = Radiomics1DNet()
        elif radio_encoder=="3D": # Mashrafi insert your model here
            self.radiomics_encoder = Radiomics1DNet()

        self.radiomics_encoder.projection = nn.Linear(radio_output_features_dim, 256)
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(projection_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, projection_dim)
        )
        
        # Contrastive loss
        self.contrastive_loss = ContrastiveLoss()
    
    
    def encode_genomics(self, genomics):
        """Encode genomics input"""
        # no_grad in case we want to freeze the encoders .. or else train the encoders too
        # with torch.no_grad():
            # pass through the network istelf fc omic, omit the last classification layer
        outputs = self.genomics_encoder.fc_omic(genomics)

        return F.normalize(outputs, dim=1)
    
    def encode_radiomics(self, images):
        """Encode image input"""
        # no_grad in case we want to freeze the encoders .. if commented then train the encoders too
        # with torch.no_grad(): #MAshrafi: if your 3D model is heavy you can avoid retrainign the whole model, freeze it using no grad
        
        # pass only to the feature extractor layers of the network (omit classification layer)
        radiomic_features = self.radiomics_encoder.feature_extractor(images)
        return F.normalize(radiomic_features, dim=1)
    
    def fuse_modalities(self, text_features, image_features):
        """Fuse text and image features with contrastive learning"""
        # Concatenate features
        fused_input = torch.cat([text_features, image_features], dim=1)
        
        # Apply fusion layer
        fused_features = self.fusion_layer(fused_input)
        
        return F.normalize(fused_features, dim=1)
    
    def forward(self, text_input, images, return_individual=False):
        """
        Forward pass with contrastive fusion
        """
        # Encode individual modalities
        genomics_features = self.encode_genomics(text_input)
        radiomics_features = self.encode_radiomics(images)
        
        # Compute contrastive losses for individual modalities
        text_image_loss = self.contrastive_loss(genomics_features, radiomics_features)
        
        # Fuse modalities
        fused_features = self.fuse_modalities(genomics_features, radiomics_features)
        
        # Additional contrastive loss with fused features
        fusion_text_loss = self.contrastive_loss(fused_features, genomics_features)
        fusion_image_loss = self.contrastive_loss(fused_features, radiomics_features)
        
        total_loss = text_image_loss + 0.5 * (fusion_text_loss + fusion_image_loss)
        
        if return_individual:
            return fused_features, genomics_features, radiomics_features, total_loss
        
        return fused_features, total_loss