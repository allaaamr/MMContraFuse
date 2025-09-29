import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel
import clip
import torchvision.models as models

class MultimodalEncoder(nn.Module):
    def __init__(self, genomics_encoder_name='bert-base-uncased',
                 radiomics_encoder_name='resnet50',
                 fusion_dim=512,
                 projection_dim=256):
        super(MultimodalEncoder, self).__init__()
        
        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(genomics_encoder_name)
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, projection_dim)
        
        # Image encoder (using pre-trained ResNet)
        
        self.image_encoder = models.resnet50(pretrained=True)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, projection_dim)
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(projection_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, projection_dim)
        )
        
        # Contrastive loss
        self.contrastive_loss = ContrastiveLoss()
        
    def encode_text(self, text_input):
        """Encode text input"""
        with torch.no_grad():
            outputs = self.text_encoder(**text_input)
            pooled_output = outputs.pooler_output
        
        text_features = self.text_projection(pooled_output)
        return F.normalize(text_features, dim=1)
    
    def encode_image(self, images):
        """Encode image input"""
        image_features = self.image_encoder(images)
        return F.normalize(image_features, dim=1)
    
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
        text_features = self.encode_text(text_input)
        image_features = self.encode_image(images)
        
        # Compute contrastive losses for individual modalities
        text_image_loss = self.contrastive_loss(text_features, image_features)
        
        # Fuse modalities
        fused_features = self.fuse_modalities(text_features, image_features)
        
        # Additional contrastive loss with fused features
        fusion_text_loss = self.contrastive_loss(fused_features, text_features)
        fusion_image_loss = self.contrastive_loss(fused_features, image_features)
        
        total_loss = text_image_loss + 0.5 * (fusion_text_loss + fusion_image_loss)
        
        if return_individual:
            return fused_features, text_features, image_features, total_loss
        
        return fused_features, total_loss