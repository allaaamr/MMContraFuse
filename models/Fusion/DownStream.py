import torch
import torch.nn as nn


class DownstreamModel(nn.Module):
    """
    Downstream model that uses pre-trained ContRG encoders
    Can be used for both linear probing and fine-tuning
    """
    def __init__(self, contrg_model, num_classes, freeze_encoder=True):
        super(DownstreamModel, self).__init__()
        
        # Store the pre-trained encoder
        self.contrg_model = contrg_model
        self.freeze_encoder = freeze_encoder
        
        # Freeze encoder parameters if doing linear probing
        if freeze_encoder:
            for param in self.contrg_model.parameters():
                param.requires_grad = False
        
        # Get the embedding dimension from ContRG
        embedding_dim = contrg_model.fusion_dim
        
        # Classification/regression head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x_path=None, x_omic=None, x_mri=None):
        """
        Forward pass matching the signature in core_utils.py
        
        Args:
            x_path: WSI features (not used in genomic/radiomic modes)
            x_omic: Genomic features
            x_mri: MRI/Radiomics features
        """
        # Set to eval mode if encoder is frozen
        if self.freeze_encoder:
            self.contrg_model.eval()
        
        # Get embeddings from ContRG model
        with torch.set_grad_enabled(not self.freeze_encoder):
            embeddings = self.contrg_model.get_embeddings(x_omic, x_mri)
        
        # Pass through classifier
        output = self.classifier(embeddings)
        
        return output