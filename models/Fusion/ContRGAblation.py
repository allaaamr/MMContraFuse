import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.Encoder.genomic import SNN
from models.Encoder.radiomics_1D import Radiomics1DNet
from models.Fusion.utils import FusionTechniques

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ContRGModelWithAblation(nn.Module):
    """
    Extended ContRG model with configurable fusion techniques
    """
    def __init__(
        self,
        base_contrg_model,
        fusion_type='average',
        projection_dim=128,
        device='cuda'
    ):
        super().__init__()
        self.contrg_model = base_contrg_model
        self.fusion_type = fusion_type
        self.projection_dim = projection_dim
        self.device = device
        
        # Initialize fusion-specific layers if needed
        if fusion_type == 'weighted_average':
            # Learnable weight parameter
            self.genomic_weight = nn.Parameter(torch.tensor(0.6))
            
        elif fusion_type == 'bilinear':
            self.bilinear_layer = nn.Bilinear(
                projection_dim, projection_dim, projection_dim
            )
            
        elif fusion_type == 'attention':
            self.attention_layer = nn.Sequential(
                nn.Linear(projection_dim, projection_dim // 2),
                nn.ReLU(),
                nn.Linear(projection_dim // 2, 1)
            )
            
        elif fusion_type == 'gated':
            self.gate_layer = nn.Linear(projection_dim * 2, projection_dim)
            
        elif fusion_type == 'tucker':
            # Tucker decomposition layer
            self.tucker_layer = nn.Sequential(
                nn.Linear(projection_dim * projection_dim, 256),
                nn.ReLU(),
                nn.Linear(256, projection_dim)
            )
            
        elif fusion_type == 'mlp':
            # MLP fusion
            self.mlp_fusion = nn.Sequential(
                nn.Linear(projection_dim * 2, projection_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(projection_dim * 2, projection_dim),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim)
            )
    
    def get_embeddings(self, genomics_data=None, radiomics_data=None):
        """
        Extract embeddings using specified fusion technique
        """
        self.contrg_model.eval()
        
        with torch.no_grad():
            if genomics_data is not None and radiomics_data is not None:
                # Get individual embeddings
                g_emb, r_emb = self.contrg_model(genomics_data, radiomics_data)
                
                # Apply selected fusion technique
                if self.fusion_type == 'average':
                    fused_emb = FusionTechniques.average_fusion(g_emb, r_emb)
                    
                elif self.fusion_type == 'weighted_average':
                    weight = torch.sigmoid(self.genomic_weight)
                    fused_emb = FusionTechniques.weighted_average_fusion(
                        g_emb, r_emb, weight
                    )
                    
                elif self.fusion_type == 'max':
                    fused_emb = FusionTechniques.max_fusion(g_emb, r_emb)
                    
                elif self.fusion_type == 'min':
                    fused_emb = FusionTechniques.min_fusion(g_emb, r_emb)
                    
                elif self.fusion_type == 'concatenation':
                    fused_emb = FusionTechniques.concatenation_fusion(g_emb, r_emb)
                    
                elif self.fusion_type == 'hadamard':
                    fused_emb = FusionTechniques.hadamard_fusion(g_emb, r_emb)
                    
                elif self.fusion_type == 'l2_distance':
                    fused_emb = FusionTechniques.l2_distance_fusion(g_emb, r_emb)
                    
                elif self.fusion_type == 'bilinear':
                    fused_emb = FusionTechniques.bilinear_fusion(
                        g_emb, r_emb, self.bilinear_layer
                    )
                    
                elif self.fusion_type == 'attention':
                    fused_emb = FusionTechniques.attention_fusion(
                        g_emb, r_emb, self.attention_layer
                    )
                    
                elif self.fusion_type == 'gated':
                    fused_emb = FusionTechniques.gated_fusion(
                        g_emb, r_emb, self.gate_layer
                    )
                    
                elif self.fusion_type == 'tucker':
                    fused_emb = FusionTechniques.tucker_fusion(
                        g_emb, r_emb, self.tucker_layer
                    )
                    
                elif self.fusion_type == 'mfb':
                    fused_emb = FusionTechniques.mfb_fusion(g_emb, r_emb)
                    
                elif self.fusion_type == 'mlp':
                    concat = torch.cat([g_emb, r_emb], dim=1)
                    fused_emb = self.mlp_fusion(concat)
                    
                else:
                    raise ValueError(f"Unknown fusion type: {self.fusion_type}")
                
                return fused_emb
                
            elif genomics_data is not None:
                g_features = self.contrg_model.genomics_encoder.fc_omic(genomics_data)
                g_emb = self.contrg_model.genomics_projection(g_features)
                return F.normalize(g_emb, dim=1)
                
            elif radiomics_data is not None:
                r_features = self.contrg_model.radiomics_encoder.feature_extractor(radiomics_data)
                r_emb = self.contrg_model.radiomics_projection(r_features)
                return F.normalize(r_emb, dim=1)
                
            else:
                raise ValueError("Must provide at least one modality")


def run_ablation_study(contrg_model, train_loader, val_loader, args):
    """
    Run ablation study with different fusion techniques
    """
    fusion_techniques = [
        'concatenation','bilinear','attention', 'gated',
        'average', 'weighted_average', 'max', 'min', 
        'hadamard', 'l2_distance', 'mlp', 'mfb'
    ]
    
    results = {}
    
    for fusion_type in fusion_techniques:
        print(f"\n{'='*50}")
        print(f"Testing fusion: {fusion_type}")
        print(f"{'='*50}")
        
        # Create model with specific fusion
        model_with_fusion = ContRGModelWithAblation(
            base_contrg_model=contrg_model,
            fusion_type=fusion_type,
            projection_dim=args.projection_dim,
            device=args.device
        )
        
        #  For concatenation fusion, the downstream
        # classifier input dimension becomes projection_dim * 2
        if fusion_type == 'concatenation':
            downstream_input_dim = args.projection_dim * 2
        else:
            downstream_input_dim = args.projection_dim
        
    return results