import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FusionTechniques:
    """
    Collection of fusion techniques for multi-modal embeddings ablation study
    """
    
    @staticmethod
    def average_fusion(g_emb, r_emb):
        """Simple average (baseline)"""
        return (g_emb + r_emb) / 2
    
    @staticmethod
    def weighted_average_fusion(g_emb, r_emb, genomic_weight=0.5):
        """Weighted average with learnable or fixed weights"""
        radiomics_weight = 1 - genomic_weight
        return genomic_weight * g_emb + radiomics_weight * r_emb
    
    @staticmethod
    def max_fusion(g_emb, r_emb):
        """Element-wise maximum fusion"""
        return torch.max(g_emb, r_emb)
    
    @staticmethod
    def min_fusion(g_emb, r_emb):
        """Element-wise minimum fusion"""
        return torch.min(g_emb, r_emb)
    
    @staticmethod
    def concatenation_fusion(g_emb, r_emb):
        """Concatenate embeddings (doubles dimension)"""
        return torch.cat([g_emb, r_emb], dim=1)
    
    @staticmethod
    def hadamard_fusion(g_emb, r_emb):
        """Element-wise multiplication (Hadamard product)"""
        return g_emb * r_emb
    
    @staticmethod
    def l2_distance_fusion(g_emb, r_emb):
        """L2 distance as fusion (captures difference)"""
        return torch.abs(g_emb - r_emb)
    
    @staticmethod
    def bilinear_fusion(g_emb, r_emb, bilinear_layer):
        """Bilinear pooling fusion
        Requires: bilinear_layer = nn.Bilinear(dim, dim, output_dim)
        """
        return bilinear_layer(g_emb, r_emb)
    
    @staticmethod
    def attention_fusion(g_emb, r_emb, attention_layer):
        """Attention-based fusion
        Requires: attention_layer to compute attention weights
        """
        # Stack embeddings
        stacked = torch.stack([g_emb, r_emb], dim=1)  # [batch, 2, dim]
        
        # Compute attention scores
        attn_scores = attention_layer(stacked.view(-1, stacked.size(-1)))
        attn_scores = attn_scores.view(stacked.size(0), 2, -1).mean(dim=-1)  # [batch, 2]
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Weighted sum
        fused = attn_weights[:, 0:1] * g_emb + attn_weights[:, 1:2] * r_emb
        return fused
    
    @staticmethod
    def gated_fusion(g_emb, r_emb, gate_layer):
        """Gated fusion mechanism
        Requires: gate_layer = nn.Linear(dim*2, dim)
        """
        concatenated = torch.cat([g_emb, r_emb], dim=1)
        gate = torch.sigmoid(gate_layer(concatenated))
        return gate * g_emb + (1 - gate) * r_emb
    
    @staticmethod
    def tucker_fusion(g_emb, r_emb, tucker_layer):
        """Tucker decomposition fusion
        Requires custom Tucker layer implementation
        """
        # Outer product
        outer = torch.einsum('bi,bj->bij', g_emb, r_emb)
        # Flatten and project
        return tucker_layer(outer.view(outer.size(0), -1))
    
    @staticmethod
    def mfb_fusion(g_emb, r_emb, factor_dim=5):
        """Multi-modal Factorized Bilinear pooling (MFB)"""
        batch_size = g_emb.size(0)
        dim = g_emb.size(1)
        
        # Expand and reshape for factorization
        g_exp = g_emb.unsqueeze(1).expand(-1, factor_dim, -1)
        r_exp = r_emb.unsqueeze(1).expand(-1, factor_dim, -1)
        
        # Element-wise multiplication and sum pooling
        fusion = (g_exp * r_exp).sum(dim=1)
        
        # Power normalization and L2 normalization
        fusion = torch.sign(fusion) * torch.sqrt(torch.abs(fusion) + 1e-8)
        fusion = F.normalize(fusion, p=2, dim=1)
        
        return fusion
