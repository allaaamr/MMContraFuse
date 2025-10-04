import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.Encoder.genomic import SNN
from models.Encoder.radiomics_1D import Radiomics1DNet


class ContRGModel(pl.LightningModule):
    """
    ContRG: Contrastive learning for Radiology and Genetics
    
    This implementation follows the CVPR 2022 paper:
    "ContIG: Self-supervised Multimodal Contrastive Learning for 
    Medical Imaging with Genetics"
    """
    def __init__(
        self,
        omic_input_dim: int, 
        radio_input_dim: int,
        radio_type='1D',
        genomics_output_dim=256,
        radiomics_output_dim=64,  # The dimensionality of the last layer before the classifier of the radiomics model
        projection_dim=128,
        temperature=0.1,
        learning_rate=1e-3,
        weight_decay=1e-6,
        max_epochs=100
    ):
        super().__init__()
        # Genomics encoder
        self.genomics_encoder = SNN( omic_input_dim)
        # Radiomics encoder
        if radio_type=="1D":
            self.radiomics_encoder = Radiomics1DNet(radio_input_dim)
        elif radio_type=="2.5D": #
            self.radiomics_encoder = Radiomics1DNet()
        elif radio_type=="3D": # 
            self.radiomics_encoder = Radiomics1DNet()
        
        # Projection heads to map to shared embedding space
        # ContIG uses projection heads to map encodings to a common space
        self.genomics_projection = self._build_projection_head(
            genomics_output_dim, projection_dim
        )
        
        self.radiomics_projection = self._build_projection_head(
            radiomics_output_dim, projection_dim
        )
        
        # Hyperparameters
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.projection_dim = projection_dim
        
        # self.save_hyperparameters(ignore=['genomics_encoder', 'radiomics_encoder'])
    
    def _build_projection_head(self, input_dim, output_dim):
        """
        Build projection head as in ContIG paper
        Two-layer MLP with ReLU activation
        """
        return nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, genomics_data, radiomics_data):
        """
        Forward pass through both encoders and projection heads
        Returns L2-normalized embeddings
        """
        # Encode each modality (Using only the feature extractor part without the classifier)
        genomics_features = self.genomics_encoder.fc_omic(genomics_data)
        radiomics_features = self.radiomics_encoder.feature_extractor(radiomics_data)
        
        # Project to shared embedding space
        genomics_embedding = self.genomics_projection(genomics_features)
        radiomics_embedding = self.radiomics_projection(radiomics_features)
        
        # L2 normalize (critical for contrastive learning)
        genomics_embedding = F.normalize(genomics_embedding, dim=1)
        radiomics_embedding = F.normalize(radiomics_embedding, dim=1)
        
        return genomics_embedding, radiomics_embedding
    
    def contrastive_loss(self, genomics_emb, radiomics_emb):
        """
        Bidirectional contrastive loss (InfoNCE)
        
        For each (genomics, radiomics) pair in the batch:
        - The pair itself is the positive sample
        - All other pairs in the batch are negative samples
        
        This is computed bidirectionally (genomics->radiomics and radiomics->genomics)
        """
        batch_size = genomics_emb.shape[0]
        
        # Compute similarity matrix: (batch_size, batch_size)
        # Each element (i,j) is the similarity between genomics[i] and radiomics[j]
        logits = torch.matmul(genomics_emb, radiomics_emb.T) / self.temperature
        
        # Labels: diagonal elements are positive pairs (same person)
        labels = torch.arange(batch_size, device=self.device)
        
        # Cross-entropy loss in both directions
        loss_g2r = F.cross_entropy(logits, labels)  # genomics -> radiomics
        loss_r2g = F.cross_entropy(logits.T, labels)  # radiomics -> genomics
        
        # Symmetric loss
        loss = (loss_g2r + loss_r2g) / 2
        
        return loss
    
    def training_step(self, batch, batch_idx):
        """Training step for contrastive pre-training"""
        radiomics_data, data_WSI, genomics_data, y_disc, event_time, censor, slide_ids = batch
        
        # Forward pass
        genomics_emb, radiomics_emb = self(genomics_data, radiomics_data)
        
        # Compute contrastive loss
        loss = self.contrastive_loss(genomics_emb, radiomics_emb)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        radiomics_data, data_WSI, genomics_data, y_disc, event_time, censor, slide_ids = batch

        
        genomics_emb, radiomics_emb = self(genomics_data, radiomics_data)
        loss = self.contrastive_loss(genomics_emb, radiomics_emb)
        
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler as in ContIG"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=0
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
    
    def get_embeddings(self, genomics_data=None, radiomics_data=None):
        """
        Extract embeddings for downstream tasks
        Can extract from either or both modalities
        """
        self.eval()  # Set model to evaluation mode (disables dropout, batch norm updates)
        with torch.no_grad():  # Don't compute gradients (saves memory & speeds up inference)

            if genomics_data is not None and radiomics_data is not None:
                # Both modalities available - return fused representation
                # After contrastive training, g_emb and r_emb for the same patient are
                # semantically close in the embedding space. 
                # Averaging them gives us a combined representation that leverages both genomics and radiomics information.
                g_emb, r_emb = self(genomics_data, radiomics_data)
                # Simple average fusion (as done in ContIG for downstream tasks)
                fused_emb = (g_emb + r_emb) / 2
                return fused_emb
            
            elif genomics_data is not None:
                # Only genomics available
                g_features = self.genomics_encoder(genomics_data)
                g_emb = self.genomics_projection(g_features)
                return F.normalize(g_emb, dim=1)
            elif radiomics_data is not None:
                # Only radiomics available
                r_features = self.radiomics_encoder(radiomics_data)
                r_emb = self.radiomics_projection(r_features)
                return F.normalize(r_emb, dim=1)
            else:
                raise ValueError("Must provide at least one modality")
