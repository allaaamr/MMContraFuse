from models.Fusion.utils import FusionTechniques
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
        max_epochs=100,
        fusion_type: str = "average",
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
        self.fusion_type = fusion_type

        print("FUSION TYPE ",self.fusion_type)
        # Projection heads to map to shared embedding space
        # ContIG uses projection heads to map encodings to a common space
        self.genomics_projection = self._build_projection_head(
            genomics_output_dim, projection_dim
        )
        
        self.radiomics_projection = self._build_projection_head(
            radiomics_output_dim, projection_dim
        )

        # 
        self._init_fusion_layers(projection_dim)

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
    
    def _init_fusion_layers(self, dim: int) -> None:
       
        self._fusion_modules = nn.ModuleDict()
        self._fusion_params = nn.ParameterDict()

        if self.fusion_type == "weighted_average":
            self._fusion_params["genomic_weight"] = nn.Parameter(torch.tensor(0.5))

        if self.fusion_type == "bilinear":
            print("bilinear ")
            self._fusion_modules["bilinear"] = nn.Bilinear(dim, dim, dim)

        if self.fusion_type == "attention":
            print("attn ")
            self._fusion_modules["attention"] = nn.Sequential(
                nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 1)
            )

        if self.fusion_type == "gated":
            self._fusion_modules["gate"] = nn.Linear(dim * 2, dim)

        if self.fusion_type == "mlp":
            self._fusion_modules["mlp"] = nn.Sequential(
                nn.Linear(dim * 2, dim * 2), nn.ReLU(), nn.Dropout(0.2), nn.Linear(dim * 2, dim)
            )

        if self.fusion_type == "tucker":
            self._fusion_modules["tucker"] = nn.Sequential(
                nn.Linear(dim * dim, 256), nn.ReLU(), nn.Linear(256, dim)
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
    

    def fuse_embeddings(self, g_emb: torch.Tensor, r_emb: torch.Tensor) -> torch.Tensor:
        ft = self.fusion_type
        M = self._fusion_modules      # modules (nn.ModuleDict)
        P = self._fusion_params       # learnable params (nn.ParameterDict)

        if ft == "average":
            return FusionTechniques.average_fusion(g_emb, r_emb)

        if ft == "weighted_average":
            gw = torch.sigmoid(P["genomic_weight"]) if "genomic_weight" in P else torch.tensor(0.5, device=g_emb.device)
            # gw is a scalar tensor; convert to float for the utils
            return FusionTechniques.weighted_average_fusion(g_emb, r_emb, genomic_weight=float(gw))

        if ft == "max":
            return FusionTechniques.max_fusion(g_emb, r_emb)

        if ft == "min":
            return FusionTechniques.min_fusion(g_emb, r_emb)

        if ft == "concatenation":
            return FusionTechniques.concatenation_fusion(g_emb, r_emb)

        if ft == "hadamard":
            return FusionTechniques.hadamard_fusion(g_emb, r_emb)

        if ft == "l2_distance":
            return FusionTechniques.l2_distance_fusion(g_emb, r_emb)

        if ft == "bilinear":
            return FusionTechniques.bilinear_fusion(g_emb, r_emb, M["bilinear"])

        if ft == "attention":
            return FusionTechniques.attention_fusion(g_emb, r_emb, M["attention"])

        if ft == "gated":
            return FusionTechniques.gated_fusion(g_emb, r_emb, M["gate"])

        if ft == "mlp":
            return M["mlp"](torch.cat([g_emb, r_emb], dim=1))

        if ft == "tucker":
            return FusionTechniques.tucker_fusion(g_emb, r_emb, M["tucker"])

        if ft == "mfb":
            return FusionTechniques.mfb_fusion(g_emb, r_emb, factor_dim=5)

        raise ValueError(f"Unknown fusion type: {ft}")


    def get_embeddings(self, genomics_data=None, radiomics_data=None):
        self.eval()
        with torch.no_grad():
            if (genomics_data is not None) and (radiomics_data is not None):
                g, r = self(genomics_data, radiomics_data)
                return self.fuse_embeddings(g, r)
            if genomics_data is not None:
                g = self.genomics_projection(self.genomics_encoder.fc_omic(genomics_data))
                return F.normalize(g, dim=1)
            if radiomics_data is not None:
                r = self.radiomics_projection(self.radiomics_encoder.feature_extractor(radiomics_data))
                return F.normalize(r, dim=1)
        raise ValueError("Must provide at least one modality")

    def get_output_dim(self) -> int:
        return self.projection_dim * 2 if self.fusion_type == "concatenation" else self.projection_dim


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
    
    # def get_embeddings(self, genomics_data=None, radiomics_data=None):
    #     """
    #     Extract embeddings for downstream tasks
    #     Can extract from either or both modalities
    #     """
    #     self.eval()  # Set model to evaluation mode (disables dropout, batch norm updates)
    #     with torch.no_grad():  # Don't compute gradients (saves memory & speeds up inference)

    #         if genomics_data is not None and radiomics_data is not None:
    #             # Both modalities available - return fused representation
    #             # After contrastive training, g_emb and r_emb for the same patient are
    #             # semantically close in the embedding space. 
    #             # Averaging them gives us a combined representation that leverages both genomics and radiomics information.
    #             g_emb, r_emb = self(genomics_data, radiomics_data)
    #             # Simple average fusion (as done in ContIG for downstream tasks)
    #             fused_emb = (g_emb + r_emb) / 2
    #             return fused_emb
            
    #         elif genomics_data is not None:
    #             # Only genomics available
    #             g_features = self.genomics_encoder(genomics_data)
    #             g_emb = self.genomics_projection(g_features)
    #             return F.normalize(g_emb, dim=1)
    #         elif radiomics_data is not None:
    #             # Only radiomics available
    #             r_features = self.radiomics_encoder(radiomics_data)
    #             r_emb = self.radiomics_projection(r_features)
    #             return F.normalize(r_emb, dim=1)
    #         else:
    #             raise ValueError("Must provide at least one modality")
