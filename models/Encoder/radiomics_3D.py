import torch
import torch.nn as nn
import torchvision.models.video as vmodels

class Radiomics3DNet(nn.Module):
    """3D CNN encoder for MRI / volumetric radiomics data"""
    def __init__(self, in_channels=1, output_dim=256):
        super().__init__()
        # Use a standard 3D ResNet backbone (same as your MRI3DHead backbone)
        self.backbone = vmodels.r3d_18(weights=None)
        self.backbone.stem[0] = nn.Conv3d(in_channels, 64, kernel_size=(3,7,7),
                                          stride=(1,2,2), padding=(1,3,3), bias=False)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # final projection layer before contrastive head
        self.feature_extractor = nn.Sequential(
            self.backbone,
            nn.Linear(feat_dim, output_dim)
        )

    def forward(self, x):
        return self.feature_extractor(x)