import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class AttentionPool(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):  # x: [B, S, D]
        w = self.attn(x)            # [B, S, 1]
        w = torch.softmax(w, dim=1) # attention over slices
        z = torch.sum(w * x, dim=1) # [B, D]
        return z, w

def _make_backbone(in_ch: int) -> nn.Module:
    """ResNet18 with adjustable input channels (1 or 3)."""
    m = resnet18(weights=None)
    if in_ch != 3:
        # replace first conv to accept in_ch
        old = m.conv1
        m.conv1 = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=False)
    # feature dim after global pooling is 512 for resnet18
    return nn.Sequential(*(list(m.children())[:-1]))  # up to avgpool; outputs [B, 512, 1, 1]

class Radio2p5DNet(nn.Module):
    """
    2.5D MRI encoder:
      - Slice-wise 2D backbone
      - Attention aggregation across slices
      - Linear head -> n_classes (hazard bins for risk OR class count for subtype)
    Forward signature matches your trainer calls: model(x_path=..., x_omic=..., x_mri=...)
    Only x_mri is used here.
    """
    def __init__(self, n_classes: int, in_ch: int = 1, embed_dim: int = 512):
        super().__init__()
        self.backbone = _make_backbone(in_ch)
        self.embed_dim = 512
        self.pool = AttentionPool(self.embed_dim, hidden=128)
        self.head = nn.Linear(self.embed_dim, n_classes)

    def forward(self, x_mri=None, x_path=None, x_omic=None):
        if x_mri is None:
            raise ValueError("Radio2p5DNet expects x_mri tensor")

        # Accept [B, S, C, H, W] or [B, S, H, W]
        if x_mri.dim() == 4:        # [B, S, H, W] -> add channel
            x_mri = x_mri.unsqueeze(2)  # [B, S, 1, H, W]
        assert x_mri.dim() == 5, f"Expected 5D tensor, got shape {tuple(x_mri.shape)}"

        B, S, C, H, W = x_mri.shape
        x = x_mri.reshape(B * S, C, H, W)               # slice batch
        f = self.backbone(x)                             # [B*S, 512, 1, 1]
        f = f.flatten(1)                                 # [B*S, 512]
        f = f.view(B, S, self.embed_dim)                 # [B, S, 512]

        z, _ = self.pool(f)                              # [B, 512]
        logits = self.head(z)                            # [B, n_classes]
        return logits
