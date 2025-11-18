import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)

class StochasticDepth(nn.Module):
    """Per-sample stochastic depth. drop_prob in [0,1)."""
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = float(drop_prob)
    def forward(self, x, residual):
        if not self.training or self.drop_prob == 0.0:
            return x + residual
        keep = 1.0 - self.drop_prob
        mask = torch.empty(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x + residual * mask / keep

# ---- robust norm helpers ----
def make_norm2d(c: int, use_gn: bool, default_groups: int = 8):
    """Return a valid 2D norm even for awkward channel counts (e.g., C=1)."""
    if not use_gn:
        return nn.InstanceNorm2d(c, affine=True)
    g = math.gcd(default_groups, c) or 1  # ensure divisor of c
    return nn.GroupNorm(g, c)

def make_norm1d(c: int, use_gn: bool):
    """For 1D vectors, prefer LayerNorm when GN is requested."""
    return nn.LayerNorm(c) if use_gn else nn.InstanceNorm1d(c, affine=True)

class AttentionLayer(nn.Module):
    def __init__(self, channel, reduction=8, dilation=4, attn_dropout=0.1, norm="gn"):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(attn_dropout)
        self.sigmoid = nn.Sigmoid()
        use_gn = (norm == "gn")

        # channel attention (1D path)
        self.c = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            make_norm1d(channel, use_gn)
        )

        # spatial attention (2D path)
        self.s = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            make_norm2d(channel // reduction, use_gn),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel // reduction, 3, padding=dilation, dilation=dilation, bias=False),
            make_norm2d(channel // reduction, use_gn),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel // reduction, 3, padding=dilation, dilation=dilation, bias=False),
            make_norm2d(channel // reduction, use_gn),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, 1, 1, bias=False),
            make_norm2d(1, use_gn)  # valid for C=1 (GN->GroupNorm(1,1) or IN2d(1))
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y_c = self.avg_pool(x).view(b, c)
        y_c = self.c(y_c).view(b, c, 1, 1)
        y_s = self.s(x)
        y = self.sigmoid(self.drop(y_c + y_s))
        return x + x * y

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, first_block=False, sd_prob=0.0, norm="gn", p_spatial_drop=0.0):
        super().__init__()
        use_gn = (norm == "gn")
        self.bn0 = make_norm2d(in_channel, use_gn)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = make_norm2d(out_channel, use_gn)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.first_block = first_block
        self.downsample = None
        if stride != 1 or in_channel != out_channel * self.expansion:
            self.downsample = conv1x1(in_channel, out_channel * self.expansion, stride)
        self.sd = StochasticDepth(sd_prob)
        self.drop2d = nn.Dropout2d(p_spatial_drop) if p_spatial_drop > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        if self.first_block:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
        else:
            out = self.bn0(x); out = self.relu(out)
            out = self.conv1(out); out = self.bn1(out); out = self.relu(out)
            out = self.conv2(out)

        out = self.drop2d(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.sd(out, identity)

class Res34_2p5D_Regularized(nn.Module):
    """
    Early-fusion 2.5D ResNet34-lite with anti-overfitting mods:
      - SliceDrop
      - Stochastic Depth per block
      - Attention with dropout
      - Dropout2d in blocks
      - MLP head with dropout + norm
      - GroupNorm/LayerNorm helpers (robust)
    """
    def __init__(
        self,
        layer_num=32,
        num_classes=4,
        p_slice_drop=0.15,
        sd_prob=0.1,
        attn_dropout=0.1,
        p_spatial_drop=0.05,
        head_hidden=128,
        head_dropout=0.3,
        norm="gn"  # "gn" or "in"
    ):
        super().__init__()
        self.layer_num = layer_num
        self.p_slice_drop = p_slice_drop
        use_gn = (norm == "gn")

        # stem
        self.init_layer = nn.Sequential(
            nn.Conv2d(layer_num, 64, kernel_size=7, stride=2, padding=3, bias=False),
            make_norm2d(64, use_gn),
            nn.ReLU(inplace=True),
        )

        # blocks (light)
        self.block1 = nn.Sequential(
            BasicBlock(64, 64, first_block=True, sd_prob=sd_prob, norm=norm, p_spatial_drop=p_spatial_drop),
            BasicBlock(64, 64, sd_prob=sd_prob, norm=norm, p_spatial_drop=p_spatial_drop),
        )

        self.attention1 = AttentionLayer(64, attn_dropout=attn_dropout, norm=norm)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # richer head
        self.head = nn.Sequential(
            nn.Linear(64, head_hidden, bias=False),
            nn.LayerNorm(head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, num_classes)
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _slice_drop(self, x):
        """x: [B, C, H, W]; randomly zero some channels (slices) during training."""
        if (not self.training) or self.p_slice_drop <= 0:
            return x
        B, C, H, W = x.shape
        k = max(1, int(round(self.p_slice_drop * C)))
        for b in range(B):
            idx = torch.randperm(C, device=x.device)[:k]
            x[b, idx] = 0
        return x

    def _prepare_input(self, x):
        # Accept [B,S,H,W] or [B,S,C,H,W]; fold slice*chan into channel dim
        if x.dim() == 4:
            B, S, H, W = x.shape
            x = x.view(B, S, 1, H, W)
        elif x.dim() == 5:
            B, S, C, H, W = x.shape
        else:
            raise ValueError(f"x_mri must be 4D/5D, got {tuple(x.shape)}")
        x = x.view(x.size(0), -1, x.size(-2), x.size(-1))  # [B, S*C, H, W]

        # pad/crop channel dim to layer_num
        cur = x.shape[1]
        if cur < self.layer_num:
            pad = torch.zeros(x.size(0), self.layer_num - cur, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        elif cur > self.layer_num:
            start = (cur - self.layer_num) // 2
            x = x[:, start:start + self.layer_num]
        return x

    def forward(self, **kwargs):
        x = kwargs["x_mri"]
        x = self._prepare_input(x)     # -> [B, layer_num, H, W]
        x = self._slice_drop(x)        # SliceDrop (train only)
        x = self.init_layer(x)
        x = self.block1(x)
        x = self.attention1(x)
        x = self.relu(x)
        x = self.gap(x).flatten(1)     # [B, 64]
        logits = self.head(x)          # [B, num_classes]
        return logits
