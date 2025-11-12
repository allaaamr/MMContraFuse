import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Encoder.deeprisk import conv3x3, conv1x1, StochasticDepth, AttentionLayer, make_norm2d

import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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


# -----------------------------
# Trunks: MRI and Genomics that expose features at configurable depths
# -----------------------------
class MRITrunk(nn.Module):
    """A truncated version of Res34_2p5D_Regularized that can output features
    at different depths. It always returns a vector feature.

    fuse_point_mri: one of {"stem", "block1", "gap"}
        - "stem": after the initial stem conv (AvgPool to vector)
        - "block1": after block1 (+ attention) (AvgPool to vector)
        - "gap": the usual GAP output (default)
    """
    def __init__(self,
                 layer_num=32,
                 feature_dim=64,
                 p_slice_drop=0.15,
                 sd_prob=0.1,
                 attn_dropout=0.1,
                 p_spatial_drop=0.05,
                 norm="gn",
                 fuse_point_mri: Literal["stem","block1","gap"] = "gap"):
        super().__init__()
        self.layer_num = layer_num
        self.p_slice_drop = p_slice_drop
        use_gn = (norm == "gn")
        self.fuse_point_mri = fuse_point_mri

        self.init_layer = nn.Sequential(
            nn.Conv2d(layer_num, 64, kernel_size=7, stride=2, padding=3, bias=False),
            make_norm2d(64, use_gn),
            nn.ReLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            BasicBlock(64, 64, first_block=True, sd_prob=sd_prob, norm=norm, p_spatial_drop=p_spatial_drop),
            BasicBlock(64, 64, sd_prob=sd_prob, norm=norm, p_spatial_drop=p_spatial_drop),
        )
        self.attention1 = AttentionLayer(64, attn_dropout=attn_dropout, norm=norm)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.feat_proj = nn.Identity()  # 64-dim already

    def _slice_drop(self, x):
        if (not self.training) or self.p_slice_drop <= 0:
            return x
        B, C, H, W = x.shape
        k = max(1, int(round(self.p_slice_drop * C)))
        for b in range(B):
            idx = torch.randperm(C, device=x.device)[:k]
            x[b, idx] = 0
        return x

    def _prepare_input(self, x):
        if x.dim() == 4:
            B, S, H, W = x.shape
            x = x.view(B, S, 1, H, W)
        elif x.dim() == 5:
            B, S, C, H, W = x.shape
        else:
            raise ValueError(f"x_mri must be 4D/5D, got {tuple(x.shape)}")
        x = x.view(x.size(0), -1, x.size(-2), x.size(-1))
        cur = x.shape[1]
        if cur < self.layer_num:
            pad = torch.zeros(x.size(0), self.layer_num - cur, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        elif cur > self.layer_num:
            start = (cur - self.layer_num) // 2
            x = x[:, start:start + self.layer_num]
        return x

    def forward(self, x_mri) -> torch.Tensor:
        x = self._prepare_input(x_mri)
        x = self._slice_drop(x)
        x = self.init_layer(x)              # [B,64,H/2,W/2]
        if self.fuse_point_mri == "stem":
            v = self.gap(x).flatten(1)      # [B,64]
            return self.feat_proj(v)
        x = self.block1(x)
        x = self.attention1(x)
        x = self.relu(x)
        if self.fuse_point_mri == "block1":
            v = self.gap(x).flatten(1)
            return self.feat_proj(v)
        # default: "gap"
        v = self.gap(x).flatten(1)
        return self.feat_proj(v)


class SNN_Block(nn.Module):
    def __init__(self, dim1, dim2, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim1, dim2, bias=False),
            nn.LayerNorm(dim2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
    def forward(self, x):
        return self.net(x)


class SNNTrunk(nn.Module):
    """Truncated SNN that can output features after k blocks (before classifier)."""
    def __init__(self, omic_input_dim: int, hidden: list, k_out: Optional[int] = None):
        super().__init__()
        assert len(hidden) >= 1
        self.hidden = hidden
        self.blocks = nn.ModuleList()
        in_dim = omic_input_dim
        for h in hidden:
            self.blocks.append(SNN_Block(in_dim, h, dropout=0.25))
            in_dim = h
        self.k_out = k_out if k_out is not None else len(hidden)
        self.out_dim = hidden[self.k_out - 1]

    def forward(self, x_omic) -> torch.Tensor:
        h = x_omic
        for i, blk in enumerate(self.blocks, start=1):
            h = blk(h)
            if i == self.k_out:
                break
        return h  # [B, out_dim]


# -----------------------------
# Fusion modules
# -----------------------------
class ConcatFusion(nn.Module):
    def __init__(self, dim_mri: int, dim_omic: int, dim_fuse: int = 128, dropout: float = 0.2):
        super().__init__()
        self.proj_mri = nn.Linear(dim_mri, dim_fuse // 2, bias=False)
        self.proj_omic = nn.Linear(dim_omic, dim_fuse // 2, bias=False)
        self.head = nn.Sequential(
            nn.LayerNorm(dim_fuse),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.out_dim = dim_fuse
    def forward(self, z_mri, z_omic):
        z = torch.cat([self.proj_mri(z_mri), self.proj_omic(z_omic)], dim=1)
        z = self.head(z)
        return z


class GatedAddFusion(nn.Module):
    """Simple gated additive fusion (per-feature gates)."""
    def __init__(self, dim_mri: int, dim_omic: int, dim_common: int = 128):
        super().__init__()
        self.pm = nn.Linear(dim_mri, dim_common, bias=False)
        self.po = nn.Linear(dim_omic, dim_common, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(dim_common * 2, dim_common),
            nn.ReLU(inplace=True),
            nn.Linear(dim_common, dim_common),
            nn.Sigmoid(),
        )
        self.out_dim = dim_common
    def forward(self, z_mri, z_omic):
        m = self.pm(z_mri)
        o = self.po(z_omic)
        g = self.gate(torch.cat([m, o], dim=1))
        return g * m + (1 - g) * o


class CrossAttentionFusion(nn.Module):
    """Self-attention over a 2-token sequence [mri, omic]."""
    def __init__(self, dim_mri: int, dim_omic: int, d_model: int = 128, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.pm = nn.Linear(dim_mri, d_model, bias=False)
        self.po = nn.Linear(dim_omic, d_model, bias=False)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.out_dim = d_model

    def forward(self, z_mri, z_omic):
        # tokens: [B, 2, D]
        tokens = torch.stack([self.pm(z_mri), self.po(z_omic)], dim=1)
        h, _ = self.attn(tokens, tokens, tokens)     # self-attn over {MRI, OMIC}
        h = self.norm1(tokens + h)                   # residual
        h2 = self.ff(h)
        h = self.norm2(h + h2)                       # residual

        # Pool 2 tokens to 1 fused token (mean or learnable pool)
        z = h.mean(dim=1)                            # [B, D]
        return z



# -----------------------------
# Top-level fusion model
# -----------------------------
class GenoMRI_Fusion(nn.Module):
    """
    Flexible fusion model for Genomics + 2.5D MRI.

    Args:
        omic_input_dim: number of genomic features
        omic_hidden: list of hidden sizes for SNN trunk (e.g., [512,512])
        n_classes: output dimension
        task: 'risk' or 'subtype' (only affects outside losses, head is logits)
        fusion: 'concat' | 'gated' | 'xattn'
        fuse_point_mri: 'stem' | 'block1' | 'gap'
        fuse_k_omic: output after k SNN blocks (1..len(omic_hidden))
        d_model / dim_fuse: internal fusion dimensions
    """
    def __init__(self,
                 omic_input_dim: int,
                 omic_hidden: list = [512, 512],
                 n_classes: int = 4,
                 task: str = 'risk',
                 fusion: Literal['concat','gated','xattn'] = 'concat',
                 fuse_point_mri: Literal['stem','block1','gap'] = 'gap',
                 fuse_k_omic: Optional[int] = None,
                 # MRI trunk params
                 layer_num: int = 32,
                 p_slice_drop: float = 0.15,
                 sd_prob: float = 0.1,
                 attn_dropout: float = 0.1,
                 p_spatial_drop: float = 0.05,
                 norm: str = 'gn',
                 # fusion dims
                 dim_fuse: int = 256,
                 d_model: int = 256,
                 nhead: int = 4,
                 head_hidden: int = 256,
                 head_dropout: float = 0.3):
        super().__init__()
        self.n_classes = n_classes
        self.task = task

        # trunks
        self.mri = MRITrunk(
            layer_num=layer_num,
            feature_dim=64,
            p_slice_drop=p_slice_drop,
            sd_prob=sd_prob,
            attn_dropout=attn_dropout,
            p_spatial_drop=p_spatial_drop,
            norm=norm,
            fuse_point_mri=fuse_point_mri,
        )
        self.snn = SNNTrunk(omic_input_dim=omic_input_dim, hidden=omic_hidden, k_out=fuse_k_omic)

        dim_mri = 64
        dim_omic = self.snn.out_dim

        # fusion
        if fusion == 'concat':
            self.fuse = ConcatFusion(dim_mri, dim_omic, dim_fuse=dim_fuse)
            fused_dim = self.fuse.out_dim
        elif fusion == 'gated':
            self.fuse = GatedAddFusion(dim_mri, dim_omic, dim_common=dim_fuse)
            fused_dim = self.fuse.out_dim
        elif fusion == 'xattn':
            self.fuse = CrossAttentionFusion(dim_mri, dim_omic, d_model=d_model, nhead=nhead)
            fused_dim = self.fuse.out_dim
        else:
            raise ValueError(f"Unknown fusion type: {fusion}")

        # classifier head on fused token
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(fused_dim, n_classes)
        )

    def forward(self, x_mri=None, x_path=None, x_omic=None):
        if x_mri is None or x_omic is None:
            raise ValueError("GenoMRI_Fusion expects both x_mri and x_omic")
        z_mri = self.mri(x_mri)    # [B, 64]
        z_omic = self.snn(x_omic)  # [B, Hk]
        z = self.fuse(z_mri, z_omic)
        logits = self.head(z)
        return logits


# -----------------------------
# Convenience factory to match your argparse flags
# -----------------------------
class FusionFactory:
    @staticmethod
    def build_from_args(args, omic_input_dim: int) -> nn.Module:
        # derive SNN hidden from args.model_size_omic
        size_dict_omic = {'small': [512, 512], 'big': [1024, 1024, 1024, 256]}
        hidden = size_dict_omic.get(getattr(args, 'model_size_omic', 'small'), [512, 512])

        # map args.fusion to our choices
        fusion_map = {
            'concat': 'concat',
            'bi_attn': 'xattn',   # treat bi_attn as cross-attn in this impl
            'tri_attn': 'xattn',  # alias
            'bi_contrast': 'gated',  # use gated add as simple alternative
            'tri_contrast': 'gated',
        }
        fusion = fusion_map.get(getattr(args, 'fusion', 'concat'), 'concat')

        # optional depth controls (provide new CLI if needed)
        fuse_point_mri = getattr(args, 'fuse_point_mri', 'gap')  # 'stem'|'block1'|'gap'
        fuse_k_omic = getattr(args, 'fuse_k_omic', None)         # int or None

        model = GenoMRI_Fusion(
            omic_input_dim=omic_input_dim,
            omic_hidden=hidden,
            n_classes=args.n_classes,
            task=args.task if hasattr(args, 'task') else 'risk',
            fusion=fusion,
            fuse_point_mri=fuse_point_mri,
            fuse_k_omic=fuse_k_omic,
            layer_num=args.layer_num,
            p_slice_drop=args.p_slice_drop,
            sd_prob=args.sd_prob,
            attn_dropout=args.attn_dropout,
            p_spatial_drop=args.p_spatial_drop,
            norm=args.norm,
            dim_fuse=args.dim_fuse,
            d_model=args.d_model,
            nhead=args.nhead,
            head_hidden=args.head_hidden,
            head_dropout=args.head_dropout,
        )
        return model
