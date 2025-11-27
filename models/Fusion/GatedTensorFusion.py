import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Encoder.utils import SNN_Block
from models.Encoder.DeepRiskA import *
from typing import *
# from models.utils import *

class LRBilinearFusion(nn.Module):
    def __init__(self, skip=0, use_bilinear=0, gate1=1, gate2=1, dim1=128, dim2=128, 
                 scale_dim1=1, scale_dim2=1, dropout_rate=0.25,
                rank=16, output_dim=4):
        super(LRBilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2
        self.rank = rank
        self.output_dim = output_dim

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))
        
        
        self.h1_factor = Parameter(torch.Tensor(self.rank, dim1 + 1, output_dim))
        self.h2_factor = Parameter(torch.Tensor(self.rank, dim2 + 1, output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))
        xavier_normal(self.h1_factor)
        xavier_normal(self.h2_factor)
        xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

        #init_max_weights(self)

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = F.dropout(self.linear_h1(vec1), 0.25)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = F.dropout(self.linear_h2(vec2), 0.25)
            o2 = self.linear_o2(h2)

        ### Fusion
        DTYPE = torch.cuda.FloatTensor
        _o1 = torch.cat((Variable(torch.ones(1, 1).type(DTYPE), requires_grad=False), o1), dim=1)
        _o2 = torch.cat((Variable(torch.ones(1, 1).type(DTYPE), requires_grad=False), o2), dim=1)
        o1_fusion = torch.matmul(_o1, self.h1_factor)
        o2_fusion = torch.matmul(_o2, self.h2_factor)
        fusion_zy = o1_fusion * o2_fusion
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        return output

class BilinearFusion(nn.Module):
    def __init__(self, skip=0, use_bilinear=0, gate1=1, gate2=1, dim1=128, dim2=128, scale_dim1=1, scale_dim2=1, mmhid=256, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        # Reduces the dimensionality of the input vectors through linear transformations and ReLU activations.Controlled by scaling factors scale_dim1
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) 
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))
        
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) 
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        #' Reduces high-dimensional outer product features.'
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), 256), nn.ReLU())
        #Produces the final fused embedding.
        self.encoder2 = nn.Sequential(nn.Linear(256+skip_dim, mmhid), nn.ReLU())
        #init_max_weights(self)

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) 
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) 
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        ### Fusion
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        o1 = torch.cat((o1, torch.ones(o1.shape[0], 1).to(device)), 1)
        o2 = torch.cat((o2, torch.ones(o2.shape[0], 1).to(device)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out

    
class RadiomicMMF(nn.Module):
    def __init__(self, omic_input_dim, in_channels=4, n_classes=4, fusion='bilinear',
                 scale_dim1=8, scale_dim2=8, gate_path=1, gate_omic=1, skip=True,
                 model_size_omic='small', layer_num=32 ):
        super(RadiomicMMF, self).__init__()

        self.size_dict_omic = {'small': [512,512], 'big': [1024, 1024, 1024, 256]}
        hidden = self.size_dict_omic[model_size_omic]
        self.fusion = fusion
        self.n_classes = n_classes

        # ----- Radiomics trunk -----
        self.init_layer = nn.Sequential(
            nn.Conv2d(layer_num, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.ReLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            BasicBlock(in_channel=64, out_channel=64, first_block=True),
            BasicBlock(in_channel=64, out_channel=64),
        )
        self.attention1 = AttentionLayer(channel=64, rank=1)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 256)  # (kept exactly as in your code)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # ----- Genomics trunk -----
        Block = SNN_Block
        fc_omic = [Block(dim1=omic_input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)
        self.classifier = nn.Linear(hidden[-1], n_classes)  # (kept, even if unused in forward)

        # ----- Fusion -----
        # Your BilinearFusion returns a 256-d fused token via mmhid=256
        self.mm = BilinearFusion(dim1=64, dim2=hidden[-1],
                                 scale_dim1=scale_dim1, gate1=gate_path,
                                 scale_dim2=scale_dim2, gate2=gate_omic,
                                 skip=skip, mmhid=256)

        # ----- Head (unchanged) -----
        self.classifier_mm = nn.Linear(256, n_classes)

    # ------------ internal helpers (don’t change behavior) ------------
    def _mri_repr(self, x_mri: torch.Tensor) -> torch.Tensor:
        x = self.init_layer(x_mri)      # [B,64,H/2,W/2]
        x = self.block1(x)
        x = self.attention1(x)
        x = self.relu(x)
        x = self.gap(x)                 # [B,64,1,1]
        x = x.view(x.size(0), -1)       # [B,64]
        return x

    def _omic_repr(self, x_omic: torch.Tensor) -> torch.Tensor:
        return self.fc_omic(x_omic)     # [B, hidden[-1]]

    # ------------ debias-compatible API (new) ------------
    def repr(self, x_mri=None, x_path=None, x_omic=None):
        """
        Return the fused representation BEFORE the final classifier_mm.
        """
        if x_mri is None or x_omic is None:
            raise ValueError("repr() expects both x_mri and x_omic")
        h_mri = self._mri_repr(x_mri)      # [B,64]
        h_omic = self._omic_repr(x_omic)   # [B, hidden[-1]]
        z = self.mm(h_mri, h_omic)         # [B,256]
        return z

    def classify_from_repr(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply the unchanged final head to a fused representation.
        """
        return self.classifier_mm(z)       # [B, n_classes]

    # ------------ forward (unchanged outcome) ------------
    def forward(self, x_mri=None, x_path=None, x_omic=None):
        z = self.repr(x_mri=x_mri, x_omic=x_omic)  # fused token [B,256]
        h_mm = self.classify_from_repr(z)          # logits via your original head
        assert len(h_mm.shape) == 2 and h_mm.shape[1] == self.n_classes
        return h_mm


