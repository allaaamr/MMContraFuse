import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models import ResNet
import torchvision.models as models


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,
                     bias=False)


class AttentionLayer(nn.Module):
    def __init__(self, channel, rank, reduction=8, dilation=4):
        super(AttentionLayer, self).__init__()
        self.reduction = reduction
        self.dilation = dilation
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        # channel attention
        self.c = nn.Sequential()
        self.c.add_module('dense1_'+str(rank), nn.Linear(channel, channel // reduction, bias=False))
        self.c.add_module('relu1_'+str(rank), nn.ReLU(inplace=True))
        self.c.add_module('dense2_'+str(rank), nn.Linear(channel // reduction, channel, bias=False))
        self.c.add_module('bn0_'+str(rank), nn.InstanceNorm1d(channel))
        
        # spatial attention
        self.s = nn.Sequential()
        self.s.add_module('conv1_'+str(rank), nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1,
                          padding=0, bias=False))
        self.s.add_module('bn1_'+str(rank), nn.InstanceNorm2d(channel // reduction))
        self.s.add_module('relu2_'+str(rank), nn.ReLU(inplace=True))

        self.s.add_module('conv2_'+str(rank), nn.Conv2d(channel // reduction, channel // reduction,
                          kernel_size=3, stride=1, dilation=self.dilation, padding=4, bias=False))
        self.s.add_module('bn2_'+str(rank), nn.InstanceNorm2d(channel // reduction))
        self.s.add_module('relu3_'+str(rank), nn.ReLU(inplace=True))

        self.s.add_module('conv3_'+str(rank), nn.Conv2d(channel // reduction, channel // reduction,
                          kernel_size=3, stride=1, dilation=self.dilation, padding=4, bias=False))
        self.s.add_module('bn3_'+str(rank), nn.InstanceNorm2d(channel // reduction))
        self.s.add_module('relu4_'+str(rank), nn.ReLU(inplace=True))

        self.s.add_module('conv4_'+str(rank), nn.Conv2d(channel // reduction, 1, kernel_size=1, stride=1,
                          padding=0, bias=False))
        self.s.add_module('bn4_'+str(rank), nn.InstanceNorm2d(1))

    def forward(self, x):
        b, c, _, _ = x.size() # Batch size, channels, height, width
        y_c = self.avg_pool(x).view(b, c) # Global average pooling -> [B, C, 1, 1] -> [B, C]
        y_c = self.c(y_c) #channel attention layers
        y_c = y_c.view(b,c, 1, 1) #y_c represents the learned importance of each channel.
        # print y_c.size()
        # print x.size()

        y_s = self.s(x) # Spatial convolution --> single-channel attention map of shape [B, 1, H, W].

        # print y_s.size()
        y = y_c + y_s  # Combine channel and spatial attention
        y = self.sigmoid(y) # Normalize to [0, 1]
        x = x + x * y # Scale input feature map by attention weights
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, first_block=False):
        super(BasicBlock, self).__init__()
        self.bn0 = nn.InstanceNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.InstanceNorm2d(out_channel)
        self.conv2 = conv3x3(out_channel, out_channel)

        self.first_block = first_block
        self.stride = stride
        if downsample is not None:
            self.downsample = nn.Sequential(
                conv1x1(in_channel, out_channel * self.expansion, stride)
            )
        else:
            self.downsample = downsample

    def forward(self, x):
        identity = x

        if self.first_block:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
        else:
            out = self.bn0(x)
            out = self.relu(out)
            out = self.conv1(out)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        # out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, first_block=False):
        super(Bottleneck, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.first_block = first_block
        self.stride = stride
        self.conv4 = conv1x1(inplanes, planes * self.expansion)
        if downsample is not None:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion, stride),
                # nn.BatchNorm2d(out_channel),
            )
        else:
            self.downsample = downsample

    def forward(self, x):
        identity = x

        if self.first_block:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            # out = self.bn3(out)
        else:
            out = self.bn0(x)
            out = self.relu(out)
            out = self.conv1(out)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            # out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.first_block:
            identity = self.conv4(x)

        out += identity
        # out = self.relu(out)

        return out


class Res34(nn.Module):
    r""" It extracts features through residual blocks and 
    integrates spatial-channel attention mechanisms 
    at different stages to focus on important features """
    
    def __init__(self, layer_num=32, num_classes=4):
        super(Res34, self).__init__()
   
        self.inplanes = 64

        self.init_layer = nn.Sequential(
            nn.Conv2d(layer_num, 64, kernel_size=7, stride=2, padding=3,
                      bias=False),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block1 = nn.Sequential(
            BasicBlock(in_channel=64, out_channel=64, first_block=True),
            BasicBlock(in_channel=64, out_channel=64),
            # BasicBlock(in_channel=64, out_channel=64),
        )

        self.attention1 = AttentionLayer(channel=64, rank=1)

        # self.block2 = nn.Sequential(
        #     BasicBlock(in_channel=64, out_channel=128, stride=2, downsample=True),
        #     BasicBlock(in_channel=128, out_channel=128),
        #     BasicBlock(in_channel=128, out_channel=128),
        #     # BasicBlock(in_channel=128, out_channel=128),
        # )

        # self.attention2 = AttentionLayer(channel=128, rank=2)

        # self.block3 = nn.Sequential(
        #     BasicBlock(in_channel=128, out_channel=256, stride=2, downsample=True),
        #     BasicBlock(in_channel=256, out_channel=256),
        #     BasicBlock(in_channel=256, out_channel=256),
        #     BasicBlock(in_channel=256, out_channel=256),
        #     BasicBlock(in_channel=256, out_channel=256),
        #     BasicBlock(in_channel=256, out_channel=256),
        # )

        # self.attention3 = AttentionLayer(channel=256, rank=3)

        # self.block4 = nn.Sequential(
        #     BasicBlock(in_channel=128, out_channel=256, stride=2, downsample=True),
        #     BasicBlock(in_channel=256, out_channel=256)
        #     # BasicBlock(in_channel=512, out_channel=512),
        # )

        # self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        # self.fc2 = nn.Linear(256, num_classes)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, **kwargs):
        x = kwargs["x_mri"] #[B, 32, H, W]
        x = self.init_layer(x) # [B, 64, H/4, W/4]
        x = self.block1(x) # Residual block processes features.
        x = self.attention1(x) # attention layer highlights important spatial regions and channels.
                                #Output shape: [B, 64, H/4, W/4]

        # x = self.block2(x) 
        # x = self.attention2(x)
                            #[B, 128, H/8, W/8]
        # x = self.block3(x)
        # x = self.attention3(x)
                            #[B, 256, H/16, W/16]
        # x = self.block4(x) #[B, 512, H/32, W/32].
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # x = self.fc(x)
        # x = self.sigmoid(x)

        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = models.vgg19().to(device)
    net = Res34(layer_num= 32).to(device)