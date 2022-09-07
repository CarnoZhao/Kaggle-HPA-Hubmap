### inference https://github.com/FrancescoSaverioZuppichini/SegFormer/blob/main/README.ipynb
import torch
# from einops import rearrange
from torch import nn
import torch.nn.functional as F

from torchvision.ops import StochasticDepth
from typing import List
from typing import Iterable

#######################################################################################################
## https://github.com/lucidrains/segformer-pytorch/blob/main/segformer_pytorch/segformer_pytorch.py
# https://github.com/UAws/CV-3315-Is-All-You-Need
class MixUpSample(nn.Module):
    def __init__( self, scale_factor=2):
        super().__init__()
        self.mixing = nn.Parameter(torch.tensor(0.5))
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.mixing *F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False) \
            + (1-self.mixing )*F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        return x


class SegformerDecoder(nn.Module):
    def __init__(
            self,
            encoder_dim = [32, 64, 160, 256],
            decoder_dim = 256,
    ):
        super().__init__()
        self.mixing = nn.Parameter(torch.FloatTensor([0.5,0.5,0.5,0.5]))
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, decoder_dim, 1, padding= 0,  bias=False), #follow mmseg to use conv-bn-relu
                nn.BatchNorm2d(decoder_dim),
                nn.ReLU(inplace=True),
                MixUpSample(2**i) if i!=0 else nn.Identity(),
            ) for i, dim in enumerate(encoder_dim)])

        self.fuse = nn.Sequential(
            nn.Conv2d(len(encoder_dim) * decoder_dim, decoder_dim, 1, padding=0, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            # nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1, bias=False),
            # nn.BatchNorm2d(decoder_dim),
            # nn.ReLU(inplace=True),
        )

    def forward(self, feature):

        out = []
        for i,f in enumerate(feature):
            f = self.mlp[i](f)
            out.append(f)

        x = self.fuse(torch.cat(out, dim = 1))
        return x, out