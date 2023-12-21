import torch
import torch.nn as nn
import torch.nn.functional as F

from module.Backbone import ResNet_Feature_Extractor, Convertlayer
from module.Decoder import fuse_decoder
from module.new_Transformer_dep import Trans


class Depth_Branch(nn.Module):
    def __init__(self, cfg):
        super(Depth_Branch, self).__init__()

        self.backbone = ResNet_Feature_Extractor()
        self.convert = Convertlayer(cfg)
        self.decode = fuse_decoder(cfg)

        unify_channels = cfg.NETWORK.UNIFIY_CHANNELS
        self.trans = Trans(unify_channels)

        # self.last_layer = nn.Sequential(
        #     nn.BatchNorm2d(66),
        #     nn.Conv2d(66, 1, 3, 1, 1),
        # )

    def forward(self, x):
        x = self.backbone(x)
        x = self.convert(x)
        x = self.decode(x)
        x = self.trans(x)
        # sal = self.last_layer(x)

        return x
