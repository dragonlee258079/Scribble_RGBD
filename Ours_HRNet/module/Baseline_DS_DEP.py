import torch.nn as nn

from module.HighResolutionNet import get_seg_model
from module.Transform_DEP import Trans


class base_ds(nn.Module):
    def __init__(self, cfg):
        super(base_ds, self).__init__()
        self.backbone = get_seg_model(cfg)
        self.trans = Trans(cfg)

    def forward(self, x):
        x = self.backbone(x)
        x = self.trans(x)

        return x
