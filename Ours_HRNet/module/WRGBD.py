import torch
import torch.nn as nn

from module.Baseline_DS_RGB import base_ds as base_ds_rgb
from module.Baseline_DS_DEP import base_ds as base_ds_dep
from module.Feature_Fusion import DS_Fusion


class WRGBD(nn.Module):
    def __init__(self, cfg):
        super(WRGBD, self).__init__()
        self.rgb_branch = base_ds_rgb(cfg)
        self.dep_branch = base_ds_dep(cfg)
        self.fusion = DS_Fusion(cfg)

        self.rgb_classifer = nn.Conv2d(48, 1, 1, 1, 0)
        self.dep_classifer = nn.Conv2d(48, 1, 1, 1, 0)
        self.rgbd_classifer = nn.Conv2d(48, 1, 1, 1, 0)

    def forward(self, rgb, dep):
        rgb_fea = self.rgb_branch(rgb)
        dep_fea = self.dep_branch(dep)
        rgbd_fea = self.fusion([rgb_fea.detach(), dep_fea.detach()])

        rgb_sal = self.rgb_classifer(rgb_fea)
        dep_sal = self.dep_classifer(dep_fea)
        rgbd_sal = self.rgbd_classifer(rgbd_fea)

        return rgb_sal, dep_sal, rgbd_sal
