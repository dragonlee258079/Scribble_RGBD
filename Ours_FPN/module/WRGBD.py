import torch
import torch.nn as nn
import torch.nn.functional as F

from module.RGB_Branch import RGB_Branch
from module.Depth_Branch import Depth_Branch
from module.Feature_Fusion import DS_Fusion


class WRGBD(nn.Module):
    def __init__(self, cfg):
        super(WRGBD, self).__init__()

        self.rgb_branch = RGB_Branch(cfg)
        self.dep_branch = Depth_Branch(cfg)

        self.fusion = DS_Fusion(66)

        self.rgb_pred = nn.Sequential(
            nn.BatchNorm2d(66),
            nn.Conv2d(66, 1, 3, 1, 1),
        )
        self.dep_pred = nn.Sequential(
            nn.BatchNorm2d(66),
            nn.Conv2d(66, 1, 3, 1, 1),
        )
        self.rgbd_pred = nn.Sequential(
            nn.BatchNorm2d(66),
            nn.Conv2d(66, 1, 3, 1, 1),
        )

    def fea_vis(self, fea):
        b, _, _, _ = fea.shape
        min_fea = torch.min(fea.view(b, -1), dim=1)[0].view(b, 1, 1, 1)
        max_fea = torch.max(fea.view(b, -1), dim=1)[0].view(b, 1, 1, 1)
        return (fea - min_fea) / (max_fea - min_fea)

    def forward(self, rgb, dep):
        rgb_fea = self.rgb_branch(rgb)
        dep_fea = self.dep_branch(dep)
        rgbd_fea = self.fusion([rgb_fea, dep_fea])

        # rgb_fea_mean = torch.mean(rgb_fea, dim=1, keepdim=True)
        # dep_fea_mean = torch.mean(dep_fea, dim=1, keepdim=True)
        # rgbd_fea_mean = torch.mean(rgbd_fea, dim=1, keepdim=True)
        #
        # rgb_fea_vis = self.fea_vis(rgb_fea_mean)
        # dep_fea_vis = self.fea_vis(dep_fea_mean)
        # rgbd_fea_vis = self.fea_vis(rgbd_fea_mean)

        rgb_sal = self.rgb_pred(rgb_fea)
        dep_sal = self.dep_pred(dep_fea)
        rgbd_sal = self.rgbd_pred(rgbd_fea)

        return rgb_sal, dep_sal, rgbd_sal
