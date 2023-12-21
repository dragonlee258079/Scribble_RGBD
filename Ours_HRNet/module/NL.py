import torch
import torch.nn as nn
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01


class NL(nn.Module):
    def __init__(self, cfg):
        super(NL, self).__init__()
        in_channels = cfg.MODEL.EXTRA.STAGE4.NUM_CHANNELS

        self.g = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0], 1, 1, 0, bias=True),
            BatchNorm2d(in_channels[0], momentum=BN_MOMENTUM)
        )
        self.theta = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0], 1, 1, 0, bias=True),
            BatchNorm2d(in_channels[0], momentum=BN_MOMENTUM)
        )
        self.phi = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0], 1, 1, 0, bias=True),
            BatchNorm2d(in_channels[0], momentum=BN_MOMENTUM)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(
                in_channels=sum(in_channels),
                out_channels=in_channels[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            BatchNorm2d(in_channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x[0].shape

        x1 = F.upsample(x[1], size=(h, w), mode='bilinear')
        x2 = F.upsample(x[2], size=(h, w), mode='bilinear')
        x3 = F.upsample(x[3], size=(h, w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], 1)

        x = self.fuse(x)

        g_x = self.g(x).view(b, c, -1)
        g_x = g_x.permute(0, 2, 1).contiguous()

        theta_x = self.theta(x).view(b, c, -1)
        theta_x = theta_x.permute(0, 2, 1).contiguous()

        phi_x = self.phi(x).view(b, c, -1)

        f = torch.matmul(theta_x, phi_x)
        f_dic_C = f / (h*w)

        res = torch.matmul(f_dic_C, g_x)
        res = res.permute(0, 2, 1).contiguous()
        res = res.view(b, c, h, w)

        res = res + x

        return res
