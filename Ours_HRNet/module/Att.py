import torch
import torch.nn as nn
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01


class cross_adjcent_att(nn.Module):
    def __init__(self, cfg):
        super(cross_adjcent_att, self).__init__()
        # self.k = nn.Sequential(
        #     nn.Conv2d(64, 64, 1, 1, 0, bias=True),
        #     BatchNorm2d(64, momentum=BN_MOMENTUM)
        # )
        # self.r = nn.Sequential(
        #     nn.Conv2d(64, 64, 1, 1, 0, bias=True),
        #     BatchNorm2d(64, momentum=BN_MOMENTUM)
        # )
        self.unfold = nn.Unfold(kernel_size=(3, 3), padding=1, stride=1)

    def _unfold(self, fea):
        b, c, h, w = fea.shape
        uf_fea = self.unfold(fea)
        uf_fea = uf_fea.transpose(2, 1).contiguous()
        uf_fea = uf_fea.view(b, h, w, c, -1)
        index = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8])
        index = index.cuda(device=uf_fea.device)
        uf_fea = torch.index_select(uf_fea, dim=-1, index=index)

        return uf_fea

    def forward(self, x_stem, x):
        # x_k = self.k(x_stem)
        # x_r = self.r(x_stem)

        uf_x_stem = self._unfold(x_stem)

        x_stem = x_stem.permute(0, 2, 3, 1)
        x_stem = x_stem.unsqueeze(dim=-2)

        affinity = torch.matmul(x_stem, uf_x_stem)
        affinity = F.softmax(affinity, dim=-1)

        uf_x = self._unfold(x)
        uf_x = uf_x.transpose(4, 3).contiguous()

        aatt_x = torch.matmul(affinity, uf_x).squeeze(dim=-2)
        aatt_x = aatt_x.permute(0, 3, 1, 2)

        return aatt_x
