from torch import nn
from module.ResNet import resnet50


def Backbone_ResNet50():
    net = resnet50(pretrained=True)
    div_2 = nn.Sequential(*list(net.children())[:3])
    div_4 = nn.Sequential(*list(net.children())[3:5])
    div_8 = net.layer2
    div_16 = net.layer3
    div_32 = net.layer4

    return div_2, div_4, div_8, div_16, div_32


class ResNet_Feature_Extractor(nn.Module):
    def __init__(self):
        super(ResNet_Feature_Extractor, self).__init__()
        (
            self.encoder2,
            self.encoder4,
            self.encoder8,
            self.encoder16,
            self.encoder32,
        ) = Backbone_ResNet50()

    def forward(self, x, last=False):
        x_en2 = self.encoder2(x)
        x_en4 = self.encoder4(x_en2)
        x_en8 = self.encoder8(x_en4)
        x_en16 = self.encoder16(x_en8)
        x_en32 = self.encoder32(x_en16)

        if last:
            return x_en32

        out = [x_en2, x_en4, x_en8, x_en16, x_en32]
        return out


class Convertlayer(nn.Module):
    def __init__(self, cfg):
        super(Convertlayer, self).__init__()
        res_channels = cfg.NETWORK.ResNet_CHANNELS
        cvt_channels = cfg.NETWORK.Convert_CHANNELS

        cvt = []
        for i in range(len(res_channels)):
            cvt.append(
                nn.Sequential(
                    nn.Conv2d(res_channels[i], cvt_channels[i], 1, 1, bias=False),
                    nn.ReLU(inplace=True),
                )
            )

        self.convert = nn.ModuleList(cvt)

    def forward(self, res_feas, last=False):
        if last:
            return self.convert[-1](res_feas)

        cvt_feas = []
        for i in range(len(res_feas)):
            cvt_feas.append(self.convert[i](res_feas[i]))
        return cvt_feas
