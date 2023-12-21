import torch
import torch.nn as nn
import torch.nn.functional as F


class decoder_module(nn.Module):
    def __init__(self, in_channels, out_channels, fuse=True):
        super(decoder_module, self).__init__()
        self.convert = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        in_channels = out_channels * 2 if fuse else out_channels
        self.decoding = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, enc_fea, dec_fea=None):
        enc_fea = self.convert(enc_fea)
        if dec_fea is not None:
            if dec_fea.size(2) != enc_fea.size(2):
                dec_fea = F.interpolate(dec_fea, scale_factor=2, mode='bilinear')
            enc_fea = torch.cat([enc_fea, dec_fea], dim=1)
        return self.decoding(enc_fea)


class fuse_decoder(nn.Module):
    def __init__(self, cfg):
        super(fuse_decoder, self).__init__()
        in_channels = cfg.NETWORK.Convert_CHANNELS
        out_channel = cfg.NETWORK.UNIFIY_CHANNELS

        self.dec_5 = decoder_module(in_channels[4], out_channel, False)
        self.dec_4 = decoder_module(in_channels[3], out_channel)
        self.dec_3 = decoder_module(in_channels[2], out_channel)
        self.dec_2 = decoder_module(in_channels[1], out_channel)
        self.dec_1 = decoder_module(in_channels[0], out_channel)

    def forward(self, enc_feas):
        dec_fea_5 = self.dec_5(enc_feas[4])
        dec_fea_4 = self.dec_4(enc_feas[3], dec_fea_5)
        dec_fea_3 = self.dec_3(enc_feas[2], dec_fea_4)
        dec_fea_2 = self.dec_2(enc_feas[1], dec_fea_3)
        dec_fea_1 = self.dec_1(enc_feas[0], dec_fea_2)

        return [dec_fea_1, dec_fea_2, dec_fea_3, dec_fea_4, dec_fea_5]
