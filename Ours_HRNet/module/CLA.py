import torch
import torch.nn as nn
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01


class CLA(nn.Module):
    def __init__(self, cfg):
        super(CLA, self).__init__()
        in_channels = cfg.MODEL.EXTRA.STAGE4.NUM_CHANNELS
        self.adap_layers = []
        for i in range(len(in_channels)):
            adap_layer_i = 'adap_layer_{}'.format(i)
            module = nn.Sequential(
                nn.Conv2d(in_channels[i], in_channels[0], 1, 1, 0, bias=False),
                BatchNorm2d(in_channels[0], momentum=BN_MOMENTUM)
            )
            self.adap_layers.append(adap_layer_i)
            self.add_module(adap_layer_i, module)

        self.la_key = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0], 3, 1, 1, bias=True),
            BatchNorm2d(in_channels[0], momentum=BN_MOMENTUM)
        )

        self.la_querys = []
        for i in range(1, len(in_channels)):
            la_query_i = 'la_query_{}'.format(i)
            module = nn.Sequential(
                nn.Conv2d(in_channels[0], in_channels[0], 3, 1, 1, bias=True),
                BatchNorm2d(in_channels[0], momentum=BN_MOMENTUM)
            )
            self.la_querys.append(la_query_i)
            self.add_module(la_query_i, module)

        self.la_values = []
        for i in range(1, len(in_channels)):
            la_value_i = 'la_values_{}'.format(i)
            module = nn.Sequential(
                nn.Conv2d(in_channels[0], in_channels[0], 3, 1, 1, bias=True),
                BatchNorm2d(in_channels[0], momentum=BN_MOMENTUM)
            )
            self.la_values.append(la_value_i)
            self.add_module(la_value_i, module)

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

    def forward(self, x):
        b, c, h, w = x[0].shape
        num_lay = len(self.la_querys)

        for i in range(len(self.adap_layers)):
            x[i] = getattr(self, self.adap_layers[i])(x[i])
            if i != 0:
                x[i] = F.interpolate(x[i], size=(h, w), mode="nearest")

        key = self.la_key(x[0]).view(b, 1, c, -1)
        key = key.permute(0, 3, 1, 2).contiguous()
        query = []
        for i in range(len(self.la_querys)):
            la_q = getattr(self, self.la_querys[i])(x[i+1])
            query.append(la_q.unsqueeze(dim=1))
        query = torch.cat(query, dim=1).view(b, num_lay, c, -1)
        query = query.permute(0, 3, 2, 1).contiguous()
        value = []
        for i in range(len(self.la_values)):
            la_v = getattr(self, self.la_values[i])(x[i+1])
            value.append(la_v.unsqueeze(dim=1))
        value = torch.cat(value, dim=1).view(b, num_lay, c, -1)
        value = value.permute(0, 3, 1, 2).contiguous()

        la_att = torch.matmul(key, query)
        la_att = F.softmax(la_att, dim=-1)

        la_res = torch.matmul(la_att, value)
        la_res = la_res.permute(0, 2, 3, 1).contiguous().squeeze(dim=1)
        la_res = la_res.view(b, c, h, w)

        la_res = x[0] + la_res
        # la_res = x[0]

        g_x = self.g(la_res).view(b, c, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(la_res).view(b, c, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(la_res).view(b, c, -1)

        f = torch.matmul(theta_x, phi_x)
        f_dic_C = f / (h*w)

        res = torch.matmul(f_dic_C, g_x)
        res = res.permute(0, 2, 1).contiguous()
        res = res.view(b, c, h, w)

        res = res + la_res

        return res
