import torch
import torch.nn as nn
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01


class Trans(nn.Module):
    def __init__(self, cfg):
        super(Trans, self).__init__()
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

        self.la_key_1 = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0]//2, 1, 1, 0, bias=True),
            BatchNorm2d(in_channels[0]//2, momentum=BN_MOMENTUM)
        )
        self.la_key_2 = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0]//2, 1, 1, 0, bias=True),
            BatchNorm2d(in_channels[0]//2, momentum=BN_MOMENTUM)
        )

        self.la_query = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0], 1, 1, 0, bias=True),
            BatchNorm2d(in_channels[0], momentum=BN_MOMENTUM)
        )

        self.la_value = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0], 1, 1, 0, bias=True),
            BatchNorm2d(in_channels[0], momentum=BN_MOMENTUM)
        )

        # self.s_key = nn.Sequential(
        #     nn.Conv2d(in_channels[0], in_channels[0], 1, 1, 0, bias=True),
        #     BatchNorm2d(in_channels[0], momentum=BN_MOMENTUM),
        #     nn.ReLU(inplace=True)
        # )
        #
        # self.s_query = nn.Sequential(
        #     nn.Conv2d(in_channels[0], in_channels[0], 1, 1, 0, bias=True),
        #     BatchNorm2d(in_channels[0], momentum=BN_MOMENTUM),
        #     nn.ReLU(inplace=True)
        # )
        #
        # self.s_value = nn.Sequential(
        #     nn.Conv2d(in_channels[0], in_channels[0], 1, 1, 0, bias=True),
        #     BatchNorm2d(in_channels[0], momentum=BN_MOMENTUM),
        #     nn.ReLU(inplace=True)
        # )

        self.cat_fuse = nn.Sequential(
            nn.Conv2d(
                in_channels=48*4,
                out_channels=48,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(48, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        self.emb = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0]//2, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels[0]//2, in_channels[0], 1, 1, 0, bias=True),
            nn.Dropout(0.1)
        )

        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0]//2, 1, 1, 0, bias=True),
            nn.BatchNorm2d(24, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels[0]//2, in_channels[0], 1, 1, 0, bias=True)
        )

        self.bn = BatchNorm2d(48, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x[0].shape
        num_lay = len(x)

        x_up = []
        for i in range(len(self.adap_layers)):
            x[i] = getattr(self, self.adap_layers[i])(x[i])
            if i != 0:
                x_up.append(F.interpolate(x[i], size=(h, w), mode="bilinear", align_corners=False))
            else:
                x_up.append(x[i])

        for k in range(3):
            for i in range(num_lay):
                x_up[i] = self.relu(self.bn(x_up[i] + self.res_block(x_up[i])))

            query = []
            for i in range(len(x_up)):
                la_q = self.la_query(x_up[i])
                la_q_split = la_q.split(c//2, dim=1)
                for a in la_q_split:
                    query.append(a.unsqueeze(dim=1))
            query = torch.cat(query, dim=1).view(b, num_lay*2, c//2, -1)
            query = query.permute(0, 3, 2, 1).contiguous()
            value = []
            for i in range(len(x_up)):
                la_v = self.la_value(x_up[i])
                la_v_split = la_v.split(c//2, dim=1)
                for a in la_v_split:
                    value.append(a.unsqueeze(dim=1))
            value = torch.cat(value, dim=1).view(b, num_lay*2, c//2, -1)
            value = value.permute(0, 3, 1, 2).contiguous()

            if k == 0:
                la_res = torch.cat(x_up, 1)
                la_res = self.cat_fuse(la_res)
                la_res = self.emb(la_res)

        # for k in range(1):

            key_1 = self.la_key_1(la_res).view(b, 1, c//2, -1)
            key_1 = key_1.permute(0, 3, 1, 2).contiguous()

            la_att = torch.matmul(key_1, query)
            la_att = F.softmax(la_att, dim=-1)

            ref_1 = torch.matmul(la_att, value)
            ref_1 = ref_1.permute(0, 2, 3, 1).contiguous().squeeze(dim=1)
            ref_1 = ref_1.view(b, c//2, h, w)

            key_2 = self.la_key_2(la_res).view(b, 1, c//2, -1)
            key_2 = key_2.permute(0, 3, 1, 2).contiguous()

            la_att = torch.matmul(key_2, query)
            la_att = F.softmax(la_att, dim=-1)

            ref_2 = torch.matmul(la_att, value)
            ref_2 = ref_2.permute(0, 2, 3, 1).contiguous().squeeze(dim=1)
            ref_2 = ref_2.view(b, c//2, h, w)

            la_res = torch.cat([ref_1, ref_2], dim=1)

            la_res = self.emb(la_res)

        # s_query = self.s_query(la_res).view(b, c, -1)
        # s_query = s_query.permute(0, 2, 1).contiguous()
        #
        # s_key = [self.s_key(x_) for x_ in x]
        # for i in range(len(s_key)):
        #     if i != len(s_key) - 1:
        #         s_key[i] = F.interpolate(s_key[i], scale_factor=0.5, mode='bilinear', align_corners=False)
        #     s_key[i] = s_key[i].view(b, c, -1)
        # s_key = torch.cat(s_key, dim=2)
        #
        # s_value = [self.s_value(x_) for x_ in x]
        # for i in range(len(s_value)):
        #     if i != len(s_value) - 1:
        #         s_value[i] = F.interpolate(s_value[i], scale_factor=0.5, mode='bilinear', align_corners=False)
        #     s_value[i] = s_value[i].view(b, c, -1)
        # s_value = torch.cat(s_value, dim=2)
        # s_value = s_value.permute(0, 2, 1).contiguous()
        #
        # sim_map = torch.matmul(s_query, s_key)
        # sim_map = (48 ** -.5) * sim_map
        # sim_map = F.softmax(sim_map, dim=-1)
        #
        # res = torch.matmul(sim_map, s_value)
        # res = res.permute(0, 2, 1).contiguous()
        # res = res.view(b, c, h, w)
        # res = res + la_res

        return la_res
