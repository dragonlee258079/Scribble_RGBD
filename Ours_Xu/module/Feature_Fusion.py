import torch
import torch.nn as nn
import torch.nn.functional as F


class cat_fuse(nn.Module):
    def __init__(self, channel):
        super(cat_fuse, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channel*2, channel, 3, 1, 1, bias=True),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb_fea, dep_fea):
        rgbd = self.conv(torch.cat([rgb_fea, dep_fea], dim=1))
        return rgbd


class FFN_2d(nn.Module):
    def __init__(self, in_channels):
        super(FFN_2d, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, 1, 1, 0, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels//2, in_channels, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class DS_Fusion(nn.Module):
    def __init__(self, in_channels):
        super(DS_Fusion, self).__init__()

        self.la_query_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 1, 1, 0, bias=True),
            nn.BatchNorm2d(in_channels//2)
        )
        self.la_query_2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 1, 1, 0, bias=True),
            nn.BatchNorm2d(in_channels//2)
        )
        # self.la_query_3 = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels//3, 1, 1, 0, bias=True),
        #     nn.BatchNorm2d(in_channels//3)
        # )

        self.la_key_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 1, 1, 0, bias=True),
            nn.BatchNorm2d(in_channels//2)
        )

        self.la_key_2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 1, 1, 0, bias=True),
            nn.BatchNorm2d(in_channels//2)
        )

        self.la_value_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 1, 1, 0, bias=True),
            nn.BatchNorm2d(in_channels//2)
        )

        self.la_value_2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 1, 1, 0, bias=True),
            nn.BatchNorm2d(in_channels//2)
        )

        # self.s_key = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=True),
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(inplace=True)
        # )
        #
        # self.s_query = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=True),
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(inplace=True)
        # )
        #
        # self.s_value = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=True),
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(inplace=True)
        # )

        self.cat_fuse = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels*2,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.BatchNorm2d(in_channels)
        )

        # self.emb = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels//2, 1, 1, 0, bias=True),
        #     nn.GELU(),
        #     nn.Conv2d(in_channels//2, in_channels, 1, 1, 0, bias=True),
        #     nn.Dropout(0.1)
        # )

        self.ffn = FFN_2d(in_channels)

        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 1, 1, 0, bias=True),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, in_channels, 1, 1, 0, bias=True)
        )

        self.drop = nn.Dropout(0.1)
        self.norm = nn.BatchNorm2d(in_channels)

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x[0].shape
        num_lay = len(x)

        for k in range(1):
            for i in range(num_lay):
                x[i] = self.relu(self.bn(x[i] + self.res_block(x[i])))

            key_1 = []
            key_2 = []
            for i in range(len(x)):
                la_k_1 = self.la_key_1(x[i])
                la_k_2 = self.la_key_2(x[i])
                # la_k_split = la_k.split(c//3, dim=1)
                # for a in la_k_split:
                key_1.append(la_k_1.unsqueeze(dim=1))
                key_2.append(la_k_2.unsqueeze(dim=1))
            key_1 = torch.cat(key_1, dim=1).view(b, num_lay, c//2, -1)
            key_1 = key_1.permute(0, 3, 2, 1).contiguous()
            key_2 = torch.cat(key_2, dim=1).view(b, num_lay, c//2, -1)
            key_2 = key_2.permute(0, 3, 2, 1).contiguous()

            value_1 = []
            value_2 = []
            for i in range(len(x)):
                la_v_1 = self.la_value_1(x[i])
                la_v_2 = self.la_value_2(x[i])
                # la_v_split = la_v.split(c//3, dim=1)
                # for a in la_v_split:
                value_1.append(la_v_1.unsqueeze(dim=1))
                value_2.append(la_v_2.unsqueeze(dim=1))
            value_1 = torch.cat(value_1, dim=1).view(b, num_lay, c//2, -1)
            value_1 = value_1.permute(0, 3, 1, 2).contiguous()
            value_2 = torch.cat(value_2, dim=1).view(b, num_lay, c//2, -1)
            value_2 = value_2.permute(0, 3, 1, 2).contiguous()

            if k == 0:
                la_res = torch.cat(x, 1)
                la_res = self.cat_fuse(la_res)
                la_res = la_res + self.drop(la_res)
                la_res = self.norm(la_res)
                la_res = la_res + self.ffn(la_res)

            # for k in range(1):

            query_1 = self.la_query_1(la_res).view(b, 1, c//2, -1)
            query_1 = query_1.permute(0, 3, 1, 2).contiguous()

            la_att = torch.matmul(query_1, key_1) * ((c//2) ** -.5)
            la_att = F.softmax(la_att, dim=-1)

            ref_1 = torch.matmul(la_att, value_1)
            ref_1 = ref_1.permute(0, 2, 3, 1).contiguous().squeeze(dim=1)
            ref_1 = ref_1.view(b, c//2, h, w)

            query_2 = self.la_query_2(la_res).view(b, 1, c//2, -1)
            query_2 = query_2.permute(0, 3, 1, 2).contiguous()

            la_att = torch.matmul(query_2, key_2) * ((c//2) ** -.5)
            la_att = F.softmax(la_att, dim=-1)

            ref_2 = torch.matmul(la_att, value_2)
            ref_2 = ref_2.permute(0, 2, 3, 1).contiguous().squeeze(dim=1)
            ref_2 = ref_2.view(b, c//2, h, w)

            la_res = torch.cat([ref_1, ref_2], dim=1)
            la_res = self.cat_conv(la_res)

            la_res = la_res + self.drop(la_res)
            la_res = self.norm(la_res)
            la_res = la_res + self.ffn(la_res)

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
        # sim_map = (c ** -.5) * sim_map
        # sim_map = F.softmax(sim_map, dim=-1)
        #
        # res = torch.matmul(sim_map, s_value)
        # res = res.permute(0, 2, 1).contiguous()
        # res = res.view(b, c, h, w)
        # res = res + la_res

        return la_res
