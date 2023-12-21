
#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import time
import os

import torch
from tqdm import tqdm

from utilities.misc import NestedTensor, save_and_clear

from dataset import build_data_loader

import torch.nn.functional as F

from PIL import Image

from torchvision import transforms

import cv2
import numpy as np


def forward_pass_without_loss(model, data, device):
    # read data
    image, depth = data['image'].to(device), data['depth'].to(device)

    depth = depth.repeat(1, 3, 1, 1)

    # forward pass
    start = time.time()
    rgb_sal, dep_sal, rgbd_sal = model(image, depth)
    end = time.time()
    time_elapse = end - start

    outputs = {}
    # outputs["disp"] = disp
    outputs["rgb_sal"] = rgb_sal
    outputs["dep_sal"] = dep_sal
    outputs["rgbd_sal"] = rgbd_sal

    return outputs, time_elapse


@torch.no_grad()
def inference(net, cfg, args, model_dir):
    net.eval()
    print("loading model from {}".format(model_dir))
    checkpoint = torch.load(model_dir)
    state_dict = checkpoint['state_dict']
    net.load_state_dict(state_dict)

    gt_root_dir = '/disk2/lilong/WRGBD/Dataset/Train/full_gt'
    percent = 0.9
    ratio_max = 0
    epo = model_dir.split('model_epo')[-1][:2]

    # get device
    device = torch.device(args.device)

    data_loaders = build_data_loader(args, cfg, mode="test")
    test_datasets = args.test_list
    for i in range(len(test_datasets)):
        print(test_datasets[i])
        save_dataset_dir = os.path.join(args.save_dir, test_datasets[i])
        if not os.path.exists(save_dataset_dir):
            os.mkdir(save_dataset_dir)

        model_name = __file__.split('/')[-3]
        model_num = model_dir.split('_')[-1]
        save_dir = os.path.join(save_dataset_dir, "{}_{}".format(model_name, model_num))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        tbar = tqdm(data_loaders[i])

        for idx, data in enumerate(tbar):
            # print(data['image_name'][0])
            # if data['image_name'][0] != '0070.jpg':
            #     continue
            # forward pass
            # if data['image_name'][0] != 'NJU2K_001784_left.jpg':
            #     continue

            outputs, time_elapse = forward_pass_without_loss(net, data, device)

            gt_dir = os.path.join(gt_root_dir, data['image_name'][0][:-4]+'.png')
            gt = cv2.imread(gt_dir, 0).astype(np.float64)
            gt /= 255.

            # image_h, image_w = int(data["size"][0]), int(data["size"][1])
            image_w, image_h = gt.shape

            rgb_sal = F.sigmoid(outputs['rgb_sal'])
            rgb_sal = rgb_sal.data.cpu().squeeze().numpy()
            rgb_sal = cv2.resize(rgb_sal, (image_h, image_w), interpolation=cv2.INTER_LINEAR)
            # rgb_sal[rgb_sal > 0.5] = 1.
            # rgb_sal[rgb_sal <= 0.5] = 0.

            dep_sal = F.sigmoid(outputs['dep_sal'])
            dep_sal = dep_sal.data.cpu().squeeze().numpy()
            dep_sal = cv2.resize(dep_sal, (image_h, image_w), interpolation=cv2.INTER_LINEAR)
            # dep_sal[dep_sal > 0.5] = 1.
            # dep_sal[dep_sal <= 0.5] = 0.

            rgb_fg_thresh = np.max(rgb_sal) * percent
            rgb_fg_thresh = rgb_fg_thresh.clip(min=0.5)
            rgb_bg_thresh = 1 - rgb_fg_thresh

            dep_fg_thresh = np.max(dep_sal) * percent
            dep_fg_thresh = dep_fg_thresh.clip(min=0.5)
            dep_bg_thresh = 1 - dep_fg_thresh

            rgb_con_fg = rgb_sal >= rgb_fg_thresh
            rgb_con_bg = rgb_sal <= rgb_bg_thresh

            dep_con_fg = dep_sal >= dep_fg_thresh
            dep_con_bg = dep_sal <= dep_bg_thresh

            rgb_fn = rgb_con_bg & (gt == 1.)
            rgb_tp = rgb_con_fg & (gt == 1.)
            dep_tp = dep_con_fg & (gt == 1.)
            dep_fn = dep_con_bg & (gt == 1.)
            # diff_fg = (rgb_con_fg != dep_con_fg) & (rgb_con_fg | dep_con_fg)
            # diff_bg = (rgb_con_bg != dep_con_bg) & (rgb_con_bg | dep_con_bg)

            # diff_fg[gt == 0.] = False
            # diff_bg[gt == 1.] = False

            # diff = diff_fg | diff_bg
            # ratio = np.sum(diff_fg) / (image_h * image_w)

            rgb_cali = rgb_fn & dep_tp
            dep_cali = dep_fn & rgb_tp
            rgb_ratio = np.sum(rgb_cali) / (image_h * image_w)
            dep_ratio = np.sum(dep_cali) / (image_h * image_w)

            ratio = rgb_ratio + 0.1 * dep_ratio

            # if ratio > ratio_max:
            if rgb_ratio > 0.01 and dep_ratio > 0.01:
                ratio_max = ratio
                print("ratio:{} rgb_ratio:{} dep_ratio:{} img_name:{}".format(round(ratio, 5), round(rgb_ratio, 5),
                                                                              round(dep_ratio, 5), data['image_name'][0]))
                img_name = data['image_name'][0]
                cv2.imwrite(os.path.join('dual_predictions_selection',
                                         epo+'_'+img_name[:-4]+'_rgb_fg_{}.png'.format(round(rgb_ratio, 5))), rgb_con_fg*255)
                cv2.imwrite(os.path.join('dual_predictions_selection',
                                         epo+'_'+img_name[:-4]+'_dep_fg_{}.png'.format(round(dep_ratio, 5))), dep_con_fg*255)
                cv2.imwrite(os.path.join('dual_predictions_selection',
                                         epo+'_'+img_name[:-4]+'_gt_{}.png'.format(round(ratio, 5))), gt*255)

    return
