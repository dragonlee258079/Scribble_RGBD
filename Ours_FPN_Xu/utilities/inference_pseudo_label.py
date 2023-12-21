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

import numpy as np

import cv2


def forward_pass_without_loss(model, data, device):
    # read data
    image, depth = data['image'].to(device), data['depth'].to(device)

    inputs = NestedTensor(image, depth)

    # forward pass
    start = time.time()
    sal = model(image)
    end = time.time()
    time_elapse = end - start

    outputs = {}
    # outputs["disp"] = disp
    outputs["sal"] = sal

    return outputs, time_elapse


@torch.no_grad()
def inference(net, cfg, args, model_dir):
    net.eval()
    print("loading model from {}".format(model_dir))
    checkpoint = torch.load(model_dir)
    state_dict = checkpoint['state_dict']
    net.load_state_dict(state_dict)

    # get device
    device = torch.device(args.device)

    data_loaders = build_data_loader(args, cfg, mode="test")
    test_datasets = args.test_list
    # pseudo_label_dir = '/data1/lilong/Weakly_RGBD/Datatset/Train/Pseudo'
    # save_dir = '/data1/lilong/Weakly_RGBD/Datatset/Train/Pseudo/densecrf/pred'
    root_dir = '/data1/lilong/Weakly_RGBD/Prediction/Train'
    save_dir = os.path.join(root_dir, 'img_99')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i in range(len(test_datasets)):
        print(test_datasets[i])

        # model_name = __file__.split('/')[-3]
        # save_dir = os.path.join(pseudo_label_dir, model_name)
        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)

        tbar = tqdm(data_loaders[i])

        for idx, data in enumerate(tbar):
            # forward pass
            outputs, time_elapse = forward_pass_without_loss(net, data, device)

            image_h, image_w = int(data["size"][0]), int(data["size"][1])

            # save output
            pred_sal = outputs['sal']
            pred_sal = F.interpolate(pred_sal, (image_w, image_h), mode='bilinear', align_corners=False)
            pred_sal = F.sigmoid(pred_sal)
            pred_sal = pred_sal > 0.99
            pred_sal = pred_sal.squeeze().cpu().data.numpy().astype(np.int)

            # np.save(os.path.join(save_dir, data["image_name"][0][:-4]+'.npy'), pred_sal)
            cv2.imwrite(os.path.join(save_dir, data["image_name"][0][:-4] + '.png'), pred_sal*255)
    return
