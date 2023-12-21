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


def forward_pass_without_loss(model, data, device):
    # read data
    image, depth = data['image'].to(device), data['depth'].to(device)
    depth = depth.repeat(1, 3, 1, 1)

    inputs = NestedTensor(image, depth)

    # forward pass
    start = time.time()
    _, _, sal = model(image, depth)
    end = time.time()
    time_elapse = end - start

    outputs = {}
    # outputs["disp"] = disp
    outputs["sal"] = sal

    return outputs, time_elapse


@torch.no_grad()
def inference(net, cfg, args, model_dir):
    # net.eval()
    # print("loading model from {}".format(model_dir))
    # checkpoint = torch.load(model_dir)
    # state_dict = checkpoint['state_dict']
    # net.load_state_dict(state_dict)

    # get device
    device = torch.device(args.device)

    data_loaders = build_data_loader(args, cfg, mode="test")
    test_datasets = args.test_step_list
    for i in range(len(test_datasets)):
        print(test_datasets[i])
        save_dataset_dir = os.path.join(args.save_dir, test_datasets[i])
        if not os.path.exists(save_dataset_dir):
            os.mkdir(save_dataset_dir)

        model_name = "_".join(model_dir.split('/')[-2:])
        save_dir = os.path.join(save_dataset_dir, model_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        tbar = tqdm(data_loaders[i])

        for idx, data in enumerate(tbar):
            # forward pass
            outputs, time_elapse = forward_pass_without_loss(net, data, device)

            # save output
            pred_sal = F.sigmoid(outputs['sal'])
            pred_sal = pred_sal.data.cpu().squeeze(0)

            image_h, image_w = int(data["size"][0]), int(data["size"][1])

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Scale((image_w, image_h))
            ])

            pred_sal = transform(pred_sal)

            save_img_dir = os.path.join(save_dir, data["image_name"][0][:-4]+'.png')
            pred_sal.save(save_img_dir)

    return
