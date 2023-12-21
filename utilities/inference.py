#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import time
import os

import torch
from tqdm import tqdm

from dataset import build_data_loader

import torch.nn.functional as F


from torchvision import transforms


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

    outputs["rgb_sal"] = rgb_sal
    outputs["dep_sal"] = dep_sal
    outputs["rgbd_sal"] = rgbd_sal

    return outputs, time_elapse


@torch.no_grad()
def std_epislon(a, epislon=1e-8):
    return torch.sqrt(
        torch.sum((a - torch.mean(a, dim=1, keepdim=True) + epislon) ** 2, dim=1, keepdim=True) / (a.shape[1] - 1)
    )


@torch.no_grad()
def get_edge(sal_prob):
    max_pool = F.max_pool2d(sal_prob, 9, 1, 4)
    neg_max_pool = F.max_pool2d(-sal_prob, 9, 1, 4)
    edge_mask = max_pool + neg_max_pool

    return edge_mask


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
    for i in range(len(test_datasets)):
        print(test_datasets[i])
        save_dataset_dir = os.path.join(args.save_dir, test_datasets[i])
        if not os.path.exists(save_dataset_dir):
            os.mkdir(save_dataset_dir)

        model_name = __file__.split('/')[-3]
        save_dir = os.path.join(save_dataset_dir, model_dir.split('/')[-1].split('.pth.tar')[0])
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        tbar = tqdm(data_loaders[i])

        for idx, data in enumerate(tbar):
            # forward pass
            outputs, time_elapse = forward_pass_without_loss(net, data, device)

            # save output
            pred_sal = F.sigmoid(outputs['rgbd_sal'])

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
