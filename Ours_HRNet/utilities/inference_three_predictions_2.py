import shutil
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

    rgb_root_dir = '/disk2/lilong/WRGBD/Dataset/Train/RGB/'
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
            # if data['image_name'][0] != 'ReDWeb-S_5491301016_a76ee9d03d_b.jpg':
            #     continue
            image_name = data['image_name'][0][:-4]

            shutil.copy(
                os.path.join(rgb_root_dir, "{}.jpg".format(image_name)),
                os.path.join(save_dir, "{}_0_img.jpg".format(image_name))
            )
            shutil.copy(
                os.path.join(gt_root_dir, "{}.png".format(image_name)),
                os.path.join(save_dir, "{}_1_gt.png".format(image_name))
            )

            outputs, time_elapse = forward_pass_without_loss(net, data, device)

            gt_dir = os.path.join(gt_root_dir, image_name+'.png')
            gt = cv2.imread(gt_dir, 0).astype(np.float64)
            gt /= 255.

            # image_h, image_w = int(data["size"][0]), int(data["size"][1])
            image_w, image_h = gt.shape

            rgb_sal = F.sigmoid(outputs['rgb_sal'])
            rgb_sal = rgb_sal.data.cpu().squeeze().numpy()
            rgb_sal = cv2.resize(rgb_sal, (image_h, image_w), interpolation=cv2.INTER_LINEAR)
            # rgb_sal[rgb_sal > 0.5] = 1.
            # rgb_sal[rgb_sal <= 0.5] = 0.
            cv2.imwrite(os.path.join(save_dir, "{}_2_rgb_pred.png".format(image_name)), rgb_sal*255)

            dep_sal = F.sigmoid(outputs['dep_sal'])
            dep_sal = dep_sal.data.cpu().squeeze().numpy()
            dep_sal = cv2.resize(dep_sal, (image_h, image_w), interpolation=cv2.INTER_LINEAR)
            # dep_sal[dep_sal > 0.5] = 1.
            # dep_sal[dep_sal <= 0.5] = 0.
            cv2.imwrite(os.path.join(save_dir, "{}_3_dep_pred.png".format(image_name)), dep_sal*255)

            rgb_fg_thresh = np.max(rgb_sal) * percent
            rgb_fg_thresh = rgb_fg_thresh.clip(min=0.5)
            rgb_bg_thresh = 1 - rgb_fg_thresh

            dep_fg_thresh = np.max(dep_sal) * percent
            dep_fg_thresh = dep_fg_thresh.clip(min=0.5)
            dep_bg_thresh = 1 - dep_fg_thresh

            rgb_con_fg = (rgb_sal >= rgb_fg_thresh).astype(np.float64)
            rgb_con_bg = (rgb_sal <= rgb_bg_thresh).astype(np.float64)
            rgb_con = rgb_con_fg + rgb_con_bg

            rgb_con = np.expand_dims(rgb_con, axis=-1).repeat(3, axis=-1)

            rgb_con[rgb_con_fg == 1.] = np.array([0, 0, 255])
            rgb_con[rgb_con_bg == 1.] = np.array([0, 255, 0])

            rgb_con_fg_3c = np.expand_dims(rgb_con_fg, axis=-1).repeat(3, axis=-1)
            rgb_con_fg_3c[rgb_con_fg == 1.] = np.array([0, 0, 255])
            rgb_con_bg_3c = np.expand_dims(rgb_con_bg, axis=-1).repeat(3, axis=-1)
            rgb_con_bg_3c[rgb_con_bg == 1.] = np.array([0, 255, 0])

            # cv2.imwrite('rgb_con.jpg', rgb_con)
            # cv2.imwrite('rgb_con_fg.jpg', rgb_con_fg_3c)
            # cv2.imwrite('rgb_con_bg.jpg', rgb_con_bg_3c)
            cv2.imwrite(os.path.join(save_dir, "{}_4_rgb_fg.png".format(image_name)), rgb_con_fg_3c)
            cv2.imwrite(os.path.join(save_dir, "{}_5_rgb_bg.png".format(image_name)), rgb_con_bg_3c)

            dep_con_fg = (dep_sal >= dep_fg_thresh).astype(np.float64)
            dep_con_bg = (dep_sal <= dep_bg_thresh).astype(np.float64)
            dep_con = dep_con_fg + dep_con_bg

            dep_con = np.expand_dims(dep_con, axis=-1).repeat(3, axis=-1)

            dep_con[dep_con_fg == 1.] = np.array([0, 0, 255])
            dep_con[dep_con_bg == 1.] = np.array([0, 255, 0])

            dep_con_fg_3c = np.expand_dims(dep_con_fg, axis=-1).repeat(3, axis=-1)
            dep_con_fg_3c[dep_con_fg == 1.] = np.array([0, 0, 255])
            dep_con_bg_3c = np.expand_dims(dep_con_bg, axis=-1).repeat(3, axis=-1)
            dep_con_bg_3c[dep_con_bg == 1.] = np.array([0, 255, 0])

            # cv2.imwrite('dep_con.jpg', dep_con)
            # cv2.imwrite('dep_con_fg.jpg', dep_con_fg_3c)
            # cv2.imwrite('dep_con_bg.jpg', dep_con_bg_3c)

            cv2.imwrite(os.path.join(save_dir, "{}_6_dep_fg.png".format(image_name)), dep_con_fg_3c)
            cv2.imwrite(os.path.join(save_dir, "{}_7_dep_bg.png".format(image_name)), dep_con_bg_3c)

            ctr_rgb_dep_fg_bg = (rgb_con_fg == 1.) & (dep_con_bg == 1.)
            ctr_rgb_dep_bg_fg = (rgb_con_bg == 1.) & (dep_con_fg == 1.)

            rgb_con[ctr_rgb_dep_fg_bg] = np.array([255, 0, 0])
            rgb_con[ctr_rgb_dep_bg_fg] = np.array([255, 0, 0])
            dep_con[ctr_rgb_dep_fg_bg] = np.array([255, 0, 0])
            dep_con[ctr_rgb_dep_bg_fg] = np.array([255, 0, 0])

            rgb_con_fg_3c[ctr_rgb_dep_fg_bg] = np.array([255, 0, 0])
            rgb_con_bg_3c[ctr_rgb_dep_bg_fg] = np.array([255, 0, 0])
            dep_con_fg_3c[ctr_rgb_dep_bg_fg] = np.array([255, 0, 0])
            dep_con_bg_3c[ctr_rgb_dep_fg_bg] = np.array([255, 0, 0])

            # cv2.imwrite('rgb_con_fg_3c_2.jpg', rgb_con_fg_3c)
            # cv2.imwrite('rgb_con_bg_3c_2.jpg', rgb_con_bg_3c)
            # cv2.imwrite('dep_con_fg_3c_2.jpg', dep_con_fg_3c)
            # cv2.imwrite('dep_con_bg_3c_2.jpg', dep_con_bg_3c)

            cv2.imwrite(os.path.join(save_dir, "{}_8_rgb_fg_2.png".format(image_name)), rgb_con_fg_3c)
            cv2.imwrite(os.path.join(save_dir, "{}_9_rgb_bg_2.png".format(image_name)), rgb_con_bg_3c)
            cv2.imwrite(os.path.join(save_dir, "{}_10_dep_fg_2.png".format(image_name)), dep_con_fg_3c)
            cv2.imwrite(os.path.join(save_dir, "{}_11_dep_bg_2.png".format(image_name)), dep_con_bg_3c)

            rgb_con[ctr_rgb_dep_fg_bg] = np.array([0, 0, 0])
            rgb_con[ctr_rgb_dep_bg_fg] = np.array([0, 0, 0])
            dep_con[ctr_rgb_dep_fg_bg] = np.array([0, 0, 0])
            dep_con[ctr_rgb_dep_bg_fg] = np.array([0, 0, 0])

            rgb_con_fg_3c[ctr_rgb_dep_fg_bg] = np.array([0, 0, 0])
            rgb_con_bg_3c[ctr_rgb_dep_bg_fg] = np.array([0, 0, 0])
            dep_con_fg_3c[ctr_rgb_dep_bg_fg] = np.array([0, 0, 0])
            dep_con_bg_3c[ctr_rgb_dep_fg_bg] = np.array([0, 0, 0])

            # cv2.imwrite('rgb_con_3.jpg', rgb_con)
            # cv2.imwrite('dep_con_3.jpg', dep_con)
            # cv2.imwrite('rgb_con_fg_3c_3.jpg', rgb_con_fg_3c)
            # cv2.imwrite('rgb_con_bg_3c_3.jpg', rgb_con_bg_3c)
            # cv2.imwrite('dep_con_fg_3c_3.jpg', dep_con_fg_3c)
            # cv2.imwrite('dep_con_bg_3c_3.jpg', dep_con_bg_3c)
            cv2.imwrite(os.path.join(save_dir, "{}_12_rgb_fg_3.png".format(image_name)), rgb_con_fg_3c)
            cv2.imwrite(os.path.join(save_dir, "{}_13_rgb_bg_3.png".format(image_name)), rgb_con_bg_3c)
            cv2.imwrite(os.path.join(save_dir, "{}_14_dep_fg_3.png".format(image_name)), dep_con_fg_3c)
            cv2.imwrite(os.path.join(save_dir, "{}_15_dep_fg_3.png".format(image_name)), dep_con_bg_3c)
            cv2.imwrite(os.path.join(save_dir, "{}_16_rgb.png".format(image_name)), rgb_con)
            cv2.imwrite(os.path.join(save_dir, "{}_17_dep.png".format(image_name)), dep_con)

            con = rgb_con + dep_con
            con[con == 510.] = 255.
            # cv2.imwrite('con.jpg', con)
            cv2.imwrite(os.path.join(save_dir, "{}_18_con.png".format(image_name)), con)

            con_fg = (con[:, :, 2] == 255).astype(np.float64)
            con_all = (np.sum(con, axis=-1) == 255.).astype(np.float64)

            con_fg = torch.from_numpy(con_fg).unsqueeze(0).unsqueeze(0)
            con_all = torch.from_numpy(con_all).unsqueeze(0).unsqueeze(0)

            rgb_sal = torch.from_numpy(rgb_sal).unsqueeze(0).unsqueeze(0)
            dep_sal = torch.from_numpy(dep_sal).unsqueeze(0).unsqueeze(0)

            rgb_sal_prob_2 = torch.cat([rgb_sal, 1 - rgb_sal], dim=1) + 1e-9
            dep_sal_prob_2 = torch.cat([dep_sal, 1 - dep_sal], dim=1) + 1e-9
            prob_mean = (rgb_sal_prob_2 + dep_sal_prob_2) / 2.
            rgb_kl_loss = F.kl_div(prob_mean.log(), rgb_sal_prob_2, reduction="none")
            dep_kl_loss = F.kl_div(prob_mean.log(), dep_sal_prob_2, reduction="none")
            JS_loss = 0.5*rgb_kl_loss.sum(dim=1) + 0.5*dep_kl_loss.sum(dim=1)
            JS_loss = JS_loss.unsqueeze(dim=1)

            rgb_sal_prob_con = rgb_sal * con_all
            dep_sal_prob_con = dep_sal * con_all

            CE_noreduce = torch.nn.BCELoss(reduction="none")

            rgb_con_loss_noreduce = CE_noreduce(rgb_sal_prob_con, con_fg * con_all)
            dep_con_loss_noreduce = CE_noreduce(dep_sal_prob_con, con_fg * con_all)

            pseudo_loss = rgb_con_loss_noreduce + dep_con_loss_noreduce
            joint_loss = 0.1 * pseudo_loss + 0.9 * JS_loss

            pseudo_loss = pseudo_loss.data.cpu().squeeze().numpy()
            JS_loss = JS_loss.data.cpu().squeeze().numpy()
            rgb_con_loss = rgb_con_loss_noreduce.data.cpu().squeeze().numpy()
            dep_con_loss = dep_con_loss_noreduce.data.cpu().squeeze().numpy()

            # pseudo_loss = (pseudo_loss - pseudo_loss.min()) / (pseudo_loss.max() - pseudo_loss.min())
            pseudo_loss = pseudo_loss / 2.
            # JS_loss = (JS_loss - JS_loss.min()) / (JS_loss.max() - JS_loss.min())
            # rgb_con_loss = (rgb_con_loss - rgb_con_loss.min()) / (rgb_con_loss.max() - rgb_con_loss.min())
            # dep_con_loss = (dep_con_loss - dep_con_loss.min()) / (dep_con_loss.max() - dep_con_loss.min())

            con_all = con_all.data.cpu().squeeze().numpy()

            rgb_con_loss = np.expand_dims(rgb_con_loss, 2)
            rgb_con_loss_heat_map = cv2.applyColorMap((rgb_con_loss * 255).astype(np.uint8), cv2.COLORMAP_JET)
            rgb_con_loss_heat_map *= con_all[:, :, np.newaxis].astype(np.uint8)
            # cv2.imwrite('rgb_con_loss_heat_map.png', rgb_con_loss_heat_map)
            cv2.imwrite(os.path.join(save_dir, "{}_19_rgb_bce.png".format(image_name)), rgb_con_loss_heat_map)

            dep_con_loss = np.expand_dims(dep_con_loss, 2)
            dep_con_loss_heat_map = cv2.applyColorMap((dep_con_loss * 255).astype(np.uint8), cv2.COLORMAP_JET)
            dep_con_loss_heat_map *= con_all[:, :, np.newaxis].astype(np.uint8)
            # cv2.imwrite('dep_con_loss_heat_map.png', dep_con_loss_heat_map)
            cv2.imwrite(os.path.join(save_dir, "{}_20_dep_bce.png".format(image_name)), dep_con_loss_heat_map)

            pseudo_loss = np.expand_dims(pseudo_loss, 2)
            pseudo_loss_heat_map = cv2.applyColorMap((pseudo_loss * 255).astype(np.uint8), cv2.COLORMAP_JET)
            pseudo_loss_heat_map *= con_all[:, :, np.newaxis].astype(np.uint8)
            # cv2.imwrite('pseudo_loss_heat_map.png', pseudo_loss_heat_map)
            cv2.imwrite(os.path.join(save_dir, "{}_21_bce.png".format(image_name)), pseudo_loss_heat_map)

            JS_loss = np.expand_dims(JS_loss, 2)
            JS_loss_heat_map = cv2.applyColorMap((JS_loss * 255).astype(np.uint8), cv2.COLORMAP_JET)
            JS_loss_heat_map *= con_all[:, :, np.newaxis].astype(np.uint8)
            # cv2.imwrite('JS_loss_heat_map.png', JS_loss_heat_map)
            cv2.imwrite(os.path.join(save_dir, "{}_22_JS.png".format(image_name)), JS_loss_heat_map)

            joint_loss = joint_loss.data.cpu().squeeze().numpy()

            joint_loss = (joint_loss - joint_loss.min()) / (joint_loss.max() - joint_loss.min())
            # joint_loss = 1. - joint_loss

            joint_loss = np.expand_dims(joint_loss, 2)
            joint_loss_heat_map = cv2.applyColorMap((joint_loss * 255).astype(np.uint8), cv2.COLORMAP_JET)
            # con_all = con_all.data.cpu().squeeze().numpy()
            joint_loss_heat_map *= con_all[:, :, np.newaxis].astype(np.uint8)
            # cv2.imwrite('joint_loss_heat_map.png', joint_loss_heat_map)
            cv2.imwrite(os.path.join(save_dir, "{}_23_joint_loss.png".format(image_name)), joint_loss_heat_map)

            joint_loss = joint_loss[:, :, 0]
            joint_loss_flatten = joint_loss[con_all == 1]
            joint_loss_sorted = np.sort(joint_loss_flatten)
            num_elements_to_keep = int(len(joint_loss_sorted)*0.6)

            threshold_value = joint_loss_sorted[num_elements_to_keep]

            con[joint_loss > threshold_value] = 0

            cv2.imwrite(os.path.join(save_dir, "{}_24_con_2.png".format(image_name)), con)

        return
