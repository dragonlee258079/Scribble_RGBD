import argparse
import os
import pprint

import torch
import numpy as np

from dataset import build_data_loader

from module.WRGBD import WRGBD
from module.loss import smoothness_loss_jing

from utilities.checkpoint_saver import Saver

from config.config import get_cfg
from solver.build import build_optimizer, build_lr_scheduler

from utilities.log import create_logger

from torch.autograd import Variable
import torch.nn.functional as F

from utilities.inference_step import inference as inference_step
from utilities.inference import inference

from ersr_loss import *

loss_ersr_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_ersr_radius = 5


def get_args_parser():
    """
    Parse arguments
    """

    parser = argparse.ArgumentParser('Set Stereo SOD', add_help=False)
    parser.add_argument("--config_file", default="./config/hrnet_w48.yaml", metavar="FILE",
                        help="path to config file")
    parser.add_argument("--device", default="cuda",
                        help="device to use for training / testing")
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--model_root_dir', default='/data3/lilong/Weakly_RGBD/Models',
                        help="dir for saving checkpoint")
    parser.add_argument('--train_data_file', default='./data/train/train_list.txt')
    parser.add_argument('--batch_size', default=9, type=int)
    parser.add_argument('--batch_size_test', default=1, type=int)
    parser.add_argument('--device_id', type=str, default="2",
                        help="choose cuda_visiable_devices")
    parser.add_argument('--test_dir', default="./data/test/", type=str)
    parser.add_argument('--test_step_list', nargs='+', default=["DUT-RGBD"])
    parser.add_argument('--test_list', nargs='+', default=["DUT-RGBD", "LFSD", "NJU2K", "NLPR",
                                                           "RGBD135", "SIP", "SSD100", "STERE",
                                                           "ReDWeb_S"])
    parser.add_argument('--save_dir', type=str, default="/data3/lilong/Weakly_RGBD/Prediction",
                        help="save prediction result")

    return parser


def save_checkpoint(iteration, model, optimizer, lr_scheduler, checkpoint_saver):
    """
    Save current state of training
    """

    # save model
    checkpoint = {
        'epoch': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }
    checkpoint_saver.save_checkpoint(checkpoint, 'model_epo{}.pth.tar'.format(str(iteration)), write_best=False)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg, args


def select_pseudo_region_union(rgb_pred, dep_pred, percent):
    bs, _, _, _ = rgb_pred.shape

    # print(rgb_pred.shape, dep_pred.shape)
    # print(torch.mean(rgb_pred), torch.mean(dep_pred))
    # print(torch.max(dep_pred.view(bs, -1), dim=1)[0], torch.max(rgb_pred.view(bs, -1), dim=1)[0], percent)

    dep_fg_thresh = torch.max(dep_pred.view(bs, -1), dim=1)[0]*percent
    dep_fg_thresh = dep_fg_thresh.clamp(min=0.5)
    dep_bg_thresh = 1 - dep_fg_thresh

    rgb_fg_thresh = torch.max(rgb_pred.view(bs, -1), dim=1)[0]*percent
    rgb_fg_thresh = rgb_fg_thresh.clamp(min=0.5)
    rgb_bg_thresh = 1 - rgb_fg_thresh

    rgb_con_fg = dep_pred >= dep_fg_thresh.view(bs, 1, 1, 1)
    rgb_con_bg = dep_pred <= dep_bg_thresh.view(bs, 1, 1, 1)

    dep_con_fg = rgb_pred >= rgb_fg_thresh.view(bs, 1, 1, 1)
    dep_con_bg = rgb_pred <= rgb_bg_thresh.view(bs, 1, 1, 1)

    # print(dep_fg_thresh, dep_bg_thresh)
    rgb_con_fg[dep_con_bg] = False
    rgb_con_bg[dep_con_fg] = False
    dep_con_fg[rgb_con_bg] = False
    dep_con_bg[rgb_con_fg] = False

    con_fg = rgb_con_fg | dep_con_fg
    con_bg = rgb_con_bg | dep_con_bg

    con = (con_fg | con_bg).to(torch.float32)
    con_fg = con_fg.to(torch.float32)

    # print(torch.unique(con))

    return con, con_fg


def calculate_pseudo_precise(con, con_fg, full_gt):
    ture_fg_num = torch.sum(full_gt[con_fg == 1] == 1)
    con_bg = con - con_fg
    ture_bg_num = torch.sum(full_gt[con_bg == 1] == 0)
    precise = (ture_fg_num + ture_bg_num) / torch.sum(con)

    return precise


def get_edge(sal_prob):
    max_pool = F.max_pool2d(sal_prob, 9, 1, 4)
    neg_max_pool = F.max_pool2d(-sal_prob, 9, 1, 4)
    edge_mask = max_pool + neg_max_pool

    return edge_mask


def main(args):
    cfg, args = setup(args)

    logger, experiment_dir = create_logger(args)

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    #build model
    logger.info("=> building model")
    model = WRGBD(cfg)
    # model.init_weights(cfg.MODEL.PRETRAINED)
    model.cuda()
    logger.info(model)

    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)

    # initiate saver
    checkpoint_saver = Saver(experiment_dir)

    # build dataloader
    data_loader_train = build_data_loader(args, cfg, mode="train")

    # loss
    CE = torch.nn.BCELoss()
    CE_noreduce = torch.nn.BCELoss(reduction="none")
    loss_ersr = ERSR().cuda()
    # loss_lsc = LocalSaliencyCoherence().cuda()
    # smooth_loss = smoothness_loss_jing

    # train
    model.train()
    max_iter = cfg.SOLVER.MAX_ITER
    max_epoches = cfg.SOLVER.MAX_EPOCHES
    con_thresh_init = cfg.SOLVER.CON_THRESH_INIT
    con_thresh_final = cfg.SOLVER.CON_THRESH_FINAL
    small_loss_init = cfg.SOLVER.SMALL_LOSS_INIT
    small_loss_final = cfg.SOLVER.SMALL_LOSS_FINAL
    num_gradual = cfg.SOLVER.NUM_GRADUAL

    rate_schedule = torch.ones(max_epoches) * con_thresh_final
    rate_schedule[:num_gradual] = torch.linspace(con_thresh_init, con_thresh_final, num_gradual)

    small_loss_rate_schedule = torch.ones(max_epoches) * small_loss_final
    small_loss_rate_schedule[:num_gradual] = torch.linspace(small_loss_init, small_loss_final, num_gradual)

    rd_lrw = np.ones(max_epoches) * 0.8
    rd_lrw[:num_gradual] = np.linspace(0.4, 0.8, num_gradual)

    for epo in range(max_epoches):

        for iteration, data in enumerate(data_loader_train):
            iteration = iteration + 1

            # read data
            image, depth = Variable(data['image'].cuda()), Variable(data['depth'].cuda())
            gt, mask = Variable(data['gt'].cuda()), Variable(data['mask'].cuda())
            rgbd = torch.cat([image, depth], dim=1)
            # rgbd = torch.cat([image, depth], dim=1)
            # gray = Variable(data['gray'].cuda())
            depth = depth.repeat(1, 3, 1, 1)

            # forward pass
            rgb_sal, dep_sal, rgbd_sal = model(image, depth)

            rgb_sal_prob = torch.sigmoid(rgb_sal)
            dep_sal_prob = torch.sigmoid(dep_sal)
            rgbd_sal_prob = torch.sigmoid(rgbd_sal)

            with torch.no_grad():
                rgb_edge_mask = get_edge(rgb_sal_prob)
                dep_edge_mask = get_edge(dep_sal_prob)
                rgbd_edge_mask = get_edge(rgbd_sal_prob)
                edge_mask = (rgb_edge_mask + dep_edge_mask + rgbd_edge_mask) / 3.

            sal_prob = torch.cat([rgb_sal_prob, dep_sal_prob, rgbd_sal_prob], dim=1)

            mask = F.interpolate(mask, size=rgb_sal.size()[2:], mode="nearest")
            gt = F.interpolate(gt, size=rgb_sal.size()[2:], mode="nearest")
            rgbd = F.interpolate(rgbd, size=rgb_sal.size()[2:], mode='bilinear', align_corners=False)

            sample = {'rgb': rgbd}
            ersr_loss = loss_ersr(sal_prob, edge_mask, loss_ersr_kernels_desc_defaults, loss_ersr_radius, sample, sal_prob.size(2),
                                sal_prob.size(3))['loss']

            con, con_fg = select_pseudo_region_union(rgb_sal_prob, dep_sal_prob, rate_schedule[epo])

            con[mask == 1] = 0
            con_fg[gt == 1] = 0

            rgb_sal_prob_2 = torch.cat([rgb_sal_prob, 1 - rgb_sal_prob], dim=1) + 1e-9
            dep_sal_prob_2 = torch.cat([dep_sal_prob, 1 - dep_sal_prob], dim=1) + 1e-9
            prob_mean = (rgb_sal_prob_2 + dep_sal_prob_2) / 2.
            rgb_kl_loss = F.kl_div(prob_mean.log(), rgb_sal_prob_2, reduction="none")
            dep_kl_loss = F.kl_div(prob_mean.log(), dep_sal_prob_2, reduction="none")
            JS_loss = 0.5*rgb_kl_loss.sum(dim=1) + 0.5*dep_kl_loss.sum(dim=1)
            JS_loss = JS_loss.unsqueeze(dim=1)

            img_size = rgb_sal.size(2) * rgb_sal.size(3) * rgb_sal.size(0)
            ratio = img_size / torch.sum(mask)

            rgb_sal_prob_scr = rgb_sal_prob * mask
            rgb_sal_prob_con = rgb_sal_prob * con
            dep_sal_prob_scr = dep_sal_prob * mask
            dep_sal_prob_con = dep_sal_prob * con
            rgbd_sal_prob_scr = rgbd_sal_prob * mask
            rgbd_sal_prob_con = rgbd_sal_prob * con

            rgb_con_loss_noreduce = CE_noreduce(rgb_sal_prob_con, con_fg * con)
            dep_con_loss_noreduce = CE_noreduce(dep_sal_prob_con, con_fg * con)
            rgbd_con_loss_noreduce = CE_noreduce(rgbd_sal_prob_con, con_fg * con)
            pseudo_loss = rgb_con_loss_noreduce + dep_con_loss_noreduce

            joint_loss = 0.1*pseudo_loss + 0.9*JS_loss

            joint_loss = joint_loss[con == 1]
            pseudo_loss = pseudo_loss[con == 1]
            rgbd_pseudo_loss = rgbd_con_loss_noreduce[con == 1]

            ind_sorted = torch.argsort(joint_loss)
            pseudo_loss_sorted = pseudo_loss[ind_sorted]
            rgbd_pseudo_loss_sorted = rgbd_pseudo_loss[ind_sorted]

            remember_rate = 1 - small_loss_rate_schedule[epo]
            num_remember = int(remember_rate * len(pseudo_loss_sorted))

            picked_pseudo_loss = torch.mean(pseudo_loss_sorted[:num_remember])
            picked_rgbd_pseudo_loss = torch.mean(rgbd_pseudo_loss_sorted[:num_remember])

            # compute lossed
            rgb_pce = ratio * CE(rgb_sal_prob_scr, gt * mask)
            dep_pce = ratio * CE(dep_sal_prob_scr, gt * mask)
            rgbd_pce = ratio * CE(rgbd_sal_prob_scr, gt * mask)

            JS_loss_mean = torch.mean(JS_loss)

            loss = rgb_pce + dep_pce + rgbd_pce + 0.7*picked_rgbd_pseudo_loss + rd_lrw[epo]*picked_pseudo_loss \
                   + 1.2*JS_loss_mean + 0.3*ersr_loss

            # backprop
            optimizer.zero_grad()
            loss.backward()

            # step optimizer
            optimizer.step()
            lr_scheduler.step()

            if (iteration - 1) % 5 == 0:
                logger.info("iter:%d total_loss:%.3f rgb_pce:%.3f, dep_pce:%.3f, rgbd_pce:%.3f, "
                            "rgbd_pseudo_loss:%.3f, pseudo_loss:%.3f, JS_loss:%.3f, ersr_loss:%.3f, lr:%.7f" %(
                            iteration, loss, rgb_pce, dep_pce, rgbd_pce, picked_rgbd_pseudo_loss,
                            picked_pseudo_loss, JS_loss_mean, ersr_loss, optimizer.param_groups[1]['lr']
                ))

            # if iteration % 10000 == 0:
            #     save_checkpoint(iteration, model, optimizer, lr_scheduler, checkpoint_saver)
            #     model.eval()
            #     model_dir = os.path.join(experiment_dir, "model_{}.pth.tar".format(str(iteration)))
            #     inference_step(model, cfg, args, model_dir)
            #     model.train()

            # if iteration % 60000 == 0:
            #     model_dir = os.path.join(experiment_dir, "model_60000.pth.tar")
            #     inference(model, cfg, args, model_dir)
            #     if os.path.exists('experiment_dir.txt'):
            #         os.remove('experiment_dir.txt')
            #     f = open('experiment_dir.txt', 'wb')
            #     f.write(experiment_dir)
            #     f.close()


        save_checkpoint(epo, model, optimizer, lr_scheduler, checkpoint_saver)

        model.eval()
        model_dir = os.path.join(experiment_dir, "model_epo{}.pth.tar".format(str(epo)))
        inference_step(model, cfg, args, model_dir)
        model.train()

    return


if __name__ == '__main__':
    ap = argparse.ArgumentParser('Stereo SOD training and evaluation script', parents=[get_args_parser()])
    args_ = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args_.device_id
    main(args_)
