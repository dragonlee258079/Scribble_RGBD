import argparse
import os
import pprint

import torch

from dataset import build_data_loader

from module.WRGBD import WRGBD

from utilities.checkpoint_saver import Saver

from config.config import get_cfg
from solver.build import build_optimizer, build_lr_scheduler

from utilities.log import create_logger

from torch.autograd import Variable
import torch.nn.functional as F

from utilities.inference_step import inference as inference_step
from utilities.inference import inference

from ERSR_loss import *

loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5


def get_args_parser():
    """
    Parse arguments
    """

    parser = argparse.ArgumentParser('Set Stereo SOD', add_help=False)
    parser.add_argument("--config_file", default="./config/wrgbd.yaml", metavar="FILE",
                        help="path to config file")
    parser.add_argument("--device", default="cuda",
                        help="device to use for training / testing")
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--model_root_dir', default='./checkpoint',
                        help="dir for saving checkpoint")
    parser.add_argument('--train_data_file', default='./data/train/train_list_xu.txt')
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--batch_size_test', default=1, type=int)
    parser.add_argument('--device_id', type=str, default="4",
                        help="choose cuda_visiable_devices")
    parser.add_argument('--test_dir', default="./data/test/", type=str)
    parser.add_argument('--test_step_list', nargs='+', default=["DUT-RGBD"])
    parser.add_argument('--test_list', nargs='+', default=["DUT-RGBD", "LFSD", "NJU2K", "NLPR",
                                                           "RGBD135", "SIP", "SSD100", "STERE",
                                                           "ReDWeb_S"])
    parser.add_argument('--unc_check_imgs', default='./data/check/unc_check_img_list.txt', type=str,
                        help="choosed images to visualize the uncertainty map")
    parser.add_argument('--save_check', default='./unc_check/npy')
    parser.add_argument('--save_dir', type=str, default="predictions",
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
    checkpoint_saver.save_checkpoint(checkpoint, 'model_{}.pth.tar'.format(str(iteration)), write_best=False)


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

    dep_fg_thresh = torch.max(dep_pred.view(bs, -1), dim=1)[0]*percent
    dep_bg_thresh = 1 - dep_fg_thresh

    rgb_fg_thresh = torch.max(rgb_pred.view(bs, -1), dim=1)[0]*percent
    rgb_bg_thresh = 1 - rgb_fg_thresh

    rgb_con_fg = dep_pred >= dep_fg_thresh.view(bs, 1, 1, 1)
    rgb_con_bg = dep_pred <= dep_bg_thresh.view(bs, 1, 1, 1)

    dep_con_fg = rgb_pred >= rgb_fg_thresh.view(bs, 1, 1, 1)
    dep_con_bg = rgb_pred <= rgb_bg_thresh.view(bs, 1, 1, 1)

    rgb_con_fg[dep_con_bg] = False
    rgb_con_bg[dep_con_fg] = False
    dep_con_fg[rgb_con_bg] = False
    dep_con_bg[rgb_con_fg] = False

    con_fg = rgb_con_fg | dep_con_fg
    con_bg = rgb_con_bg | dep_con_bg

    con = (con_fg | con_bg).to(torch.float32)
    con_fg = con_fg.to(torch.float32)

    return con, con_fg


def get_edge(sal_prob):
    max_pool = F.max_pool2d(sal_prob, 5, 1, 2)
    neg_max_pool = F.max_pool2d(-sal_prob, 5, 1, 2)
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

    for epo in range(max_epoches):

        for iteration, data in enumerate(data_loader_train):
            iteration = iteration + 1

            # read data
            image, depth = Variable(data['image'].cuda()), Variable(data['depth'].cuda())
            gt, mask = Variable(data['gt'].cuda()), Variable(data['mask'].cuda())
            rgbd = torch.cat([image, depth], dim=1)
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
            lsc_loss = loss_ersr(sal_prob, edge_mask, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, sal_prob.size(2),
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

            loss = rgb_pce + dep_pce + rgbd_pce + 0.3*picked_rgbd_pseudo_loss + 0.6*picked_pseudo_loss + 1.2*JS_loss_mean + 0.3*lsc_loss

            # backprop
            optimizer.zero_grad()
            loss.backward()

            # step optimizer
            optimizer.step()
            lr_scheduler.step()

            if (iteration - 1) % 5 == 0:
                logger.info("iter:%d total_loss:%.3f rgb_pce:%.3f, dep_pce:%.3f, rgbd_pce:%.3f, "
                            "rgbd_pseudo_loss:%.3f, pseudo_loss:%.3f, JS_loss:%.3f, lsc_loss:%.3f, lr:%.7f" %(
                            iteration, loss, rgb_pce, dep_pce, rgbd_pce, picked_rgbd_pseudo_loss,
                            picked_pseudo_loss, JS_loss_mean, lsc_loss, optimizer.param_groups[1]['lr']
                ))

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
