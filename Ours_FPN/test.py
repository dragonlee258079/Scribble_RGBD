import argparse
import os
import torch

from module.WRGBD import WRGBD
from utilities.inference import inference

from config.config import get_cfg


def get_args_parser():
    """
    Parse arguments
    """

    parser = argparse.ArgumentParser('Set Stereo SOD', add_help=False)
    parser.add_argument("--config_file", default="./config/wrgbd.yaml", metavar="FILE",
                        help="path to config file")
    parser.add_argument("--device", default="cuda",
                        help="device to use for training / testing")
    parser.add_argument('--batch_size_test', default=1, type=int)
    parser.add_argument('--device_id', type=str, default="4",
                        help="choose cuda_visiable_devices")
    parser.add_argument('--model_root_dir', default='./checkpoint',
                        help="dir for saving checkpoint")
    parser.add_argument('--save_dir', type=str, default="predictions",
                        help="save prediction result")
    parser.add_argument('--test_model', type=str, default="ours_fpn",
                        help="checkpoint name for test. [ours_fpn, ours_hrnet, ours_fpn_xu]")
    parser.add_argument('--test_dir', default="./data/test/", type=str)
    parser.add_argument('--test_list', nargs='+', default=["DUT-RGBD", "LFSD", "NJU2K", "NLPR",
                                                           "RGBD135", "SIP", "SSD100", "STERE", "ReDWeb_S"])
    parser.add_argument('--num_workers', default=1, type=int)
    return parser


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg, args


def main(args):
    cfg, args = setup(args)

    # get device
    device = torch.device(args.device)

    #build model
    net = WRGBD(cfg).to(device)

    model_root_dir = args.model_root_dir
    test_model = args.test_model

    print("Start inference {}".format(test_model))
    model_dir = os.path.join(model_root_dir, "{}.pth.tar".format(test_model))
    inference(net, cfg, args, model_dir)

    return


if __name__ == '__main__':
    ap = argparse.ArgumentParser('Stereo SOD training and evaluation script', parents=[get_args_parser()])
    args_ = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args_.device_id
    main(args_)
