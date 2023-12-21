import torch

from utilities.misc import NestedTensor


def write_summary(stats, summary, epoch, mode):
    """
    write the current epoch result to tensorboard
    """
    summary.writer.add_scalar(mode + '/R_I', stats['R_I'], epoch)
    summary.writer.add_scalar(mode + '/R_S', stats['R_S'], epoch)
    summary.writer.add_scalar(mode + '/R_C', stats['R_C'], epoch)
    summary.writer.add_scalar(mode + '/Cycle', stats['Cycle'], epoch)
    summary.writer.add_scalar(mode + '/sal', stats['sal'], epoch)
    summary.writer.add_scalar(mode + '/con', stats['con'], epoch)


def forward_pass(model, data, criterion, device, stats, idx=0, logger=None):
    """
    forward pass of the model given input
    """
    # read data
    left, right = data['left'].to(device), data['right'].to(device)
    gt, edge = data['gt'].to(device), data['edge'].to(device)

    # build the input
    inputs = NestedTensor(left, right)

    # forward pass
    disp, att, sal, con = model(inputs)

    # compute losses
    losses = criterion(left, right, sal, con, att, gt, edge)

    # get the loss
    stats["R_I"] += losses["R_I"].item()
    stats["R_S"] += losses["R_S"].item()
    stats["R_C"] += losses["R_C"].item()
    stats["Cycle"] += losses["Cycle"].item()

    stats["sal"] += losses["sal"].item()
    stats["con"] += losses["con"].item()

    # log for eval only
    if logger is not None:
        logger.info('Index %d, R_I %d, R_S %d, R_C %d, Cycle %d, sal %d, con %d' %
                    (idx, losses["R_I"].item(), losses["R_S"].item(), losses["R_C"].item(),
                          losses["Cycle"].item(), losses["sal"].item(), losses["con"].item()))

    return losses
