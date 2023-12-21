import math
import sys
from typing import Iterable

import torch
from tqdm import tqdm

from utilities.forward_pass import forward_pass, write_summary
from utilities.summary_logger import TensorboardSummary


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, criterion: object, optimizer: torch.optim.Optimizer,
                    lr_scheduler:torch.optim.lr_scheduler._LRScheduler,
                    device: torch.device, epoch: int, summary: TensorboardSummary,
                    amp: object = None):
    """
    train model for 1 epoch
    """
    model.train()

    # initialize stats
    train_stats = {"R_I": 0.0, "R_S": 0.0, "R_C": 0.0, "Cycle": 0.0, "sal": 0.0, "con": 0.0}

    # tbar = tqdm(data_loader)
    for idx, data in enumerate(data_loader):
        # forward pass
        loss = forward_pass(model, data, criterion, device, train_stats)

        if idx % 5 == 0:
            print("iter:%d total_loss:%.3f R_I:%.3f R_S:%.3f R_C:%.3f Cycle:%.3f sal:%.3f con:%.3f" %(
                idx, loss["aggregate"], loss["R_I"], loss["R_S"], loss["R_C"], loss["Cycle"],
                loss["sal"], loss["con"]
            ))

        # backprop
        optimizer.zero_grad()
        if amp is not None:
            with amp.scale_loss(loss["aggregate"], optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss["aggregate"].backward()

        # step optimizer
        optimizer.step()
        lr_scheduler.step()

        # clear cache
        torch.cuda.empty_cache()

    # log to tensorboard
    write_summary(train_stats, summary, epoch, 'train')

    return
