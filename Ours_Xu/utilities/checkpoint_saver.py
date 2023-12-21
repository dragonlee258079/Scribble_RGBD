import glob
import os

import torch


class Saver(object):

    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir

        # self.save_experiment_config()

    def save_checkpoint(self, state, filename="model.pth.tar", write_best=True):
        """Save checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)

        # best_pred = state['best_pred']
        # if write_best:
        #     with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
        #         f.write(str(best_pred))

    # def save_experiment_config(self):
    #     with open(os.path.join(self.experiment_dir, 'parameters.txt'), 'w') as file:
    #         config_dict = vars(self.args)
    #         for k in vars(self.args):
    #             file.write(f"{k}={config_dict[k]} \n")
