import torch.utils.data as data

from dataset.dataload import BaseDataset

import os


class _iteration():
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return self.next()

    def next(self):
        while 1:
            for d in self.dataset:
                yield d


def build_data_loader(args, cfg, mode):
    '''
    Build data loader

    :param args: arg parser object
    :return: train, validation and test dataloaders
    '''
    if mode == "train":
        train_dataset = BaseDataset(args.train_data_file, cfg, 'train')
        data_loader_train = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.num_workers, pin_memory=True)
        return data_loader_train
    elif mode == "test":
        test_file_root = args.test_dir
        test_list = args.test_list
        test_dataloaders = []
        for t in test_list:
            test_dir = os.path.join(test_file_root, t+'.txt')
            dataset = BaseDataset(test_dir, cfg, 'test')
            data_loader = data.DataLoader(dataset, batch_size=args.batch_size_test, shuffle=False,
                                          num_workers=args.num_workers, pin_memory=True)
            test_dataloaders.append(data_loader)
        return test_dataloaders
    else:
        raise RuntimeError
