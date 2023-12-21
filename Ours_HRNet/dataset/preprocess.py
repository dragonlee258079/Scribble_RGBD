import numpy as np
import torch
from torchvision import transforms
from PIL import Image


class normalization(object):
    def __init__(self, split, scale_size=None):
        self.split = split
        if self.split == 'train':
            self.img_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            self.depth_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.454], [1]),
                ]
            )
            self.mask_transform = transforms.ToTensor()
            self.gray_transform = transforms.ToTensor()
        elif self.split == 'test':
            assert scale_size
            self.img_transform = transforms.Compose(
                [
                    transforms.Resize((scale_size, scale_size), interpolation=Image.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            self.depth_transform = transforms.Compose(
                [
                    transforms.Resize((scale_size, scale_size), interpolation=Image.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.454], [1]),
                ]
            )
        else:
            raise Exception("Split not recognized")

    def __call__(self, input):
        if self.split == 'train':
            image, depth, gt, mask = input['image'], input['depth'], input['gt'], input['mask']
            input['image'] = self.img_transform(image)
            input['depth'] = self.depth_transform(depth)
            input['gt'] = self.mask_transform(gt)
            input['mask'] = self.mask_transform(mask)
            # input['gray'] = self.gray_transform(gray)
        elif self.split == 'test':
            image, depth = input['image'], input['depth']
            input['image'] = self.img_transform(image)
            input['depth'] = self.depth_transform(depth)
        return input
