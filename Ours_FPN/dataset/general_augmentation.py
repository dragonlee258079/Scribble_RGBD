from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import types
import collections
from albumentations.core.transforms_interface import BasicTransform


class Scale(object):
    """Rescale the input PIL.Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size):
        super(Scale, self).__init__()
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = (size, size)

    def _scale(self, img, interpolation=Image.BILINEAR):
        return img.resize(self.size, interpolation)

    def __call__(self, input):
        image, depth, gt, mask = input['image'], input['depth'], input['gt'], input['mask']
        input['image'] = self._scale(image)
        input['depth'] = self._scale(depth)
        # input['gray'] = self._scale(gray)
        input['gt'] = self._scale(gt, interpolation=Image.NEAREST)
        input['mask'] = self._scale(mask, interpolation=Image.NEAREST)
        return input


class Random_Crop(object):
    def __init__(self, t_size):
        self.t_size = t_size

    def _crop(self, img, x1, y1, x2, y2):
        return img.crop((x1, y1, x2, y2))

    def __call__(self, input):
        image, depth, gt, mask = input['image'], input['depth'], input['gt'], input['mask']
        w, h = image.size
        if w != self.t_size and h != self.t_size:
            x1 = random.randint(0, w - self.t_size)
            y1 = random.randint(0, h - self.t_size)
            input['image'] = self._crop(image, x1, y1, x1 + self.t_size, y1 + self.t_size)
            input['depth'] = self._crop(depth, x1, y1, x1 + self.t_size, y1 + self.t_size)
            input['gt'] = self._crop(gt, x1, y1, x1 + self.t_size, y1 + self.t_size)
            input['mask'] = self._crop(mask, x1, y1, x1 + self.t_size, y1 + self.t_size)
            # input['gray'] = self._crop(gray, x1, y1, x1 + self.t_size, y1 + self.t_size)
        return input


class Random_Flip(object):

    def _flip(self, img):
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    def __call__(self, input):
        image, depth, gt, mask = input['image'], input['depth'], input['gt'], input['mask']
        if random.random() < 0.5:
            input['image'] = self._flip(image)
            input['depth'] = self._flip(depth)
            input['gt'] = self._flip(gt)
            input['mask'] = self._flip(mask)
            # input['gray'] = self._flip(gray)
        return input
