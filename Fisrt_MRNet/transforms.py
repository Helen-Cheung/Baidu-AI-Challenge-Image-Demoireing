import random
import paddle
import numpy as np
import cv2


class Compose:
    def __init__(self, transforms):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms

    def __call__(self, im1, im2):

        if isinstance(im1, str):
            im1 = cv2.imread(im1)
        if isinstance(im2, str):
            im2 = cv2.imread(im2)
        if im1 is None or im2 is None:
            raise ValueError('Can\'t read The image file {} and {}!'.format(im1, im2))

        for op in self.transforms:
            outputs = op(im1, im2)
            im1 = outputs[0]
            im2 = outputs[1]
        im1 = normalize(im1)
        im2 = normalize(im2)
        return im1, im2


class Flip(object):
    def __init__(self, prob_lr=0.5, prob_ud=0.5):
        self.prob_lr = prob_lr
        self.prob_ud = prob_ud

    def __call__(self, im1, im2):
        if random.random() < self.prob_lr:
            im1 = np.fliplr(im1)
            im2 = np.fliplr(im2)
        if random.random() < self.prob_ud:
            im1 = np.flipud(im1)
            im2 = np.flipud(im2)
        return im1, im2


class ToTensor(object):
    def __call__(self, im1, im2):
        im1 = np.ascontiguousarray(im1.transpose((2, 0, 1)).astype(np.float32))
        im2 = np.ascontiguousarray(im2.transpose((2, 0, 1)).astype(np.float32))
        im1 = paddle.to_tensor(im1)
        im2 = paddle.to_tensor(im2)
        return im1, im2


def normalize(x, centralize=True, normalize=True, val_range=255.0):
    if centralize:
        x = x - val_range / 2
    if normalize:
        x = x / val_range

    return x


def normalize_reverse(x, centralize=True, normalize=True, val_range=255.0):
    if normalize:
        x = x * val_range
    if centralize:
        x = x + val_range / 2

    return x
