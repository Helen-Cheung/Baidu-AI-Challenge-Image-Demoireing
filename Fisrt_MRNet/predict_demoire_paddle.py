import argparse
import glob
import os.path

import paddle
import paddle.nn as nn
import cv2
import numpy as np
from MRNET import MRNET
from utils import load_pretrained_model
import paddle.nn.functional as F
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Model testing')

    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='The path of dataset root',
        type=str,
        default='/data/datasets/moire/baidu/moire_testB_dataset/')

    parser.add_argument(
        '--pretrained',
        dest='pretrained',
        help='The pretrained of model',
        type=str,
        default='./model/paddle_model.pdparams')

    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='batch_size',
        type=int,
        default=32
    )

    return parser.parse_args()


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


def main(args):
    model = MRNET()

    if args.pretrained is not None:
        load_pretrained_model(model, args.pretrained)

    im_files = glob.glob(os.path.join(args.dataset_root, "images/*.jpg"))
    t = 0
    for i, im in enumerate(im_files):
        print(im)
        model.eval()
        img = cv2.imread(im)

        img_ = img.transpose(2, 0, 1).astype(np.float32)
        img_ = normalize(paddle.to_tensor(img_).unsqueeze(0))

        factor = 16
        h, w = img_.shape[2], img_.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        img_ = F.pad(img_, (0, padw, 0, padh), 'constant', 0)
        t1 = time.time()
        output = model(img_).squeeze(0)
        t2 = time.time()
        t += (t2 - t1)

        output = normalize_reverse(output)
        output = output.numpy()
        demoire_img = output.transpose(1, 2, 0)

        demoire_img = np.clip(demoire_img, 0, 255.0)
        demoire_img = demoire_img.astype(np.uint8)

        img_out = demoire_img[:img.shape[0], :img.shape[1], :]

        save_path = "./output/pre"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_path, im.split('/')[-1]), img_out)
    print('Time: ', t / len(im_files))


if __name__ == '__main__':
    args = parse_args()
    main(args)
