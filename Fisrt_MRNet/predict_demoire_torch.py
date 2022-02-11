import argparse
import glob
import os

import torch
import cv2
import numpy as np
from MRNETTORCH import MRNET_TORCH
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(description='Model testing')

    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='The path of dataset root',
        type=str,
        default='/data/datasets/moire/baidu/moire_testA_dataset/')

    parser.add_argument(
        '--pretrained',
        dest='pretrained',
        help='The pretrained of model',
        type=str,
        default='./model/torch_model.pth.tar')

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
    model = MRNET_TORCH().cuda()

    if args.pretrained is not None:
        chechpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage.cuda(0))
        model.load_state_dict(chechpoint)

    im_files = glob.glob(os.path.join(args.dataset_root, "images/*.jpg"))

    for i, im in enumerate(im_files):
        print(im)
        model.eval()
        with torch.no_grad():
            img = cv2.imread(im)

            img_ = img.transpose(2, 0, 1).astype(np.float32)
            img_ = normalize(torch.from_numpy(img_).unsqueeze(0).cuda())

            factor = 16
            h, w = img_.shape[2], img_.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            img_ = F.pad(img_, (0, padw, 0, padh), 'constant', 0)

            output = model(img_).squeeze(dim=0)

            output = normalize_reverse(output)
            output = output.data.cpu().numpy()
            demoire_img = output.transpose(1, 2, 0)

            demoire_img = np.clip(demoire_img, 0, 255.0)
            demoire_img = demoire_img.astype(np.uint8)

            img_out = demoire_img[:img.shape[0], :img.shape[1], :]

            save_path = "./output/pre_torch"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_path, im.split('/')[-1]), img_out)


if __name__ == '__main__':
    args = parse_args()
    main(args)
