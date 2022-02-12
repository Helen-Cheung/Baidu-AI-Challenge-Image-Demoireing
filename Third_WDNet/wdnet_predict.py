import argparse
import glob
import os.path
from math import ceil
import paddle
import paddle.nn as nn
import cv2
from net_utils import PAN
import os

from model import WDNet, WaveletTransform
from utils import load_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser(description='Model testing')

    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='The path of dataset root',
        type=str,
        default='../moire_testA_dataset')

    parser.add_argument(
        '--pretrained',
        dest='pretrained',
        help='The pretrained of model',
        type=str,
        default='./train_result/model/epoch_1800/model.pdparams')  

    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='batch_size',
        type=int,
        default=32
    )

    return parser.parse_args()


def chop_forward(model, inp, shave=8, min_size=160000):
    b, c, h, w = inp.shape
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    
    mod_size = 4
    if h_size%mod_size:
        h_size = ceil(h_size/mod_size)*mod_size  # The ceil() function returns the uploaded integer of a number
    if w_size%mod_size:
        w_size = ceil(w_size/mod_size)*mod_size
        
    inputlist = [
        inp[:, :, 0:h_size, 0:w_size],
        inp[:, :, 0:h_size, (w - w_size):w],
        inp[:, :, (h - h_size):h, 0:w_size],
        inp[:, :, (h - h_size):h,  (w - w_size):w] 
    ]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(4):
            with paddle.no_grad():
                input_batch = inputlist[i] 
                output_batch = model(input_batch)
            outputlist.append(output_batch) 
    else:
        outputlist = [
            chop_forward(model, patch) \
            for patch in inputlist]

    scale=1
    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    with paddle.no_grad(): 
        output_ht = paddle.zeros_like(inp)

    output_ht[:, :, 0:h_half, 0:w_half] = outputlist[0][:, :, 0:h_half, 0:w_half]
    output_ht[:, :, 0:h_half, w_half:w] = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output_ht[:, :, h_half:h, 0:w_half] = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output_ht[:, :, h_half:h, w_half:w] = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output_ht


def main(args):
    # model = PAN(in_nc=3, out_nc=3, nf=64, unf=64, nb=4)
    model = WDNet()

    if args.pretrained is not None:
        load_pretrained_model(model, args.pretrained)

    im_files = glob.glob(os.path.join(args.dataset_root, "images/*.jpg"))

    for i, im in enumerate(im_files):
        print(os.path.basename(im))
        img = cv2.imread(im)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = paddle.to_tensor(img)
        img /= 255.0
        img = paddle.transpose(img, [2, 0, 1])
        model.eval()

        img = img.unsqueeze(0)
        
        img_out = model(img)

        # try:
        #     img_out = model(img)
        # except:
        #     img_out = chop_forward(model, img)
            
        img_out = img_out.squeeze(0)
        img_out = img_out * 255.0
        img_out = paddle.clip(img_out, 0, 255)
        img_out = paddle.transpose(img_out, [1, 2, 0])
        img_out = img_out.numpy()

        save_path = "output/images"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_path, im.split('/')[-1]), img_out)


if __name__ == '__main__':
    args = parse_args()
    main(args)

