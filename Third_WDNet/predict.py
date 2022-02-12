import argparse
import glob
import time
import os.path
import os
import paddle
import paddle.nn as nn
import cv2
import shutil
import sys

from model import WDNet, WaveletTransform
from net_utils import PAN
from utils import load_pretrained_model, chop_forward, average_inference


def parse_args():
    parser = argparse.ArgumentParser(description='Model testing')
    parser.add_argument('--dataset_root',dest='dataset_root',help='The path of dataset root',type=str,default='../moire_testB_dataset')
    parser.add_argument('--pretrained',dest='pretrained',help='The pretrained of model',type=str,default='./weight') 
    parser.add_argument('--batch_size',dest='batch_size',help='batch_size',type=int,default=32)
    return parser.parse_args()


def pwdnet_inference(model, img, wavelet_dec, wavelet_rec):
    model.eval()
    img = paddle.to_tensor(img)
    img /= 255.0
    img = paddle.transpose(img, [2, 0, 1])
    img = img.unsqueeze(0)
    img_in = wavelet_dec(img)
    img_out = model(img_in)
    img_out = wavelet_rec(img_out)
    img_out = nn.functional.interpolate(img_out, size=img.shape[2:], mode="bilinear")
    img_out = img_out + img
    img_out = img_out.squeeze(0)

    img_out = img_out * 255.0
    img_out = paddle.clip(img_out, 0, 255)
    img_out = paddle.transpose(img_out, [1, 2, 0])
    img_out = img_out.numpy()
    return img_out


def pan_inference(model, img, height, width):
    model.eval()
    img = paddle.to_tensor(img)
    img /= 255.0
    img = paddle.transpose(img, [2, 0, 1])
    img = img.unsqueeze(0)

    if height * width < 120000:
        img_out = model(img)
    else:
        img_out = chop_forward(model, img)
        
    img_out = img_out.squeeze(0)
    img_out = img_out * 255.0
    img_out = paddle.clip(img_out, 0, 255)
    img_out = paddle.transpose(img_out, [1, 2, 0])
    img_out = img_out.numpy()
    return img_out


def main(args, save_path='./output/pre01', with_flip=False):
    # load pwdnet
    pwdnet = WDNet()
    wavelet_dec = WaveletTransform(scale=2, dec=True)
    wavelet_rec = WaveletTransform(scale=2, dec=False)

    # load pan net
    pannet = PAN(in_nc=3, out_nc=3, nf=64, unf=64, nb=6)

    if args.pretrained is not None:
        pwdnet_path = os.path.join(args.pretrained, "pwdnet_epoch_395", "model.pdparams")
        load_pretrained_model(pwdnet, pwdnet_path)
        pwdnet.eval()

        pannet_path = os.path.join(args.pretrained, "epoch_980", "model.pdparams")  
        load_pretrained_model(pannet, pannet_path)
        pannet.eval()
    
    im_files = glob.glob(os.path.join(args.dataset_root, "images/*.jpg"))

    time_sum = 0
    for idx, im in enumerate(im_files):
        sys.stdout.write("{} | {}".format(idx+1, len(im_files)))
        img = cv2.imread(im)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # flip
        if with_flip:
            img = cv2.flip(img, 90)

        height, width, _ = img.shape

        start = time.time()
        # stage 1
        img_out = pwdnet_inference(pwdnet, img, wavelet_dec, wavelet_rec)

        # stage 2
        img_out = pan_inference(pannet, img_out, height, width)
        end = time.time()

        time_sum += end-start

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_path, im.split('/')[-1]), img_out)

    
    sys.stdout.write('The running time of an image is: {:2f} s'.format(time_sum / len(im_files)))


if __name__ == '__main__':
    args = parse_args()
    path1 = './output/pre01'
    path2 = './output/pre02'

    main(args, save_path=path1)
    main(args, save_path=path2, with_flip=True)
    average_inference(path1, path2)

    shutil.rmtree(path1)
    shutil.rmtree(path2)