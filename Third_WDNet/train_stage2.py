import argparse
import os.path
import random
import time
import datetime
import sys
import glob
import cv2
import os

import numpy as np
import paddle
import paddle.distributed as dist
import matplotlib.pylab as plt

from transforms import RandomHorizontalFlip, Normalize
from dataset import Dataset
from net_utils import PAN
from utils import load_pretrained_model, calculate_psnr
from losses import compute_charnonnier_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    parser.add_argument('--dataset_root',dest='dataset_root',help='The path of dataset root',type=str,\
         default='../fine_ture_dataset/')  
    parser.add_argument('--batch_size',dest='batch_size',help='batch_size',type=int,default=8)
    parser.add_argument('--max_epochs',dest='max_epochs',help='max_epochs',type=int,default=2000)
    parser.add_argument('--log_iters',dest='log_iters',help='log_iters',type=int,default=10)
    parser.add_argument('--save_interval',dest='save_interval',help='save_interval',type=int,default=5)
    parser.add_argument('--sample_interval',dest='sample_interval',help='sample_interval',type=int,default=100)
    parser.add_argument('--with_eval',dest='with_eval',help='with eval',type=bool,default=True)
    parser.add_argument('--seed',dest='seed',help='random seed',type=int,default=24)
    return parser.parse_args()


def eval(epoch, dataset_root = '../fine_ture_dataset', pretrained = './train_result/model'):
    print("Eval epoch:{}".format(epoch))
    model = PAN(in_nc=3, out_nc=3, nf=64, unf=64, nb=6)

    if pretrained is not None:
        pretrained = os.path.join(pretrained, "epoch_{}".format(epoch), "model.pdparams")
        print("pretrained: ",pretrained)
        load_pretrained_model(model, pretrained)

    val_partition = ["moire_train_{:05d}.jpg".format(v) for v in range(970, 1000)]
    im_files = glob.glob(os.path.join(dataset_root, "images/*.jpg"))
    im_files = [v for v in im_files if os.path.basename(v) in val_partition]
    im_files.sort()
    
    target_im_files = glob.glob(os.path.join(dataset_root, "gts/*.jpg"))
    target_im_files = [v for v in target_im_files if os.path.basename(v) in val_partition]
    target_im_files.sort()

    psnr = 0.0

    for i, im in enumerate(im_files):
        img = cv2.imread(im)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = paddle.to_tensor(img)
        img /= 255.0
        img = paddle.transpose(img, [2, 0, 1])
        model.eval()

        img = img.unsqueeze(0)
        img_out = model(img)
        img_out = img_out.squeeze(0)

        img_out = img_out * 255.0
        img_out = paddle.clip(img_out, 0, 255)
        img_out = paddle.transpose(img_out, [1, 2, 0])
        img_out = img_out.numpy()

        img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

        # calculate psnr
        target = cv2.imread(target_im_files[i])
        psnr += calculate_psnr(img_out, target)
    
    return psnr / len(im_files)


def main(args):
    paddle.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dist.init_parallel_env()

    transforms = [RandomHorizontalFlip(), Normalize()]

    dataset = Dataset(dataset_root=args.dataset_root, transforms=transforms, val=False, patch_size=256)
    dataloader = paddle.io.DataLoader(dataset, batch_size=args.batch_size,
                                      num_workers=6, shuffle=True, drop_last=True,
                                      return_list=True)


    generator = PAN(in_nc=3, out_nc=3, nf=64, unf=64, nb=6)
    generator.train()

    optimizer = paddle.optimizer.Adam(parameters=generator.parameters(), learning_rate=1e-4,
                                      beta1=0.5, beta2=0.999)

    max_psnr = 0.0
    psnr_list = []
    max_psnr_index = 1

    prev_time = time.time()
    for epoch in range(1, args.max_epochs + 1):
        for i, data_batch in enumerate(dataloader):
            real_A = data_batch[0]    # [b, c, h, w]
            real_B = data_batch[1]    # [b, c, h, w]

            fake_B = generator(real_A)

            loss_pixel = compute_charnonnier_loss(fake_B, real_B)
    
            loss_G = loss_pixel
            loss_G.backward()

            optimizer.step()
            generator.clear_gradients()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = args.max_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            if i % args.log_iters == 0:
                sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [G loss: %f, pixel: %f] ETA: %s" %
                                 (epoch, args.max_epochs, i, len(dataloader), loss_G.numpy()[0], loss_pixel.numpy()[0], time_left))


        if epoch % args.save_interval == 0:
            current_save_dir = os.path.join("train_result", "model", f'epoch_{epoch}')
            if not os.path.exists(current_save_dir):
                try:
                    os.mkdir(current_save_dir)
                except:
                    print("Path already exists.")

            paddle.save(generator.state_dict(),
                        os.path.join(current_save_dir, 'model.pdparams'))
            paddle.save(optimizer.state_dict(),
                        os.path.join(current_save_dir, 'model.pdopt'))
        

        if epoch % args.save_interval == 0 and args.with_eval:
            psnr = eval(epoch)
            psnr_list.append(psnr)

            plt.plot(psnr_list)
            plt.savefig("psnr_list.png")

            if psnr > max_psnr:
                max_psnr = psnr
                max_psnr_index = epoch

            sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] PSNR: [%.4f]" %
                                (epoch, args.max_epochs, i, len(dataloader), psnr))
            sys.stdout.write("\r Max PSNR: [%.4f] || Epoch: [%d]" % (max_psnr, max_psnr_index))


if __name__ == '__main__':
    args = parse_args()
    main(args)