import argparse
import os.path
import random
import time
import datetime
import sys

import numpy as np
import paddle

from transforms import ToTensor, Flip
from dataset import Dataset
from MRNET import MRNET
from losses import L1_Charbonnier_loss

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='The path of dataset root',
        type=str,
        default='/data/datasets/moire/baidu/moire_all_patch_1/')

    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='batch_size',
        type=int,
        default=4
    )

    parser.add_argument(
        '--max_epochs',
        dest='max_epochs',
        help='max_epochs',
        type=int,
        default=1000
    )

    parser.add_argument(
        '--log_iters',
        dest='log_iters',
        help='log_iters',
        type=int,
        default=10
    )

    parser.add_argument(
        '--save_interval',
        dest='save_interval',
        help='save_interval',
        type=int,
        default=10
    )

    parser.add_argument(
        '--seed',
        dest='seed',
        help='random seed',
        type=int,
        default=1234
    )

    return parser.parse_args()


def main(args):
    paddle.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    transforms = [
        Flip(),
        ToTensor()
    ]
    dataset = Dataset(dataset_root=args.dataset_root, transforms=transforms)
    dataloader = paddle.io.DataLoader(dataset, batch_size=args.batch_size,
                                      num_workers=4, shuffle=True, drop_last=True,
                                      return_list=True)

    # Loss functions
    criterion_pixelwise = L1_Charbonnier_loss()

    model = MRNET()

    clip = paddle.nn.ClipGradByNorm(clip_norm=20.0)
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=1e-4, T_max=args.max_epochs, eta_min=1e-8)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=scheduler, weight_decay=0.0, grad_clip=clip)

    prev_time = time.time()
    for epoch in range(1, args.max_epochs + 1):
        for i, data_batch in enumerate(dataloader):
            inputs = data_batch[0]
            labels = data_batch[1]

            outputs = model(inputs)

            loss = criterion_pixelwise(outputs, labels)
            loss.backward()

            optimizer.step()
            model.clear_gradients()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = args.max_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            if i % args.log_iters == 0:
                sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %s" %
                                 (epoch, args.max_epochs,
                                  i, len(dataloader),
                                  loss.numpy()[0],
                                  time_left))

        scheduler.step()

        if epoch % args.save_interval == 0:
            current_save_dir = os.path.join("train_result", "model", f'epoch_{epoch}')
            if not os.path.exists(current_save_dir):
                os.makedirs(current_save_dir)
            paddle.save(model.state_dict(),
                        os.path.join(current_save_dir, 'model.pdparams'))
            paddle.save(optimizer.state_dict(),
                        os.path.join(current_save_dir, 'model.pdopt'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
