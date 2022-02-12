import os
import glob
import cv2
import paddle
from model import WaveletTransform, WDNet
import paddle.nn as nn
from utils import load_pretrained_model, calculate_psnr


def eval(epoch, dataset_root = '../moire_train_dataset', pretrained = './train_result/model'):
    print("Eval epoch:{}".format(epoch))
    model = WDNet()

    if pretrained is not None:
        pretrained = os.path.join(pretrained, "epoch_{}".format(epoch), "model.pdparams")
        print("pretrained: ",pretrained)
        # load_pretrained_model(model, pretrained)

    wavelet_dec = WaveletTransform(scale=2, dec=True)
    wavelet_rec = WaveletTransform(scale=2, dec=False)

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

        # save_path
        save_path = "output/epoch_{:03d}".format(epoch)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

        if i % 5 == 0:
            print("saving {}".format(os.path.basename(im.split('/')[-1])))
        cv2.imwrite(os.path.join(save_path, im.split('/')[-1]), img_out)

        # calculate psnr
        target = cv2.imread(target_im_files[i])
        psnr += calculate_psnr(img_out, target)
    
    return psnr / len(im_files)


def eval_with_out_model(epoch, dataset_root = '../moire_train_dataset', ressult_save_path='./output'):
    val_partition = ["moire_train_{:05d}.jpg".format(v) for v in range(970, 1000)]
    im_files = glob.glob(os.path.join(ressult_save_path, "epoch_{:03d}/*.jpg".format(epoch)))
    im_files = [v for v in im_files if os.path.basename(v) in val_partition]
    im_files.sort()

    print(im_files[:10])
    
    target_im_files = glob.glob(os.path.join(dataset_root, "gts/*.jpg"))
    target_im_files = [v for v in target_im_files if os.path.basename(v) in val_partition]
    target_im_files.sort()

    print("\n ##########################")
    print(target_im_files[:10])

    psnr = 0.0

    for i, im in enumerate(im_files):
        img = cv2.imread(im)
        
        target = cv2.imread(target_im_files[i])
        psnr += calculate_psnr(img, target)
    
    return psnr / len(im_files)


if __name__ == "__main__":
    psnr = eval(5)
    # psnr = eval_with_out_model(20)
    print(psnr)